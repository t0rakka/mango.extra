// Copyright 2026 Distance Technologies Oy. For internal use only.
//
// CUDA port of the wide self-synchronising Huffman decode: 2048-bit chunks,
// 128-wide in-block sync, checkpoints every 256 bits, eight writers per chunk,
// then the same integer IDCT the Vulkan path already matches against the CPU.
//
#include "mango_cuda_launch.hpp"

namespace gpujpeg
{
namespace
{

using Chunk = CudaChunk;
using Seg = CudaSeg;
using Tables = CudaTables;
using Img = CudaImg;

constexpr int kLb = kCudaLb;
constexpr int kLut = kCudaLut;
constexpr int kNcp = kCudaNcp;
constexpr int kPieces = kCudaPieces;
constexpr int kChunkBits = kCudaChunkBits;
constexpr int kSyncThreads = kCudaSyncThreads;

__device__ __forceinline__ unsigned bswap32(unsigned x)
{
    return __byte_perm(x, 0, 0x0123);
}

struct Rd
{
    const unsigned* w;
    unsigned nw;
    unsigned wi, a, b;

    __device__ __forceinline__ unsigned word(unsigned i) const
    {
        return i < nw ? bswap32(__ldg(w + i)) : 0u;
    }

    __device__ __forceinline__ void seek(unsigned pos)
    {
        wi = pos >> 5;
        a = word(wi);
        b = word(wi + 1);
    }

    __device__ __forceinline__ unsigned peek(unsigned pos)
    {
        unsigned at = pos >> 5;
        while (wi != at)
        {
            a = b;
            ++wi;
            b = word(wi + 1);
        }
        return __funnelshift_l(b, a, pos & 31);
    }
};

__device__ __forceinline__ int symEntry(int sym, int len, bool dc)
{
    int s = sym & 15;
    int r = sym >> 4;
    int za = dc ? 1 : (s ? r + 1 : (r == 15 ? 16 : 64));
    return (len + s) | (s << 5) | (za << 9);
}

__device__ __forceinline__ int huffSlow(const int* maxc, const int* voff, const std::uint8_t* hv,
    unsigned bits, int* len)
{
    int l = kLb + 1;
    int code = int(bits >> (32 - l));
    while (l <= 16 && code > maxc[l])
    {
        ++l;
        code = int(bits >> (32 - l));
    }
    if (l > 16)
    {
        *len = 16;
        return -1;
    }
    *len = l;
    return hv[code + voff[l]];
}

struct Tab
{
    const std::uint16_t* lut;
    const int* maxc;
    const int* voff;
    const std::uint8_t* hv;
};

__device__ __forceinline__ int hent(const Tab& t, int ti, unsigned bits, bool dc)
{
    int e = t.lut[(ti << kLb) | int(bits >> (32 - kLb))];
    if (e)
        return e;
    int len = 0;
    int sym = huffSlow(t.maxc + ti * 18, t.voff + ti * 18, t.hv + ti * 256, bits, &len);
    return sym < 0 ? (16 | ((dc ? 1 : 64) << 9)) : symEntry(sym, len, dc);
}

__device__ __forceinline__ int extendMag(unsigned v, int s)
{
    return int(v) < (1 << (s - 1)) ? int(v) + (-1 << s) + 1 : int(v);
}

__device__ __forceinline__ void step(Rd& r, const Tab& t, unsigned& pos, int c, int& z,
    unsigned dcp, unsigned acp, bool write, short* coef, int* dcv)
{
    unsigned bits = r.peek(pos);
    bool dc = z == 0;
    int ti = int(((dc ? dcp : acp) >> (3 * c)) & 7u);
    int h = hent(t, ti, bits, dc);
    int tot = h & 31;
    int s = (h >> 5) & 15;
    int za = (h >> 9) & 127;
    if (write && s)
    {
        int v = extendMag((bits << (tot - s)) >> (32 - s), s);
        int zi = z + za - 1;
        if (dc)
            *dcv = v;
        else if (zi >= 0 && zi < 64 && coef)
            coef[zi] = short(v);
    }
    pos += unsigned(tot);
    z += za;
}

__device__ __forceinline__ bool runChunk(Rd& r, const Tab& t, unsigned& pos, int& c, int& z,
    unsigned dcp, unsigned acp, int bpm, unsigned base, unsigned end,
    unsigned long long* cps, int* cpc, int& n, bool compare)
{
    int done = 0;
    for (int j = 0; j <= kNcp; ++j)
    {
        unsigned mark = j < kNcp
            ? min(base + unsigned((j + 1) * (kChunkBits / kPieces)), end)
            : end;
        int guard = 0;
        while (pos < mark && guard < 4096)
        {
            unsigned before = pos;
            step(r, t, pos, c, z, dcp, acp, false, nullptr, nullptr);
            if (pos == before)
                break;
            if (z >= 64)
            {
                z = 0;
                c = (c + 1 == bpm) ? 0 : c + 1;
                ++done;
            }
            ++guard;
        }
        if (j == kNcp)
            break;
        unsigned long long st = ((unsigned long long)pos << 16) | ((unsigned long long)c << 8) | unsigned(z);
        if (compare && st == cps[j])
        {
            int shift = done - cpc[j];
            for (int k = j; k < kNcp; ++k)
                cpc[k] += shift;
            n += shift;
            return true;
        }
        cps[j] = st;
        cpc[j] = done;
    }
    n = done;
    return false;
}

__device__ __forceinline__ Tab loadTabs(const Tables* src, std::uint16_t* lut, int* maxc, int* voff, std::uint8_t* hv)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    for (int i = tid; i < 8 * kLut; i += nthreads)
        lut[i] = src->lut[i];
    for (int i = tid; i < 8 * 18; i += nthreads)
    {
        maxc[i] = src->maxc[i];
        voff[i] = src->voff[i];
    }
    for (int i = tid; i < 8 * 256; i += nthreads)
        hv[i] = src->hv[i];
    __syncthreads();
    Tab t;
    t.lut = lut;
    t.maxc = maxc;
    t.voff = voff;
    t.hv = hv;
    return t;
}

__global__ void __launch_bounds__(kSyncThreads)
syncPass(const unsigned* words, unsigned nwords, const Chunk* chunks, int nch,
    const Tables* tables, Img img, int pass, int rounds,
    unsigned long long* stIn, unsigned long long* stOut, int* cnt,
    unsigned long long* cps, int* cpc)
{
    __shared__ std::uint16_t lut[8 * kLut];
    __shared__ int maxc[8 * 18];
    __shared__ int voff[8 * 18];
    __shared__ std::uint8_t hv[8 * 256];
    __shared__ unsigned long long so[kSyncThreads];
    Tab tab = loadTabs(tables, lut, maxc, voff, hv);

    int tid = threadIdx.x;
    int i = blockIdx.x * kSyncThreads + tid;
    bool valid = i < nch;
    bool fixed = !valid || chunks[i].bit0 == chunks[i].seg0;
    Rd r;
    r.w = words;
    r.nw = nwords;

    unsigned long long myIn = 0, myOut = 0;
    int n = 0;
    if (valid)
    {
        if (pass == 0)
        {
            unsigned pos = chunks[i].bit0;
            int c = 0, z = 0;
            myIn = ((unsigned long long)pos << 16);
            r.seek(pos);
            runChunk(r, tab, pos, c, z, img.dc_pack, img.ac_pack, img.bpm,
                chunks[i].bit0, chunks[i].bit1, cps + i * kNcp, cpc + i * kNcp, n, false);
            myOut = ((unsigned long long)pos << 16) | ((unsigned long long)c << 8) | unsigned(z);
        }
        else
        {
            myIn = stIn[i];
            myOut = stOut[i];
            n = cnt[i];
        }
    }
    unsigned long long globPrev = (!pass && false) ? 0 : 0;
    if (pass > 0 && valid && tid == 0 && !fixed)
        globPrev = stOut[i - 1];
    so[tid] = myOut;
    __syncthreads();

    for (int it = 0; it < rounds; ++it)
    {
        bool ch = false;
        if (valid && !fixed && (tid > 0 || pass > 0))
        {
            unsigned long long want = tid == 0 ? globPrev : so[tid - 1];
            bool sameSeg = tid == 0
                ? (i > 0 && chunks[i].seg0 == chunks[i - 1].seg0)
                : (chunks[i].seg0 == chunks[i - 1].seg0);
            if (tid > 0 && !sameSeg)
                want = myIn;
            if (sameSeg && want != myIn)
            {
                myIn = want;
                unsigned pos = unsigned(want >> 16);
                int c = int((want >> 8) & 255);
                int z = int(want & 255);
                r.seek(pos);
                bool same = runChunk(r, tab, pos, c, z, img.dc_pack, img.ac_pack, img.bpm,
                    chunks[i].bit0, chunks[i].bit1, cps + i * kNcp, cpc + i * kNcp, n, true);
                unsigned long long e = same
                    ? myOut
                    : ((unsigned long long)pos << 16) | ((unsigned long long)c << 8) | unsigned(z);
                if (e != myOut)
                {
                    myOut = e;
                    ch = true;
                }
            }
        }
        __syncthreads();
        so[tid] = myOut;
        if (!__syncthreads_or(ch))
            break;
    }
    if (valid)
    {
        stIn[i] = myIn;
        stOut[i] = myOut;
        cnt[i] = n;
    }
}

__global__ void prefixBlocks(const Seg* segs, const int* cnt, int* firstb)
{
    Seg s = segs[blockIdx.x];
    __shared__ int sh[256];
    __shared__ int run;
    if (threadIdx.x == 0)
        run = 0;
    __syncthreads();
    for (int tile = 0; tile < s.nchunks; tile += 256)
    {
        int ci = tile + threadIdx.x;
        int n = ci < s.nchunks ? cnt[s.chunk0 + ci] : 0;
        int self = n;
        sh[threadIdx.x] = n;
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1)
        {
            int v = sh[threadIdx.x];
            if (threadIdx.x >= off)
                v += sh[threadIdx.x - off];
            __syncthreads();
            sh[threadIdx.x] = v;
            __syncthreads();
        }
        if (ci < s.nchunks)
            firstb[s.chunk0 + ci] = s.block0 + run + sh[threadIdx.x] - self;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int last = min(255, s.nchunks - tile - 1);
            run += sh[last];
        }
        __syncthreads();
    }
}

__global__ void decodeWrite(const unsigned* words, unsigned nwords, const Chunk* chunks, int nch,
    const Tables* tables, Img img, const unsigned long long* stIn, const unsigned long long* cps,
    const int* cpc, const int* firstb, short* coef, int* dcd)
{
    __shared__ std::uint16_t lut[8 * kLut];
    __shared__ int maxc[8 * 18];
    __shared__ int voff[8 * 18];
    __shared__ std::uint8_t hv[8 * 256];
    Tab tab = loadTabs(tables, lut, maxc, voff, hv);

    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gi / kPieces;
    int piece = gi % kPieces;
    if (i >= nch)
        return;
    Chunk c = chunks[i];
    unsigned long long s = piece == 0 ? stIn[i] : cps[i * kNcp + piece - 1];
    unsigned pos = unsigned(s >> 16);
    int bc = int((s >> 8) & 255);
    int z = int(s & 255);
    int B = firstb[i] + (piece == 0 ? 0 : cpc[i * kNcp + piece - 1]);
    unsigned mark = piece < kNcp
        ? min(c.bit0 + unsigned((piece + 1) * (kChunkBits / kPieces)), c.bit1)
        : c.bit1;
    Rd r;
    r.w = words;
    r.nw = nwords;
    r.seek(pos);
    if (z != 0)
    {
        int guard = 0;
        while (z < 64 && guard < 256)
        {
            unsigned before = pos;
            step(r, tab, pos, bc, z, img.dc_pack, img.ac_pack, false, nullptr, nullptr);
            if (pos == before)
                break;
            ++guard;
        }
        z = 0;
        bc = (bc + 1 == img.bpm) ? 0 : bc + 1;
        ++B;
    }
    int guard = 0;
    while (pos < mark && B < c.block1 && guard < 4096)
    {
        short* row = coef + std::size_t(B) * 64;
        int dcv = 0;
        do
        {
            unsigned before = pos;
            step(r, tab, pos, bc, z, img.dc_pack, img.ac_pack, true, row, &dcv);
            if (pos == before)
                break;
        } while (z < 64);
        dcd[B] = dcv;
        z = 0;
        bc = (bc + 1 == img.bpm) ? 0 : bc + 1;
        ++B;
        ++guard;
    }
}

__global__ void dcSegment(const Seg* segs, int* dcd, int bpm, unsigned comp)
{
    Seg s = segs[blockIdx.x];
    __shared__ int s0[256], s1[256], s2[256];
    int per = (s.mcus + 255) / 256;
    int m0 = threadIdx.x * per;
    int m1 = min(m0 + per, s.mcus);
    int a0 = 0, a1 = 0, a2 = 0;
    for (int m = m0; m < m1; ++m)
    {
        const int* p = dcd + std::size_t(s.mcu0 + m) * bpm;
        for (int k = 0; k < bpm; ++k)
        {
            int ci = int((comp >> (2 * k)) & 3u);
            int v = p[k];
            a0 += ci == 0 ? v : 0;
            a1 += ci == 1 ? v : 0;
            a2 += ci == 2 ? v : 0;
        }
    }
    s0[threadIdx.x] = a0;
    s1[threadIdx.x] = a1;
    s2[threadIdx.x] = a2;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1)
    {
        int b0 = s0[threadIdx.x], b1 = s1[threadIdx.x], b2 = s2[threadIdx.x];
        if (threadIdx.x >= off)
        {
            b0 += s0[threadIdx.x - off];
            b1 += s1[threadIdx.x - off];
            b2 += s2[threadIdx.x - off];
        }
        __syncthreads();
        s0[threadIdx.x] = b0;
        s1[threadIdx.x] = b1;
        s2[threadIdx.x] = b2;
        __syncthreads();
    }
    int p0 = s0[threadIdx.x] - a0;
    int p1 = s1[threadIdx.x] - a1;
    int p2 = s2[threadIdx.x] - a2;
    for (int m = m0; m < m1; ++m)
    {
        int* p = dcd + std::size_t(s.mcu0 + m) * bpm;
        for (int k = 0; k < bpm; ++k)
        {
            int ci = int((comp >> (2 * k)) & 3u);
            int v = p[k];
            if (ci == 0) { p0 += v; p[k] = p0; }
            else if (ci == 1) { p1 += v; p[k] = p1; }
            else { p2 += v; p[k] = p2; }
        }
    }
}

__device__ __forceinline__ void idctPass(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7,
    int& x0, int& x1, int& x2, int& x3, int& y0, int& y1, int& y2, int& y3)
{
    int n0 = (s2 + s6) * 2217;
    int t2 = n0 + s6 * (-7567);
    int t3 = n0 + s2 * 3135;
    int t0 = (s0 + s4) << 12;
    int t1 = (s0 - s4) << 12;
    x0 = t0 + t3;
    x3 = t0 - t3;
    x1 = t1 + t2;
    x2 = t1 - t2;
    int p1 = s7 + s1;
    int p2 = s5 + s3;
    int p3 = s7 + s3;
    int p4 = s5 + s1;
    int p5 = (p3 + p4) * 4816;
    p1 = p1 * (-3685) + p5;
    p2 = p2 * (-10497) + p5;
    p3 = p3 * (-8034);
    p4 = p4 * (-1597);
    y0 = p1 + p3 + s7 * 1223;
    y1 = p2 + p4 + s5 * 8410;
    y2 = p2 + p3 + s3 * 12586;
    y3 = p1 + p4 + s1 * 6149;
}

__device__ __forceinline__ int clamp8(int v)
{
    return min(max(v, 0), 255);
}

__device__ __forceinline__ int sampleAt(const int* nat, const Img& img, int mcuLocal, int first, int hsf, int vsf, int lx, int ly)
{
    int blockW = max(img.xblock / max(hsf, 1), 1);
    int blockH = max(img.yblock / max(vsf, 1), 1);
    int bx = min(lx / blockW, hsf - 1);
    int by = min(ly / blockH, vsf - 1);
    int spx = ((lx % blockW) * 8) / blockW;
    int spy = ((ly % blockH) * 8) / blockH;
    spx = min(max(spx, 0), 7);
    spy = min(max(spy, 0), 7);
    int block = mcuLocal * img.bpm + first + by * hsf + bx;
    return nat[block * 64 + spy * 8 + spx];
}

__global__ void idctColor(const short* coef, const int* dcd, const int* qt, Img img, unsigned char* rgba)
{
    __shared__ int nat[32 * 64];
    __shared__ int tmp[32 * 64];
    const int zigzag[64] = {
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    };

    int lane = threadIdx.x;
    int slot = lane >> 3;
    int col = lane & 7;
    int group0 = blockIdx.x * img.mcus_per_group;
    int used = img.mcus_per_group * img.bpm;
    int nmcu = img.xmcu * ((img.height + img.yblock - 1) / img.yblock);
    bool live = false;
    int mcuLocal = 0, b = 0;
    if (slot < used)
    {
        mcuLocal = slot / img.bpm;
        b = slot - mcuLocal * img.bpm;
        live = (group0 + mcuLocal) < nmcu;
    }
    if (live)
    {
        int global = group0 + mcuLocal;
        int base = (global * img.bpm + b) * 64;
        int dst = slot * 64;
        for (int i = col; i < 64; i += 8)
        {
            int v = coef[base + i];
            if (i == 0)
                v = dcd[global * img.bpm + b];
            nat[dst + zigzag[i]] = v;
        }
    }
    __syncthreads();
    if (live)
    {
        int dst = slot * 64;
        int qsel = 0;
        if (b >= img.y_off && b < img.y_off + img.y_hsf * img.y_vsf)
            qsel = img.y_qt;
        else if (img.components > 1 && b >= img.cb_off && b < img.cb_off + img.cb_hsf * img.cb_vsf)
            qsel = img.cb_qt;
        else if (img.components > 2 && b >= img.cr_off && b < img.cr_off + img.cr_hsf * img.cr_vsf)
            qsel = img.cr_qt;
        int qbase = qsel * 64;
        int n0 = nat[dst + col];
        int n1 = nat[dst + col + 8];
        int n2 = nat[dst + col + 16];
        int n3 = nat[dst + col + 24];
        int n4 = nat[dst + col + 32];
        int n5 = nat[dst + col + 40];
        int n6 = nat[dst + col + 48];
        int n7 = nat[dst + col + 56];
        if ((n1 | n2 | n3 | n4 | n5 | n6 | n7) != 0)
        {
            int x0, x1, x2, x3, y0, y1, y2, y3;
            idctPass(
                n0 * qt[qbase + col], n1 * qt[qbase + col + 8], n2 * qt[qbase + col + 16], n3 * qt[qbase + col + 24],
                n4 * qt[qbase + col + 32], n5 * qt[qbase + col + 40], n6 * qt[qbase + col + 48], n7 * qt[qbase + col + 56],
                x0, x1, x2, x3, y0, y1, y2, y3);
            const int bias = 0x200;
            x0 += bias; x1 += bias; x2 += bias; x3 += bias;
            tmp[dst + col * 8 + 0] = (x0 + y3) >> 10;
            tmp[dst + col * 8 + 1] = (x1 + y2) >> 10;
            tmp[dst + col * 8 + 2] = (x2 + y1) >> 10;
            tmp[dst + col * 8 + 3] = (x3 + y0) >> 10;
            tmp[dst + col * 8 + 4] = (x3 - y0) >> 10;
            tmp[dst + col * 8 + 5] = (x2 - y1) >> 10;
            tmp[dst + col * 8 + 6] = (x1 - y2) >> 10;
            tmp[dst + col * 8 + 7] = (x0 - y3) >> 10;
        }
        else
        {
            int dc = (n0 * qt[qbase + col]) << 2;
            for (int k = 0; k < 8; ++k)
                tmp[dst + col * 8 + k] = dc;
        }
    }
    __syncthreads();
    if (live)
    {
        int dst = slot * 64;
        int row = col;
        int x0, x1, x2, x3, y0, y1, y2, y3;
        idctPass(
            tmp[dst + row], tmp[dst + row + 8], tmp[dst + row + 16], tmp[dst + row + 24],
            tmp[dst + row + 32], tmp[dst + row + 40], tmp[dst + row + 48], tmp[dst + row + 56],
            x0, x1, x2, x3, y0, y1, y2, y3);
        const int shift = 17;
        const int bias = 0x10000 + (128 << shift);
        x0 += bias; x1 += bias; x2 += bias; x3 += bias;
        nat[dst + row * 8 + 0] = clamp8((x0 + y3) >> shift);
        nat[dst + row * 8 + 1] = clamp8((x1 + y2) >> shift);
        nat[dst + row * 8 + 2] = clamp8((x2 + y1) >> shift);
        nat[dst + row * 8 + 3] = clamp8((x3 + y0) >> shift);
        nat[dst + row * 8 + 4] = clamp8((x3 - y0) >> shift);
        nat[dst + row * 8 + 5] = clamp8((x2 - y1) >> shift);
        nat[dst + row * 8 + 6] = clamp8((x1 - y2) >> shift);
        nat[dst + row * 8 + 7] = clamp8((x0 - y3) >> shift);
    }
    __syncthreads();

    int area = img.xblock * img.yblock;
    int span = img.mcus_per_group * area;
    int ymcu = (img.height + img.yblock - 1) / img.yblock;
    for (int p = lane; p < span; p += 256)
    {
        int mcuI = p / area;
        int pix = p - mcuI * area;
        int global = group0 + mcuI;
        if (global >= ymcu * img.xmcu)
            continue;
        int lx = pix % img.xblock;
        int ly = pix / img.xblock;
        int mcuX = global % img.xmcu;
        int mcuY = global / img.xmcu;
        int px = mcuX * img.xblock + lx;
        int py = mcuY * img.yblock + ly;
        if (px >= img.width || py >= img.height)
            continue;
        int Y = sampleAt(nat, img, mcuI, img.y_off, img.y_hsf, img.y_vsf, lx, ly);
        int R, G, B;
        if (img.components == 1)
            R = G = B = Y;
        else
        {
            int Cb = sampleAt(nat, img, mcuI, img.cb_off, img.cb_hsf, img.cb_vsf, lx, ly) - 128;
            int Cr = sampleAt(nat, img, mcuI, img.cr_off, img.cr_hsf, img.cr_vsf, lx, ly) - 128;
            R = clamp8(Y + ((Cr * 91881) >> 16));
            G = clamp8(Y - ((Cb * 22554 + Cr * 46802) >> 16));
            B = clamp8(Y + ((Cb * 116130) >> 16));
        }
        unsigned char* out = rgba + (std::size_t(py) * img.width + px) * 4;
        out[0] = std::uint8_t(R);
        out[1] = std::uint8_t(G);
        out[2] = std::uint8_t(B);
        out[3] = 255;
    }
}

} // namespace

void mangoCudaHuffman(const CudaFrame& f)
{
    cudaStream_t st = f.stream;
    int groups = (f.nchunks + kCudaSyncThreads - 1) / kCudaSyncThreads;
    for (int pass = 0; pass < kCudaPasses; ++pass)
    {
        syncPass<<<groups, kCudaSyncThreads, 0, st>>>(
            f.words, f.nwords, f.chunks, f.nchunks, f.tables, f.img, pass, kCudaRounds,
            f.stIn, f.stOut, f.cnt, f.cps, f.cpc);
    }
    prefixBlocks<<<f.nsegs, 256, 0, st>>>(f.segs, f.cnt, f.firstb);
    int writers = f.nchunks * kCudaPieces;
    int wgroups = (writers + 127) / 128;
    decodeWrite<<<wgroups, 128, 0, st>>>(
        f.words, f.nwords, f.chunks, f.nchunks, f.tables, f.img,
        f.stIn, f.cps, f.cpc, f.firstb, f.coef, f.dcd);
    dcSegment<<<f.nsegs, 256, 0, st>>>(f.segs, f.dcd, f.img.bpm, f.img.comp_pack);
}

void mangoCudaPixels(const CudaFrame& f)
{
    cudaStream_t st = f.stream;
    int nmcu = f.img.xmcu * ((f.img.height + f.img.yblock - 1) / f.img.yblock);
    int igroups = (nmcu + f.img.mcus_per_group - 1) / f.img.mcus_per_group;
    idctColor<<<igroups, 256, 0, st>>>(f.coef, f.dcd, f.qt, f.img, f.rgba);
}

} // namespace gpujpeg

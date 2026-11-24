// Copyright 2026 Distance Technologies Oy. For internal use only.
//
#include "mango_gpu2_backend.hpp"
#include "decoded_texture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <mango/vulkan/compiler.hpp>

namespace gpujpeg
{
namespace
{

constexpr int kLb = 10;
constexpr int kLut = 1 << kLb;
constexpr int kNcp = 7;
constexpr int kPieces = kNcp + 1;
constexpr std::uint32_t kChunkBits = 2048;
constexpr int kMaxPasses = 8;
constexpr int kSyncRounds = 128;

constexpr int kBindings = 17;
constexpr std::uint32_t kImageBinding = 14;
constexpr int kDcTile = 1024; // MCUs per DC scan workgroup: 256 threads x 4

enum Pipe { PipeSync, PipePrefix, PipeWrite, PipeDcLocal, PipeDcGroups, PipeIdct, PipeCount };

struct ChunkGpu
{
    std::uint32_t bit0;
    std::uint32_t bit1;
    std::uint32_t seg0;
    std::uint32_t seg1;
    std::int32_t block0;
    std::int32_t block1;
};

struct SegGpu
{
    std::int32_t chunk0;
    std::int32_t nchunks;
    std::int32_t block0;
    std::int32_t mcu0;
    std::int32_t mcus;
};

// 16-bit LUT entries two per word, Huffman values four per word: the whole
// thing fits in shared memory next to the sync state.
struct TablesGpu
{
    std::uint32_t lut[8 * kLut / 2];
    std::int32_t maxc[8 * 18];
    std::int32_t voff[8 * 18];
    std::uint32_t hv[8 * 256 / 4];
};

struct Push
{
    std::int32_t nch;
    std::int32_t pass;
    std::int32_t rounds;
    std::int32_t bpm;
    std::uint32_t dc_pack;
    std::uint32_t ac_pack;
    std::uint32_t comp_pack;
    std::uint32_t nwords;
    std::int32_t width, height, xmcu, xblock, yblock, components;
    std::int32_t y_hsf, y_vsf, y_qt, y_off;
    std::int32_t cb_hsf, cb_vsf, cb_qt, cb_off;
    std::int32_t cr_hsf, cr_vsf, cr_qt, cr_off;
    std::int32_t mcus_per_group;
    std::int32_t nmcu;
};
static_assert(sizeof(Push) <= 128, "push constants");

constexpr char kCommon[] = R"glsl(
#version 450
layout(std430, binding = 0) readonly buffer Words { uint words[]; };
struct Chunk { uint bit0; uint bit1; uint seg0; uint seg1; int block0; int block1; };
layout(std430, binding = 1) readonly buffer Chunks { Chunk chunks[]; };
struct Seg { int chunk0; int nchunks; int block0; int mcu0; int mcus; };
layout(std430, binding = 2) readonly buffer Segs { Seg segs[]; };
layout(std430, binding = 3) readonly buffer Tables { uint tlut[4096]; int tmaxc[144]; int tvoff[144]; uint thv[512]; };
layout(std430, binding = 4) buffer StIn { uvec2 stIn[]; };
layout(std430, binding = 5) buffer StOut { uvec2 stOut[]; };
layout(std430, binding = 6) buffer Cnt { int cnt[]; };
layout(std430, binding = 7) buffer Cps { uvec2 cps[]; };
layout(std430, binding = 8) buffer Cpc { int cpc[]; };
layout(std430, binding = 9) buffer First { int firstb[]; };
layout(std430, binding = 10) buffer Coef { uint coef[]; };
layout(std430, binding = 11) buffer Dcd { int dcd[]; };
layout(std430, binding = 12) readonly buffer Qt { int qt[256]; };
layout(std430, binding = 13) buffer Flags { uint flags[]; };
layout(std430, binding = 15) readonly buffer SegStart { uint segStart[]; };
layout(std430, binding = 16) buffer Agg { ivec4 agg[]; };

layout(push_constant) uniform Push {
    int nch;
    int pass;
    int rounds;
    int bpm;
    uint dcPack;
    uint acPack;
    uint compPack;
    uint nwords;
    int width, height, xmcu, xblock, yblock, components;
    int y_hsf, y_vsf, y_qt, y_off;
    int cb_hsf, cb_vsf, cb_qt, cb_off;
    int cr_hsf, cr_vsf, cr_qt, cr_off;
    int mcusPerGroup;
    int nmcu;
} pc;
)glsl";

constexpr char kHuffman[] = R"glsl(
shared uint slut[4096];
shared int smaxc[144];
shared int svoff[144];
shared uint shv[512];

void loadTabs()
{
    uint t = gl_LocalInvocationIndex;
    uint n = gl_WorkGroupSize.x;
    for (uint i = t; i < 4096u; i += n)
        slut[i] = tlut[i];
    for (uint i = t; i < 144u; i += n)
    {
        smaxc[i] = tmaxc[i];
        svoff[i] = tvoff[i];
    }
    for (uint i = t; i < 512u; i += n)
        shv[i] = thv[i];
    barrier();
}

struct Rd { uint wi; uint a; uint b; };

uint wordAt(uint i)
{
    return i < pc.nwords ? words[i] : 0u;
}

void seek(inout Rd r, uint pos)
{
    r.wi = pos >> 5;
    r.a = wordAt(r.wi);
    r.b = wordAt(r.wi + 1u);
}

uint peek(inout Rd r, uint pos)
{
    uint at = pos >> 5;
    while (r.wi != at)
    {
        r.a = r.b;
        r.wi++;
        r.b = wordAt(r.wi + 1u);
    }
    uint n = pos & 31u;
    return n == 0u ? r.a : ((r.a << n) | (r.b >> (32u - n)));
}

int symEntry(int sym, int len, bool dc)
{
    int s = sym & 15;
    int r = sym >> 4;
    int za = dc ? 1 : (s != 0 ? r + 1 : (r == 15 ? 16 : 64));
    return (len + s) | (s << 5) | (za << 9);
}

int hent(int ti, uint bits, bool dc)
{
    int idx = (ti << 10) | int(bits >> 22);
    uint w = slut[idx >> 1];
    int e = int((idx & 1) != 0 ? (w >> 16) : (w & 0xffffu));
    if (e != 0)
        return e;
    int l = 11;
    int code = int(bits >> 21);
    while (l <= 16 && code > smaxc[ti * 18 + l])
    {
        ++l;
        code = int(bits >> uint(32 - l));
    }
    if (l > 16)
        return 16 | ((dc ? 1 : 64) << 9);
    int k = ti * 256 + code + svoff[ti * 18 + l];
    int sym = int((shv[k >> 2] >> (uint(k & 3) * 8u)) & 0xffu);
    return symEntry(sym, l, dc);
}

int tableOf(int c, bool dc)
{
    return int(((dc ? pc.dcPack : pc.acPack) >> (3u * uint(c))) & 7u);
}

void stepSkip(inout Rd r, inout uint pos, int c, inout int z)
{
    uint bits = peek(r, pos);
    bool dc = z == 0;
    int h = hent(tableOf(c, dc), bits, dc);
    pos += uint(h & 31);
    z += (h >> 9) & 127;
}

int nextBlock(int c)
{
    return c + 1 == pc.bpm ? 0 : c + 1;
}

uvec2 pack(uint pos, int c, int z)
{
    return uvec2(pos, uint((c << 8) | z));
}
)glsl";

constexpr char kSyncMain[] = R"glsl(
layout(local_size_x = 128) in;

shared uvec2 so[128];
shared uint anyCh;

// Walk from (pos, c, z) to end, recording the state at the first symbol
// boundary past each 256-bit mark. With cmp, stop at the first mark whose
// state matches this chunk's previous walk and shift the later block counts.
bool runChunk(inout Rd r, inout uint pos, inout int c, inout int z, uint base, uint end,
              uint cpBase, inout int n, bool cmp)
{
    int done = 0;
    for (int j = 0; j <= 7; ++j)
    {
        uint mark = j < 7 ? min(base + uint((j + 1) * 256), end) : end;
        while (pos < mark)
        {
            stepSkip(r, pos, c, z);
            if (z >= 64)
            {
                z = 0;
                c = nextBlock(c);
                ++done;
            }
        }
        if (j == 7)
            break;
        uvec2 st = pack(pos, c, z);
        uint at = cpBase + uint(j);
        if (cmp && st == cps[at])
        {
            int shift = done - cpc[at];
            for (int k = j; k < 7; ++k)
                cpc[cpBase + uint(k)] += shift;
            n += shift;
            return true;
        }
        cps[at] = st;
        cpc[at] = done;
    }
    n = done;
    return false;
}

void main()
{
    if (pc.pass > 0 && flags[pc.pass - 1] == 0u)
        return;
    loadTabs();

    int tid = int(gl_LocalInvocationID.x);
    int i = int(gl_WorkGroupID.x) * 128 + tid;
    bool valid = i < pc.nch;
    bool known = !valid || chunks[i].bit0 == chunks[i].seg0;
    uint cpBase = uint(max(i, 0)) * 7u;

    Rd r;
    uvec2 myIn = uvec2(0u);
    uvec2 myOut = uvec2(0u);
    int n = 0;
    if (valid)
    {
        if (pc.pass == 0)
        {
            uint pos = chunks[i].bit0;
            int c = 0;
            int z = 0;
            myIn = pack(pos, 0, 0);
            seek(r, pos);
            runChunk(r, pos, c, z, chunks[i].bit0, chunks[i].bit1, cpBase, n, false);
            myOut = pack(pos, c, z);
        }
        else
        {
            myIn = stIn[i];
            myOut = stOut[i];
            n = cnt[i];
        }
    }
    uvec2 out0 = myOut;
    uvec2 globPrev = uvec2(0u);
    if (pc.pass > 0 && valid && tid == 0 && !known)
        globPrev = stOut[i - 1];
    so[tid] = myOut;
    if (tid == 0)
        anyCh = 0u;
    barrier();

    bool more = false;
    for (int it = 0; it < pc.rounds; ++it)
    {
        bool ch = false;
        if (valid && !known && (tid > 0 || pc.pass > 0))
        {
            uvec2 want = tid == 0 ? globPrev : so[tid - 1];
            if (want != myIn)
            {
                myIn = want;
                uint pos = want.x;
                int c = int((want.y >> 8) & 255u);
                int z = int(want.y & 255u);
                seek(r, pos);
                bool same = runChunk(r, pos, c, z, chunks[i].bit0, chunks[i].bit1, cpBase, n, true);
                uvec2 e = same ? myOut : pack(pos, c, z);
                if (e != myOut)
                {
                    myOut = e;
                    ch = true;
                }
            }
        }
        barrier();
        so[tid] = myOut;
        if (ch)
            anyCh = 1u;
        barrier();
        more = anyCh != 0u;
        barrier();
        if (tid == 0)
            anyCh = 0u;
        if (!more)
            break;
    }
    if (valid)
    {
        stIn[i] = myIn;
        stOut[i] = myOut;
        cnt[i] = n;
    }
    int last = min(128, pc.nch - int(gl_WorkGroupID.x) * 128) - 1;
    if (tid == last && (myOut != out0 || more || (pc.pass == 0 && gl_NumWorkGroups.x > 1u)))
        flags[pc.pass] = 1u;
}
)glsl";

constexpr char kPrefixMain[] = R"glsl(
layout(local_size_x = 256) in;

shared int sh[256];
shared int run;

void main()
{
    Seg s = segs[gl_WorkGroupID.x];
    uint t = gl_LocalInvocationID.x;
    if (t == 0u)
        run = 0;
    barrier();
    for (int tile = 0; tile < s.nchunks; tile += 256)
    {
        int ci = tile + int(t);
        int n = ci < s.nchunks ? cnt[s.chunk0 + ci] : 0;
        sh[t] = n;
        barrier();
        for (uint off = 1u; off < 256u; off <<= 1u)
        {
            int v = sh[t];
            if (t >= off)
                v += sh[t - off];
            barrier();
            sh[t] = v;
            barrier();
        }
        if (ci < s.nchunks)
            firstb[s.chunk0 + ci] = s.block0 + run + sh[t] - n;
        barrier();
        if (t == 0u)
            run += sh[min(255, s.nchunks - tile - 1)];
        barrier();
    }
}
)glsl";

constexpr char kWriteMain[] = R"glsl(
layout(local_size_x = 128) in;

// A block belongs to the piece its DC symbol starts in: a piece first skips
// the rest of a block begun before it, and finishes its own last block past
// its end. AC values go out two per word as they pass; the buffer is zeroed.
void main()
{
    loadTabs();
    int gi = int(gl_GlobalInvocationID.x);
    int i = gi >> 3;
    int piece = gi & 7;
    if (i >= pc.nch)
        return;
    Chunk ch = chunks[i];
    uvec2 s = piece == 0 ? stIn[i] : cps[uint(i) * 7u + uint(piece - 1)];
    uint pos = s.x;
    int bc = int((s.y >> 8) & 255u);
    int z = int(s.y & 255u);
    int B = firstb[i] + (piece == 0 ? 0 : cpc[i * 7 + piece - 1]);
    uint mark = piece < 7 ? min(ch.bit0 + uint((piece + 1) * 256), ch.bit1) : ch.bit1;

    Rd r;
    seek(r, pos);
    if (z != 0)
    {
        while (z < 64)
            stepSkip(r, pos, bc, z);
        z = 0;
        bc = nextBlock(bc);
        ++B;
    }
    while (pos < mark && B < ch.block1)
    {
        uint base = uint(B) * 32u;
        int dcv = 0;
        uint pend = 0u;
        int pq = -1;
        do
        {
            uint bits = peek(r, pos);
            bool dc = z == 0;
            int h = hent(tableOf(bc, dc), bits, dc);
            int tot = h & 31;
            int sz = (h >> 5) & 15;
            int za = (h >> 9) & 127;
            if (sz != 0)
            {
                int v = int((bits << uint(tot - sz)) >> uint(32 - sz));
                if (v < (1 << (sz - 1)))
                    v = v - (1 << sz) + 1;
                int zi = z + za - 1;
                if (dc)
                    dcv = v;
                else if (zi < 64)
                {
                    int q = zi >> 1;
                    if (q != pq)
                    {
                        if (pq >= 0)
                            coef[base + uint(pq)] = pend;
                        pend = 0u;
                        pq = q;
                    }
                    pend |= (uint(v) & 0xffffu) << (uint(zi & 1) * 16u);
                }
            }
            pos += uint(tot);
            z += za;
        } while (z < 64);
        if (pq >= 0)
            coef[base + uint(pq)] = pend;
        dcd[B] = dcv;
        z = 0;
        bc = nextBlock(bc);
        ++B;
    }
}
)glsl";

constexpr char kDcScan[] = R"glsl(
layout(local_size_x = 256) in;

// Segmented scan element: x = a segment starts inside, yzw = DC sums per
// component after the last start. a comes before b.
ivec4 combine(ivec4 a, ivec4 b)
{
    return b.x != 0 ? b : ivec4(a.x, a.yzw + b.yzw);
}

bool isStart(int m)
{
    return ((segStart[m >> 5] >> uint(m & 31)) & 1u) != 0u;
}

int compOf(int k)
{
    return int((pc.compPack >> (2u * uint(k))) & 3u);
}

shared ivec4 sa[256];

ivec4 scan256(ivec4 v, uint t)
{
    sa[t] = v;
    barrier();
    for (uint off = 1u; off < 256u; off <<= 1u)
    {
        ivec4 x = sa[t];
        if (t >= off)
            x = combine(sa[t - off], x);
        barrier();
        sa[t] = x;
        barrier();
    }
    return sa[t];
}
)glsl";

// pass 0: per-tile totals into agg[group]. pass 1: absolute DC values.
constexpr char kDcLocalMain[] = R"glsl(
void main()
{
    uint t = gl_LocalInvocationID.x;
    int g = int(gl_WorkGroupID.x);
    int m0 = g * 1024 + int(t) * 4;
    int m1 = min(m0 + 4, pc.nmcu);
    ivec4 acc = ivec4(0);
    for (int m = m0; m < m1; ++m)
    {
        if (isStart(m))
            acc = ivec4(1, 0, 0, 0);
        int p = m * pc.bpm;
        for (int k = 0; k < pc.bpm; ++k)
        {
            int ci = compOf(k);
            int v = dcd[p + k];
            if (ci == 0) acc.y += v;
            else if (ci == 1) acc.z += v;
            else acc.w += v;
        }
    }
    ivec4 incl = scan256(acc, t);
    if (pc.pass == 0)
    {
        if (t == 255u)
            agg[g] = incl;
        return;
    }
    ivec4 excl = t == 0u ? ivec4(0) : sa[t - 1u];
    ivec4 run = combine(agg[int(gl_NumWorkGroups.x) + g], excl);
    for (int m = m0; m < m1; ++m)
    {
        if (isStart(m))
            run = ivec4(1, 0, 0, 0);
        int p = m * pc.bpm;
        for (int k = 0; k < pc.bpm; ++k)
        {
            int ci = compOf(k);
            int v = dcd[p + k];
            if (ci == 0) { run.y += v; dcd[p + k] = run.y; }
            else if (ci == 1) { run.z += v; dcd[p + k] = run.z; }
            else { run.w += v; dcd[p + k] = run.w; }
        }
    }
}
)glsl";

// One workgroup: exclusive prefix of the tile totals into agg[ngroups + g].
constexpr char kDcGroupsMain[] = R"glsl(
shared ivec4 carry;

void main()
{
    uint t = gl_LocalInvocationID.x;
    int ngroups = pc.nch;
    if (t == 0u)
        carry = ivec4(0);
    barrier();
    for (int tile = 0; tile < ngroups; tile += 256)
    {
        int g = tile + int(t);
        ivec4 v = g < ngroups ? agg[g] : ivec4(0);
        ivec4 incl = scan256(v, t);
        ivec4 excl = t == 0u ? ivec4(0) : sa[t - 1u];
        if (g < ngroups)
            agg[ngroups + g] = combine(carry, excl);
        barrier();
        if (t == 0u)
            carry = combine(carry, sa[255]);
        barrier();
    }
}
)glsl";

constexpr char kIdctMain[] = R"glsl(
layout(local_size_x = 256) in;
#ifdef LINEAR_OUT
layout(std430, binding = 14) writeonly buffer Rgba { uint rgbaOut[]; };
#else
layout(binding = 14, rgba8) uniform writeonly image2D uOut;
#endif

const int zigzag[64] = int[](
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
);

// 32 block slots. nat holds dezigzag'd coeffs, then the finished samples.
shared int nat[32 * 64];
shared int tmp[32 * 64];

void idct_pass(inout int x0, inout int x1, inout int x2, inout int x3,
               inout int y0, inout int y1, inout int y2, inout int y3,
               int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7)
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

int clamp_u8(int v)
{
    return clamp(v, 0, 255);
}

int qt_index_of(int b)
{
    if (b >= pc.y_off && b < pc.y_off + pc.y_hsf * pc.y_vsf)
        return pc.y_qt;
    if (pc.components > 1 && b >= pc.cb_off && b < pc.cb_off + pc.cb_hsf * pc.cb_vsf)
        return pc.cb_qt;
    if (pc.components > 2 && b >= pc.cr_off && b < pc.cr_off + pc.cr_hsf * pc.cr_vsf)
        return pc.cr_qt;
    return 0;
}

int sample_at(int mcu_local, int first_block, int hsf, int vsf, int lx, int ly)
{
    int block_w = max(pc.xblock / max(hsf, 1), 1);
    int block_h = max(pc.yblock / max(vsf, 1), 1);
    int bx = min(lx / block_w, hsf - 1);
    int by = min(ly / block_h, vsf - 1);
    int spx = clamp(((lx % block_w) * 8) / block_w, 0, 7);
    int spy = clamp(((ly % block_h) * 8) / block_h, 0, 7);
    int block = mcu_local * pc.bpm + first_block + by * hsf + bx;
    return nat[block * 64 + spy * 8 + spx];
}

void main()
{
    int lane = int(gl_LocalInvocationIndex);
    int slot = lane >> 3;
    int col = lane & 7;
    int group0 = int(gl_WorkGroupID.x) * pc.mcusPerGroup;
    int used = pc.mcusPerGroup * pc.bpm;

    int mcu_local = 0;
    int b = 0;
    bool live = false;
    if (slot < used)
    {
        mcu_local = slot / pc.bpm;
        b = slot - mcu_local * pc.bpm;
        live = (group0 + mcu_local) < pc.nmcu;
    }

    if (live)
    {
        int blk = (group0 + mcu_local) * pc.bpm + b;
        int dst = slot * 64;
        for (int i = col; i < 64; i += 8)
        {
            int v;
            if (i == 0)
                v = dcd[blk];
            else
            {
                uint w = coef[uint(blk) * 32u + uint(i >> 1)];
                v = (i & 1) != 0 ? (int(w) >> 16) : (int(w << 16) >> 16);
            }
            nat[dst + zigzag[i]] = v;
        }
    }
    barrier();

    if (live)
    {
        int dst = slot * 64;
        int qbase = qt_index_of(b) * 64;
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
            idct_pass(x0, x1, x2, x3, y0, y1, y2, y3,
                n0 * qt[qbase + col], n1 * qt[qbase + col + 8],
                n2 * qt[qbase + col + 16], n3 * qt[qbase + col + 24],
                n4 * qt[qbase + col + 32], n5 * qt[qbase + col + 40],
                n6 * qt[qbase + col + 48], n7 * qt[qbase + col + 56]);
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
    barrier();

    if (live)
    {
        int dst = slot * 64;
        int row = col;
        int x0, x1, x2, x3, y0, y1, y2, y3;
        idct_pass(x0, x1, x2, x3, y0, y1, y2, y3,
            tmp[dst + row], tmp[dst + row + 8], tmp[dst + row + 16], tmp[dst + row + 24],
            tmp[dst + row + 32], tmp[dst + row + 40], tmp[dst + row + 48], tmp[dst + row + 56]);
        const int shift = 17;
        const int bias = 0x10000 + (128 << shift);
        x0 += bias; x1 += bias; x2 += bias; x3 += bias;
        nat[dst + row * 8 + 0] = clamp_u8((x0 + y3) >> shift);
        nat[dst + row * 8 + 1] = clamp_u8((x1 + y2) >> shift);
        nat[dst + row * 8 + 2] = clamp_u8((x2 + y1) >> shift);
        nat[dst + row * 8 + 3] = clamp_u8((x3 + y0) >> shift);
        nat[dst + row * 8 + 4] = clamp_u8((x3 - y0) >> shift);
        nat[dst + row * 8 + 5] = clamp_u8((x2 - y1) >> shift);
        nat[dst + row * 8 + 6] = clamp_u8((x1 - y2) >> shift);
        nat[dst + row * 8 + 7] = clamp_u8((x0 - y3) >> shift);
    }
    barrier();

    int area = pc.xblock * pc.yblock;
    int span = pc.mcusPerGroup * area;
    for (int p = lane; p < span; p += 256)
    {
        int mcu_i = p / area;
        int pix = p - mcu_i * area;
        int global_mcu = group0 + mcu_i;
        if (global_mcu >= pc.nmcu)
            continue;
        int lx = pix % pc.xblock;
        int ly = pix / pc.xblock;
        int px = (global_mcu % pc.xmcu) * pc.xblock + lx;
        int py = (global_mcu / pc.xmcu) * pc.yblock + ly;
        if (px >= pc.width || py >= pc.height)
            continue;

        int Y = sample_at(mcu_i, pc.y_off, pc.y_hsf, pc.y_vsf, lx, ly);
        int R, G, B;
        if (pc.components == 1)
        {
            R = G = B = Y;
        }
        else
        {
            int Cb = sample_at(mcu_i, pc.cb_off, pc.cb_hsf, pc.cb_vsf, lx, ly) - 128;
            int Cr = sample_at(mcu_i, pc.cr_off, pc.cr_hsf, pc.cr_vsf, lx, ly) - 128;
            R = clamp_u8(Y + ((Cr * 91881) >> 16));
            G = clamp_u8(Y - ((Cb * 22554 + Cr * 46802) >> 16));
            B = clamp_u8(Y + ((Cb * 116130) >> 16));
        }
#ifdef LINEAR_OUT
        rgbaOut[py * pc.width + px] = uint(R) | (uint(G) << 8) | (uint(B) << 16) | 0xff000000u;
#else
        imageStore(uOut, ivec2(px, py), vec4(float(R), float(G), float(B), 255.0) / 255.0);
#endif
    }
}
)glsl";

int symEntryHost(int sym, int len, bool dc)
{
    int s = sym & 15;
    int r = sym >> 4;
    int za = dc ? 1 : (s ? r + 1 : (r == 15 ? 16 : 64));
    return (len + s) | (s << 5) | (za << 9);
}

void buildSlot(const mango::image::jpeg::HuffmanCodeTable& src, bool dc,
    std::vector<std::uint16_t>& lut, std::vector<std::uint8_t>& hv, TablesGpu& dst, int slot)
{
    for (int i = 0; i < 18; ++i)
        dst.maxc[slot * 18 + i] = -1;
    dst.maxc[slot * 18 + 17] = 0x7fffffff;
    int code = 0;
    int p = 0;
    for (int len = 1; len <= 16; ++len)
    {
        int nl = src.bits[len - 1];
        if (nl)
        {
            dst.voff[slot * 18 + len] = p - code;
            for (int k = 0; k < nl && p + k < 256; ++k)
            {
                int sym = src.values[p + k];
                hv[slot * 256 + p + k] = std::uint8_t(sym);
                if (len <= kLb)
                {
                    int lo = (code + k) << (kLb - len);
                    int n = 1 << (kLb - len);
                    std::uint16_t ent = std::uint16_t(symEntryHost(sym, len, dc));
                    for (int i = 0; i < n; ++i)
                        lut[slot * kLut + lo + i] = ent;
                }
            }
            p += nl;
            code += nl;
            dst.maxc[slot * 18 + len] = code - 1;
        }
        code <<= 1;
    }
}

void memBarrier(VkCommandBuffer cmd, VkAccessFlags src, VkAccessFlags dst,
    VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = src,
        .dstAccessMask = dst,
    };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void computeBarrier(VkCommandBuffer cmd)
{
    memBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

} // namespace

bool MangoGpu2Decoder::extract(mango::ConstMemory jpeg, std::string& error)
{
    stream = {};
    extract_us = 0;
    uploaded = false;
    m_jpeg = jpeg;
    m_pipeline = false;
    m_static_copied = false;

    const mango::u64 t0 = nowUs();
    mango::image::jpeg::JpegDecoder decoder(jpeg);
    stream = decoder.extractEntropy();
    extract_us = nowUs() - t0;
    if (!stream.ok)
    {
        error = stream.error.empty() ? "extractEntropy failed" : stream.error;
        return false;
    }
    if (stream.packed_bytes == 0 || stream.segments.empty())
    {
        error = "entropy stream is empty";
        return false;
    }
    if (stream.blocks_in_mcu <= 0 || stream.blocks_in_mcu > 10)
    {
        error = "MCU block count is outside 1..10";
        return false;
    }
    return true;
}

bool MangoGpu2Decoder::upload(mango::vulkan::Allocator& allocator, std::string& error)
{
    if (!stream.ok)
    {
        error = "entropy stream is not extracted";
        return false;
    }
    if (uploaded)
        return true;

    const mango::u64 t0 = nowUs();
    const int bpm = stream.blocks_in_mcu;

    std::vector<std::uint8_t> bytes;
    bytes.reserve(stream.packed_bytes + 64);
    std::vector<ChunkGpu> chunk_host;
    std::vector<SegGpu> seg_host;
    std::uint32_t bit = 0;
    int block = 0;
    int mcu = 0;
    for (const auto& seg : stream.segments)
    {
        const std::uint32_t seg_bit = bit;
        const std::uint8_t* p = seg.bytes.address;
        bool ff = false;
        for (size_t i = 0; i < seg.bytes.size; ++i)
        {
            const std::uint8_t b = p[i];
            if (ff && b == 0)
            {
                ff = false;
                continue;
            }
            ff = b == 0xff;
            bytes.push_back(b);
            bit += 8;
        }
        const std::uint32_t seg_end = bit;
        SegGpu sg = {};
        sg.chunk0 = int(chunk_host.size());
        sg.block0 = block;
        sg.mcu0 = mcu;
        sg.mcus = seg.mcus;
        for (std::uint32_t at = seg_bit; at < seg_end; at += kChunkBits)
        {
            ChunkGpu c = {};
            c.bit0 = at;
            c.bit1 = std::min(at + kChunkBits, seg_end);
            c.seg0 = seg_bit;
            c.seg1 = seg_end;
            c.block0 = block;
            c.block1 = block + seg.mcus * bpm;
            chunk_host.push_back(c);
        }
        sg.nchunks = int(chunk_host.size()) - sg.chunk0;
        seg_host.push_back(sg);
        block += seg.mcus * bpm;
        mcu += seg.mcus;
    }
    if (chunk_host.empty())
    {
        error = "no entropy chunks";
        return false;
    }
    chunks = int(chunk_host.size());
    m_nblk = block;
    m_nsegs = int(seg_host.size());

    const size_t nbytes = ((bytes.size() + 3u) & ~size_t(3)) + 32u;
    bytes.resize(nbytes, 0);
    m_nwords = std::uint32_t(nbytes / 4);

    TablesGpu tab = {};
    std::vector<std::uint16_t> lut(8 * kLut, 0);
    std::vector<std::uint8_t> hv(8 * 256, 0);
    for (const auto& table : stream.huffman)
    {
        const int slot = table.cls * 4 + table.id;
        if (slot < 0 || slot >= 8)
        {
            error = "Huffman table id is outside 0..3";
            return false;
        }
        buildSlot(table, table.cls == 0, lut, hv, tab, slot);
    }
    for (int i = 0; i < 8 * kLut / 2; ++i)
        tab.lut[i] = std::uint32_t(lut[2 * i]) | (std::uint32_t(lut[2 * i + 1]) << 16);
    for (int i = 0; i < 8 * 256 / 4; ++i)
    {
        tab.hv[i] = std::uint32_t(hv[4 * i]) | (std::uint32_t(hv[4 * i + 1]) << 8)
            | (std::uint32_t(hv[4 * i + 2]) << 16) | (std::uint32_t(hv[4 * i + 3]) << 24);
    }

    m_dc_pack = m_ac_pack = m_comp_pack = 0;
    for (int b = 0; b < bpm; ++b)
    {
        const auto& info = stream.block[b];
        m_dc_pack |= std::uint32_t(info.dc & 7) << (3 * b);
        m_ac_pack |= std::uint32_t((4 + info.ac) & 7) << (3 * b);
        m_comp_pack |= std::uint32_t(info.component & 3) << (2 * b);
    }

    std::int32_t natural_qt[4 * 64] = {};
    {
        mango::image::jpeg::JpegDecoder info_decoder(m_jpeg);
        const mango::image::jpeg::JpegInfo& info = info_decoder.info();
        for (int t = 0; t < 4; ++t)
        {
            if (!info.quant[t])
                continue;
            for (int i = 0; i < 64; ++i)
                natural_qt[t * 64 + i] = info.quant[t][i];
        }
    }

    using mango::vulkan::MemoryUsage;
    auto staging = [&](mango::vulkan::BufferAllocation& buf, const void* src, size_t size) -> bool
    {
        buf = allocator.createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload, true);
        if (!buf || !buf.mapped)
            return false;
        std::memcpy(buf.mapped, src, size);
        allocator.flush(buf.allocation, 0, size);
        return true;
    };
    auto device = [&](mango::vulkan::BufferAllocation& buf, size_t size, bool dst) -> bool
    {
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (dst)
            usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buf = allocator.createBuffer(std::max<size_t>(size, 16), usage, MemoryUsage::GpuOnly, false);
        return bool(buf);
    };

    // The bitstream goes word-swapped so the shader reads big-endian words directly.
    std::vector<std::uint32_t> words(m_nwords);
    for (std::uint32_t i = 0; i < m_nwords; ++i)
    {
        const std::uint8_t* b = bytes.data() + i * 4;
        words[i] = (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16)
            | (std::uint32_t(b[2]) << 8) | std::uint32_t(b[3]);
    }

    const size_t nch = chunk_host.size();
    const int nmcu = stream.xmcu * stream.ymcu;
    const int nblk_alloc = std::max(m_nblk, nmcu * bpm);
    const int dc_groups = (nmcu + kDcTile - 1) / kDcTile;
    std::vector<std::uint32_t> seg_start(size_t(nmcu + 31) / 32 + 1, 0u);
    for (const SegGpu& sg : seg_host)
    {
        if (sg.mcu0 < nmcu)
            seg_start[size_t(sg.mcu0) >> 5] |= 1u << (sg.mcu0 & 31);
    }
    const bool ok =
        staging(m_bits, words.data(), words.size() * 4)
        && staging(m_chunks_host, chunk_host.data(), nch * sizeof(ChunkGpu))
        && staging(m_segs_host, seg_host.data(), seg_host.size() * sizeof(SegGpu))
        && staging(m_tables_host, &tab, sizeof(tab))
        && staging(m_qt_host, natural_qt, sizeof(natural_qt))
        && device(m_bits_dev, words.size() * 4, true)
        && device(m_chunks, nch * sizeof(ChunkGpu), true)
        && device(m_segs, seg_host.size() * sizeof(SegGpu), true)
        && device(m_tables, sizeof(tab), true)
        && device(m_qt, sizeof(natural_qt), true)
        && device(m_st_in, nch * 8, false)
        && device(m_st_out, nch * 8, false)
        && device(m_cnt, nch * 4, false)
        && device(m_cps, nch * kNcp * 8, false)
        && device(m_cpc, nch * kNcp * 4, false)
        && device(m_first, nch * 4, false)
        && device(m_coef, size_t(nblk_alloc) * 32 * 4, true)
        && device(m_dcd, size_t(nblk_alloc) * 4, true)
        && device(m_flags, kMaxPasses * 4, true)
        && staging(m_seg_start_host, seg_start.data(), seg_start.size() * 4)
        && device(m_seg_start, seg_start.size() * 4, true)
        && device(m_agg, size_t(dc_groups) * 2 * 16, false);
    if (!ok)
    {
        error = "gpu2 buffer allocation failed";
        release(allocator);
        return false;
    }

    m_seg_start_words = std::uint32_t(seg_start.size());
    m_static_copied = false;
    upload_us = nowUs() - t0;
    uploaded = true;
    return true;
}

void MangoGpu2Decoder::release(mango::vulkan::Allocator& allocator)
{
    mango::vulkan::BufferAllocation* all[] = {
        &m_bits, &m_chunks_host, &m_segs_host, &m_tables_host, &m_qt_host,
        &m_bits_dev, &m_chunks, &m_segs, &m_tables, &m_qt,
        &m_st_in, &m_st_out, &m_cnt, &m_cps, &m_cpc, &m_first, &m_coef, &m_dcd, &m_flags,
        &m_seg_start_host, &m_seg_start, &m_agg,
    };
    for (auto* buf : all)
    {
        if (*buf)
            allocator.destroyBuffer(*buf);
        *buf = {};
    }

    if (m_device)
    {
        for (int i = 0; i < PipeCount; ++i)
        {
            if (m_pipes[i])
                vkDestroyPipeline(m_device, m_pipes[i], nullptr);
            if (m_shaders[i])
                vkDestroyShaderModule(m_device, m_shaders[i], nullptr);
            m_pipes[i] = VK_NULL_HANDLE;
            m_shaders[i] = VK_NULL_HANDLE;
        }
        if (m_pipe_layout)
            vkDestroyPipelineLayout(m_device, m_pipe_layout, nullptr);
        if (m_layout)
            vkDestroyDescriptorSetLayout(m_device, m_layout, nullptr);
        if (m_desc_pool)
            vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
        if (m_queries)
            vkDestroyQueryPool(m_device, m_queries, nullptr);
        if (m_pool)
            vkDestroyCommandPool(m_device, m_pool, nullptr);
        if (m_fence)
            vkDestroyFence(m_device, m_fence, nullptr);
    }
    m_pipe_layout = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    m_desc_pool = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    m_queries = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_cmd = VK_NULL_HANDLE;
    m_fence = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_queue = VK_NULL_HANDLE;
    m_pipeline = false;
    m_static_copied = false;
    uploaded = false;
}

bool MangoGpu2Decoder::createPipelines(mango::vulkan::VulkanWindow& window, DecodedTexture& texture,
    std::string& error)
{
    const mango::u64 t_compile = nowUs();
    m_linear = bool(texture.publish_alloc);
    mango::vulkan::Compiler compiler;
    const std::string sources[PipeCount] = {
        std::string(kCommon) + kHuffman + kSyncMain,
        std::string(kCommon) + kPrefixMain,
        std::string(kCommon) + kHuffman + kWriteMain,
        std::string(kCommon) + kDcScan + kDcLocalMain,
        std::string(kCommon) + kDcScan + kDcGroupsMain,
        std::string(kCommon) + (m_linear ? "#define LINEAR_OUT 1\n" : "") + kIdctMain,
    };
    for (int i = 0; i < PipeCount; ++i)
    {
        mango::vulkan::Shader shader = compiler.compile(sources[i].c_str(), mango::vulkan::ShaderStage::Compute);
        if (!shader)
        {
            error = std::string("gpu2 shader compile failed:\n") + shader.log;
            return false;
        }
        m_shaders[i] = mango::vulkan::Compiler::createShaderModule(m_device, shader);
        if (!m_shaders[i])
        {
            error = "gpu2 shader module failed";
            return false;
        }
    }

    VkDescriptorSetLayoutBinding bindings[kBindings] = {};
    for (std::uint32_t i = 0; i < kBindings; ++i)
    {
        const bool image = i == kImageBinding && !m_linear;
        bindings[i] = { i, image ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    VkDescriptorSetLayoutCreateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = kBindings,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(m_device, &set_info, nullptr, &m_layout) != VK_SUCCESS)
    {
        error = "gpu2 descriptor layout failed";
        return false;
    }
    VkPushConstantRange push = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push) };
    VkPipelineLayoutCreateInfo pl = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    };
    if (vkCreatePipelineLayout(m_device, &pl, nullptr, &m_pipe_layout) != VK_SUCCESS)
    {
        error = "gpu2 pipeline layout failed";
        return false;
    }
    for (int i = 0; i < PipeCount; ++i)
    {
        VkComputePipelineCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_shaders[i],
                .pName = "main",
            },
            .layout = m_pipe_layout,
        };
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &info, nullptr, &m_pipes[i]) != VK_SUCCESS)
        {
            error = "gpu2 pipeline failed";
            return false;
        }
    }

    VkDescriptorPoolSize sizes[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_linear ? std::uint32_t(kBindings) : std::uint32_t(kBindings - 1) },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
    };
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = m_linear ? 1u : 2u,
        .pPoolSizes = sizes,
    };
    if (vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_desc_pool) != VK_SUCCESS)
    {
        error = "gpu2 descriptor pool failed";
        return false;
    }
    VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_layout,
    };
    if (vkAllocateDescriptorSets(m_device, &ai, &m_set) != VK_SUCCESS)
    {
        error = "gpu2 descriptor set failed";
        return false;
    }

    const VkBuffer buffers[kBindings] = {
        m_bits_dev.buffer, m_chunks.buffer, m_segs.buffer, m_tables.buffer,
        m_st_in.buffer, m_st_out.buffer, m_cnt.buffer, m_cps.buffer, m_cpc.buffer,
        m_first.buffer, m_coef.buffer, m_dcd.buffer, m_qt.buffer, m_flags.buffer,
        m_linear ? texture.publish_alloc.buffer : VK_NULL_HANDLE, m_seg_start.buffer, m_agg.buffer,
    };
    VkDescriptorBufferInfo buf_infos[kBindings] = {};
    VkWriteDescriptorSet writes[kBindings] = {};
    for (std::uint32_t i = 0; i < kBindings; ++i)
    {
        if (i == kImageBinding && !m_linear)
            continue;
        buf_infos[i] = { buffers[i], 0, VK_WHOLE_SIZE };
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buf_infos[i],
        };
    }
    VkDescriptorImageInfo img_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = texture.view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    if (!m_linear)
    {
        writes[kImageBinding] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_set,
            .dstBinding = kImageBinding,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &img_info,
        };
    }
    vkUpdateDescriptorSets(m_device, kBindings, writes, 0, nullptr);

    VkCommandPoolCreateInfo pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = window.graphicsQueueFamilyIndex(),
    };
    if (vkCreateCommandPool(m_device, &pool_ci, nullptr, &m_pool) != VK_SUCCESS)
    {
        error = "gpu2 command pool failed";
        return false;
    }
    VkCommandBufferAllocateInfo cmd_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(m_device, &cmd_ai, &m_cmd) != VK_SUCCESS)
    {
        error = "gpu2 command buffer failed";
        return false;
    }
    VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(m_device, &fence_ci, nullptr, &m_fence) != VK_SUCCESS)
    {
        error = "gpu2 fence failed";
        return false;
    }

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(window.physicalDevice(), &props);
    m_tick_ns = props.limits.timestampPeriod;
    VkQueryPoolCreateInfo qp = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 8,
    };
    if (m_tick_ns <= 0.0f || vkCreateQueryPool(m_device, &qp, nullptr, &m_queries) != VK_SUCCESS)
        m_queries = VK_NULL_HANDLE;

    compile_us = nowUs() - t_compile;
    compile_last_us = compile_us;
    m_pipeline = true;
    return true;
}

bool MangoGpu2Decoder::decode(mango::vulkan::VulkanWindow& window,
    mango::vulkan::Allocator& allocator, DecodedTexture& texture, std::string& error)
{
    (void)allocator;
    if (!uploaded)
    {
        error = "entropy buffer is not uploaded";
        return false;
    }
    if (!texture.image || !texture.view)
    {
        error = "DecodedTexture missing for mango-gpu2";
        return false;
    }

    m_device = window.device();
    m_queue = window.graphicsQueue();
    compile_last_us = 0;
    decode_us = huffman_us = pixel_us = wall_us = 0;

    if (!m_pipeline && !createPipelines(window, texture, error))
        return false;

    mango::image::jpeg::JpegDecoder info_decoder(m_jpeg);
    const mango::image::jpeg::JpegInfo& info = info_decoder.info();
    Push push = {};
    push.nch = chunks;
    push.rounds = kSyncRounds;
    push.bpm = stream.blocks_in_mcu;
    push.dc_pack = m_dc_pack;
    push.ac_pack = m_ac_pack;
    push.comp_pack = m_comp_pack;
    push.nwords = m_nwords;
    push.width = info.width;
    push.height = info.height;
    push.xmcu = info.xmcu;
    push.xblock = info.xblock;
    push.yblock = info.yblock;
    push.components = info.components;
    int off = 0;
    auto setComp = [&](int idx, std::int32_t& hsf, std::int32_t& vsf, std::int32_t& qt, std::int32_t& o)
    {
        if (idx >= info.components)
        {
            hsf = vsf = 1;
            qt = 0;
            o = 0;
            return;
        }
        hsf = info.component[idx].hsf;
        vsf = info.component[idx].vsf;
        qt = info.component[idx].qt;
        o = off;
        off += hsf * vsf;
    };
    setComp(0, push.y_hsf, push.y_vsf, push.y_qt, push.y_off);
    setComp(1, push.cb_hsf, push.cb_vsf, push.cb_qt, push.cb_off);
    setComp(2, push.cr_hsf, push.cr_vsf, push.cr_qt, push.cr_off);
    push.mcus_per_group = std::max(32 / std::max(push.bpm, 1), 1);
    push.nmcu = info.xmcu * info.ymcu;

    VkCommandBuffer cmd = m_cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin);
    if (m_queries)
    {
        vkCmdResetQueryPool(cmd, m_queries, 0, 8);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queries, 0);
    }

    memBarrier(cmd, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    auto copy = [&](const mango::vulkan::BufferAllocation& src, const mango::vulkan::BufferAllocation& dst, VkDeviceSize size)
    {
        VkBufferCopy region = { 0, 0, size };
        vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region);
    };
    copy(m_bits, m_bits_dev, VkDeviceSize(m_nwords) * 4);
    if (!m_static_copied)
    {
        copy(m_chunks_host, m_chunks, VkDeviceSize(chunks) * sizeof(ChunkGpu));
        copy(m_segs_host, m_segs, VkDeviceSize(m_nsegs) * sizeof(SegGpu));
        copy(m_tables_host, m_tables, sizeof(TablesGpu));
        copy(m_qt_host, m_qt, sizeof(std::int32_t) * 256);
        copy(m_seg_start_host, m_seg_start, VkDeviceSize(m_seg_start_words) * 4);
        m_static_copied = true;
    }
    vkCmdFillBuffer(cmd, m_coef.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, m_dcd.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, m_flags.buffer, 0, VK_WHOLE_SIZE, 0);
    memBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    auto stamp = [&](VkPipelineStageFlagBits stage, std::uint32_t index)
    {
        if (m_queries)
            vkCmdWriteTimestamp(cmd, stage, m_queries, index);
    };
    stamp(VK_PIPELINE_STAGE_TRANSFER_BIT, 1);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_layout, 0, 1, &m_set, 0, nullptr);
    auto dispatch = [&](Pipe pipe, std::uint32_t groups)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipes[pipe]);
        vkCmdPushConstants(cmd, m_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, std::max(groups, 1u), 1, 1);
    };

    const std::uint32_t sync_groups = (std::uint32_t(chunks) + 127u) / 128u;
    for (int pass = 0; pass < kMaxPasses; ++pass)
    {
        push.pass = pass;
        dispatch(PipeSync, sync_groups);
        computeBarrier(cmd);
    }
    stamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 2);
    dispatch(PipePrefix, std::uint32_t(m_nsegs));
    computeBarrier(cmd);
    stamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 3);
    dispatch(PipeWrite, (std::uint32_t(chunks) * kPieces + 127u) / 128u);
    computeBarrier(cmd);
    stamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 4);
    {
        const std::uint32_t dc_groups = (std::uint32_t(push.nmcu) + kDcTile - 1u) / kDcTile;
        push.pass = 0;
        dispatch(PipeDcLocal, dc_groups);
        computeBarrier(cmd);
        const std::int32_t nch = push.nch;
        push.nch = std::int32_t(dc_groups);
        dispatch(PipeDcGroups, 1);
        computeBarrier(cmd);
        push.nch = nch;
        push.pass = 1;
        dispatch(PipeDcLocal, dc_groups);
    }
    stamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 5);

    if (m_linear)
        computeBarrier(cmd);
    else
    {
    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkAccessFlags src_access = VK_ACCESS_SHADER_WRITE_BIT;
    if (texture.layout != VK_IMAGE_LAYOUT_UNDEFINED && texture.layout != VK_IMAGE_LAYOUT_GENERAL)
    {
        src_stage |= VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        src_access |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    }
    VkImageMemoryBarrier to_general = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = texture.layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0u : src_access,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = texture.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_UNDEFINED : texture.layout,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    VkMemoryBarrier coeffs_ready = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, src_stage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &coeffs_ready, 0, nullptr, 1, &to_general);
    texture.layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    const std::uint32_t idct_groups = (std::uint32_t(push.nmcu) + std::uint32_t(push.mcus_per_group) - 1u)
        / std::uint32_t(push.mcus_per_group);
    dispatch(PipeIdct, idct_groups);
    stamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 6);
    if (m_linear)
        texture.recordPublishFromBuffer(cmd);
    else
        texture.recordPublishComputeStores(cmd);
    stamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 7);
    vkEndCommandBuffer(cmd);

    const mango::u64 t0 = nowUs();
    vkResetFences(m_device, 1, &m_fence);
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(m_queue, 1, &submit, m_fence) != VK_SUCCESS)
    {
        error = "vkQueueSubmit failed";
        return false;
    }
    const VkResult wr = vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, 20ull * 1000ull * 1000ull * 1000ull);
    if (wr != VK_SUCCESS)
    {
        error = wr == VK_TIMEOUT ? "gpu2 fence timed out" : "vkWaitForFences failed";
        return false;
    }
    wall_us = nowUs() - t0;
    decode_us = wall_us;

    if (m_queries)
    {
        std::uint64_t ts[8] = {};
        if (vkGetQueryPoolResults(m_device, m_queries, 0, 8, sizeof(ts), ts, sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
        {
            auto us = [&](int a, int b) { return mango::u64(double(ts[b] - ts[a]) * m_tick_ns / 1000.0); };
            for (int i = 0; i < 6; ++i)
                stage_us[i] = us(i, i + 1);
            huffman_us = us(0, 5);
            pixel_us = us(5, 6);
            decode_us = huffman_us + pixel_us;
            publish_us = us(6, 7);
        }
    }
    return true;
}

} // namespace gpujpeg

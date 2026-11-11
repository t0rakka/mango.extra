// Copyright 2026 Distance Technologies Oy. For internal use only.
//
#include "mango_gpu_backend.hpp"
#include "decoded_texture.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mango/vulkan/compiler.hpp>

namespace gpujpeg
{

namespace
{

struct TableGpu
{
    std::int32_t mincode[16];
    std::int32_t maxcode[16];
    std::int32_t valptr[16];
    std::uint32_t lut[256];
    std::uint32_t vals[256];
};

// Flat so the shader loads one word per lookup. Array-of-struct was copying
// the whole table into locals on every symbol.
struct TablesFlat
{
    std::int32_t mincode[128];
    std::int32_t maxcode[128];
    std::int32_t valptr[128];
    std::uint32_t lut[2048];
    std::uint32_t vals[2048];
};

struct SegGpu
{
    std::uint32_t offset;
    std::uint32_t size;
    std::uint32_t y0;
    std::uint32_t y1;
};

struct PushData
{
    std::int32_t xmcu;
    std::int32_t blocks;
    std::int32_t segCount;
    std::int32_t chunkBytes;
    std::int32_t waveBase;
    std::int32_t coeffLimit;
    std::uint32_t pack[10];
};

// Per-segment cursor carried from one 64-chunk wave to the next.
// std430 stride is 48 bytes; the shader struct must match.
struct CarryGpu
{
    std::uint32_t pos;
    std::uint32_t nbits;
    std::uint32_t buf;
    std::uint32_t z;
    std::uint32_t bi;
    std::int32_t p0;
    std::int32_t p1;
    std::int32_t p2;
    std::int32_t p3;
    std::uint32_t coeffBase;
    std::uint32_t rounds;
    std::uint32_t pad;
};
static_assert(sizeof(CarryGpu) == 48, "CarryGpu must match the Huffman shader");

// Must match layout(local_size_x) in the Huffman shader.
constexpr std::uint32_t kSyncLanes = 64;
// Preferred chunk. Grown per file so the largest segment still fits in one
// workgroup and every lane walks its own piece in a single dispatch.
constexpr std::uint32_t kTargetChunk = 256;

std::uint32_t chooseChunkBytes(const mango::image::jpeg::EntropyStream& stream)
{
    std::uint32_t max_size = 1;
    for (const auto& seg : stream.segments)
        max_size = std::max(max_size, std::uint32_t(seg.bytes.size));
    std::uint32_t chunk = kTargetChunk;
    const std::uint32_t need = (max_size + kSyncLanes - 1u) / kSyncLanes;
    if (need > chunk)
        chunk = need;
    return chunk;
}

constexpr char kHuffmanSource[] = R"glsl(
#version 450
layout(local_size_x = 64) in;

struct Seg {
    uint offset;
    uint size;
    uint y0;
    uint y1;
};

layout(std430, set = 0, binding = 0) readonly buffer Bits { uint bits[]; };
layout(std430, set = 0, binding = 1) readonly buffer Segs { Seg segs[]; };
layout(std430, set = 0, binding = 2) readonly buffer Tables {
    int mincode[128];
    int maxcode[128];
    int valptr[128];
    uint lut[2048];
    uint vals[2048];
} tb;
layout(std430, set = 0, binding = 3) writeonly buffer Coeffs { int coeffs[]; };

struct Carry {
    uint pos;
    uint nbits;
    uint buf;
    uint z;
    uint bi;
    int p0;
    int p1;
    int p2;
    int p3;
    uint coeffBase;
    uint rounds;
    uint pad;
};
layout(std430, set = 0, binding = 4) buffer Stats { Carry carry[]; };

layout(push_constant) uniform Push {
    int xmcu;
    int blocks;
    int segCount;
    int chunkBytes;
    int waveBase;
    int coeffLimit;
    uint pack[10];
} pc;

struct St {
    uint pos;
    uint nbits;
    uint buf;
    uint z;
    uint bi;
    uint n;
    int d0;
    int d1;
    int d2;
    int d3;
};

struct Reader {
    uint pos;
    uint endp;
    uint buf;
    uint nbits;
};

uint loadByte(uint i)
{
    uint w = bits[i >> 2];
    return (w >> ((i & 3u) * 8u)) & 0xffu;
}

uint pullByte(inout Reader r)
{
    if (r.pos >= r.endp)
        return 0u;
    uint b = loadByte(r.pos);
    r.pos++;
    if (b == 0xffu)
    {
        if (r.pos < r.endp && loadByte(r.pos) == 0u)
            r.pos++;
        else
            r.pos = r.endp;
    }
    return b;
}

void fill(inout Reader r, uint need)
{
    while (r.nbits < need && r.pos < r.endp)
    {
        r.buf = (r.buf << 8) | pullByte(r);
        r.nbits += 8u;
    }
}

uint getBits(inout Reader r, uint n)
{
    // Real magnitudes are 1..11. A speculative lane can ask for 32 or more;
    // that shift is undefined, and on AMD it collapses to a mask of zero.
    // Return without shifting. cursorEq rejects any cursor that still
    // reports nbits >= 32, so this lane is resynchronized from its neighbor.
    if (n >= 32u)
        return 0u;
    fill(r, n);
    if (r.nbits < n)
        return 0u;
    r.nbits -= n;
    return (r.buf >> r.nbits) & ((1u << n) - 1u);
}

int extendSign(int v, int n)
{
    if (n <= 0 || n >= 31)
        return v;
    int vt = 1 << (n - 1);
    if (v < vt)
        v -= (1 << n) - 1;
    return v;
}

int decodeSym(inout Reader r, int ti)
{
    if (ti < 0 || ti > 7)
        return 0;
    fill(r, 8u);
    if (r.nbits >= 8u)
    {
        uint look = (r.buf >> (r.nbits - 8u)) & 0xffu;
        uint packed = tb.lut[ti * 256 + int(look)];
        uint len = packed & 0xffu;
        if (len > 0u && len <= 8u)
        {
            r.nbits -= len;
            return int(packed >> 8);
        }
    }

    int base = ti * 16;
    int code = 0;
    for (int L = 0; L < 16; ++L)
    {
        if (r.nbits == 0u)
            fill(r, 8u);
        if (r.nbits == 0u)
            return 0;
        r.nbits -= 1u;
        code = (code << 1) | int((r.buf >> r.nbits) & 1u);
        int mx = tb.maxcode[base + L];
        if (mx >= 0 && code <= mx)
        {
            int idx = tb.valptr[base + L] + code - tb.mincode[base + L];
            if (idx < 0 || idx > 255)
                return 0;
            return int(tb.vals[ti * 256 + idx]);
        }
    }
    return 0;
}

St walkChunk(St start, uint limitByte, uint segEndByte, uint dst, bool doWrite,
    int p0, int p1, int p2, int p3, uint endIdx)
{
    Reader r;
    r.pos = start.pos;
    r.endp = segEndByte;
    r.buf = start.buf;
    r.nbits = start.nbits;

    St s;
    s.pos = start.pos;
    s.nbits = start.nbits;
    s.buf = start.buf;
    s.z = start.z;
    s.bi = start.bi;
    s.n = 0u;
    s.d0 = 0;
    s.d1 = 0;
    s.d2 = 0;
    s.d3 = 0;

    int pred0 = p0;
    int pred1 = p1;
    int pred2 = p2;
    int pred3 = p3;
    uint limit = limitByte * 8u;
    uint cur0 = r.pos * 8u - r.nbits;
    int budget = 64;
    if (limit > cur0)
        budget = int(limit - cur0) + 64;

    for (int guard = 0; guard < budget; ++guard)
    {
        if (r.nbits > r.pos * 8u)
            break;
        uint cur = r.pos * 8u - r.nbits;
        if (cur >= limit)
            break;
        if (r.nbits == 0u && r.pos >= r.endp)
            break;

        uint bi = s.bi;
        if (bi >= uint(pc.blocks))
            bi = 0u;
        uint pack = pc.pack[bi];
        bool isDc = (s.z == 0u);
        int ti = isDc ? int(pack & 0xffu) : int((pack >> 8) & 0xffu);
        int pi = int((pack >> 16) & 0xffu);
        int symbol = decodeSym(r, ti);

        if (isDc)
        {
            int diff = 0;
            if (symbol != 0)
                diff = extendSign(int(getBits(r, uint(symbol))), symbol);
            int shown = diff;
            if (doWrite)
            {
                if (pi == 0) { shown = diff + pred0; pred0 = shown; }
                else if (pi == 1) { shown = diff + pred1; pred1 = shown; }
                else if (pi == 2) { shown = diff + pred2; pred2 = shown; }
                else { shown = diff + pred3; pred3 = shown; }
                if (dst + s.n < endIdx)
                    coeffs[dst + s.n] = shown;
            }
            if (pi == 0) s.d0 += diff;
            else if (pi == 1) s.d1 += diff;
            else if (pi == 2) s.d2 += diff;
            else s.d3 += diff;
            s.n += 1u;
            s.z = 1u;
        }
        else if (symbol == 0)
        {
            uint left = 64u - s.z;
            if (doWrite)
            {
                for (uint k = 0u; k < left; ++k)
                {
                    if (dst + s.n + k < endIdx)
                        coeffs[dst + s.n + k] = 0;
                }
            }
            s.n += left;
            s.z = 0u;
            s.bi += 1u;
            if (s.bi >= uint(pc.blocks))
                s.bi = 0u;
        }
        else
        {
            int run = symbol >> 4;
            int nb = symbol & 15;
            if (nb == 0)
            {
                uint zeros = 16u;
                bool endBlock = (s.z + zeros >= 64u);
                if (endBlock)
                    zeros = 64u - s.z;
                if (doWrite)
                {
                    for (uint k = 0u; k < zeros; ++k)
                    {
                        if (dst + s.n + k < endIdx)
                            coeffs[dst + s.n + k] = 0;
                    }
                }
                s.n += zeros;
                if (endBlock)
                {
                    s.z = 0u;
                    s.bi += 1u;
                    if (s.bi >= uint(pc.blocks))
                        s.bi = 0u;
                }
                else
                    s.z += zeros;
            }
            else
            {
                int mag = extendSign(int(getBits(r, uint(nb))), nb);
                uint room = 64u - s.z;
                uint zeros = uint(run);
                bool hasValue = true;
                if (zeros >= room)
                {
                    zeros = room;
                    hasValue = false;
                }
                if (doWrite)
                {
                    for (uint k = 0u; k < zeros; ++k)
                    {
                        if (dst + s.n + k < endIdx)
                            coeffs[dst + s.n + k] = 0;
                    }
                }
                s.n += zeros;
                s.z += zeros;
                if (hasValue && s.z < 64u)
                {
                    if (doWrite && dst + s.n < endIdx)
                        coeffs[dst + s.n] = mag;
                    s.n += 1u;
                    s.z += 1u;
                }
                if (s.z >= 64u)
                {
                    s.z = 0u;
                    s.bi += 1u;
                    if (s.bi >= uint(pc.blocks))
                        s.bi = 0u;
                }
            }
        }
    }

    s.pos = r.pos;
    s.nbits = r.nbits;
    s.buf = r.buf;
    return s;
}

// Cursor identity. A shift of 32 or more is undefined in GLSL; AMD masks the
// count to 5 bits, so (1u << 32) becomes 1 and the mask is 0. Every speculative
// cursor then compares equal and lanes after the first commit a bad block.
bool cursorEq(St a, St b)
{
    if (a.pos != b.pos || a.nbits != b.nbits || a.z != b.z || a.bi != b.bi)
        return false;
    if (a.nbits == 0u)
        return true;
    if (a.nbits >= 32u)
        return false;
    uint mask = (1u << a.nbits) - 1u;
    return (a.buf & mask) == (b.buf & mask);
}

shared St spec[64];
shared St got[64];
shared uint matched[64];
shared uint frontier;
shared int baseN[64];
shared int p0s[64];
shared int p1s[64];
shared int p2s[64];
shared int p3s[64];

void main()
{
    uint lid = gl_LocalInvocationID.x;
    uint gid = gl_WorkGroupID.x;
    Seg seg = segs[gid];
    uint segBegin = seg.offset;
    uint segEnd = seg.offset + seg.size;
    uint chunk = max(uint(pc.chunkBytes), 1u);
    uint nchunks = (seg.size + chunk - 1u) / chunk;
    if (nchunks > 64u)
        nchunks = 64u;

    // Every lane speculates from its own byte boundary. Lane 0 is the real
    // start. Later lanes adopt the corrected cursor one chunk at a time, and
    // skip ahead when a speculation already landed on that cursor.
    if (lid < nchunks)
    {
        St start;
        start.pos = segBegin + lid * chunk;
        start.nbits = 0u;
        start.buf = 0u;
        start.z = 0u;
        start.bi = 0u;
        start.n = 0u;
        start.d0 = 0;
        start.d1 = 0;
        start.d2 = 0;
        start.d3 = 0;
        uint end = start.pos + chunk;
        if (end > segEnd)
            end = segEnd;
        spec[lid] = walkChunk(start, end, segEnd, 0u, false, 0, 0, 0, 0, 0u);
    }
    if (lid == 0u)
        frontier = 1u;
    barrier();

    uint rounds = 0u;
    for (uint round = 0u; round < 64u; ++round)
    {
        if (frontier >= nchunks)
            break;
        rounds = round + 1u;

        if (lid >= frontier && lid < nchunks)
        {
            uint end = segBegin + (lid + 1u) * chunk;
            if (end > segEnd)
                end = segEnd;
            St st = walkChunk(spec[lid - 1u], end, segEnd, 0u, false, 0, 0, 0, 0, 0u);
            got[lid] = st;
            matched[lid] = cursorEq(st, spec[lid]) ? 1u : 0u;
        }
        barrier();

        if (lid == 0u)
        {
            uint f = frontier;
            while (f < nchunks)
            {
                spec[f] = got[f];
                uint ok = matched[f];
                f += 1u;
                if (ok == 0u)
                    break;
            }
            frontier = f;
        }
        barrier();
    }

    if (lid == 0u)
    {
        int acc = 0;
        int a0 = 0;
        int a1 = 0;
        int a2 = 0;
        int a3 = 0;
        for (uint i = 0u; i < nchunks; ++i)
        {
            baseN[i] = acc;
            p0s[i] = a0;
            p1s[i] = a1;
            p2s[i] = a2;
            p3s[i] = a3;
            acc += int(spec[i].n);
            a0 += spec[i].d0;
            a1 += spec[i].d1;
            a2 += spec[i].d2;
            a3 += spec[i].d3;
        }
        Carry outc;
        outc.pos = 0u;
        outc.nbits = 0u;
        outc.buf = 0u;
        outc.z = 0u;
        outc.bi = 0u;
        outc.p0 = 0;
        outc.p1 = 0;
        outc.p2 = 0;
        outc.p3 = 0;
        outc.coeffBase = 0u;
        outc.rounds = rounds;
        outc.pad = 0u;
        carry[gid] = outc;
    }
    barrier();

    if (lid < nchunks)
    {
        uint begin = segBegin + lid * chunk;
        uint end = begin + chunk;
        if (end > segEnd)
            end = segEnd;
        St start;
        if (lid == 0u)
        {
            start.pos = segBegin;
            start.nbits = 0u;
            start.buf = 0u;
            start.z = 0u;
            start.bi = 0u;
            start.n = 0u;
            start.d0 = 0;
            start.d1 = 0;
            start.d2 = 0;
            start.d3 = 0;
        }
        else
            start = spec[lid - 1u];
        uint stride = uint(pc.xmcu) * uint(pc.blocks) * 64u;
        uint dst = seg.y0 * stride + uint(baseN[lid]);
        // Pad bits sit between the last block and the restart marker. Decoding
        // them emits coefficients past this segment and overwrites the next row.
        uint endIdx = seg.y1 * stride;
        walkChunk(start, end, segEnd, dst, true, p0s[lid], p1s[lid], p2s[lid], p3s[lid], endIdx);
    }
}
)glsl";

struct IdctPush
{
    std::int32_t width;
    std::int32_t height;
    std::int32_t xmcu;
    std::int32_t mcu_y0;
    std::int32_t mcu_rows;
    std::int32_t xblock;
    std::int32_t yblock;
    std::int32_t blocks_in_mcu;
    std::int32_t components;
    std::int32_t y_hsf, y_vsf, y_qt, y_off;
    std::int32_t cb_hsf, cb_vsf, cb_qt, cb_off;
    std::int32_t cr_hsf, cr_vsf, cr_qt, cr_off;
    std::int32_t mcus_per_group;
};

// Spectral int32 coefficients → RGBA8. Quant table is natural 8x8 order.
// 256 threads, 8 per block. Host dispatch uses the same 32 block slots.
constexpr char kIdctSource[] = R"glsl(
#version 450
layout(local_size_x = 256) in;

layout(binding = 0, rgba8) uniform writeonly image2D uOut;
layout(binding = 1) readonly buffer Coeffs { int c[]; };
layout(binding = 2) readonly buffer Qt { int qt[4 * 64]; };

layout(push_constant) uniform PC {
    int width;
    int height;
    int xmcu;
    int mcu_y0;
    int mcu_rows;
    int xblock;
    int yblock;
    int blocks_in_mcu;
    int components;
    int y_hsf, y_vsf, y_qt, y_off;
    int cb_hsf, cb_vsf, cb_qt, cb_off;
    int cr_hsf, cr_vsf, cr_qt, cr_off;
    int mcus_per_group;
} pc;

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
    int spx = ((lx % block_w) * 8) / block_w;
    int spy = ((ly % block_h) * 8) / block_h;
    spx = clamp(spx, 0, 7);
    spy = clamp(spy, 0, 7);
    int block = mcu_local * pc.blocks_in_mcu + first_block + by * hsf + bx;
    return nat[block * 64 + spy * 8 + spx];
}

void main()
{
    int lane = int(gl_LocalInvocationIndex);
    int block_slot = lane >> 3;
    int col = lane & 7;
    int group_mcu0 = int(gl_WorkGroupID.x) * pc.mcus_per_group;
    int mcu_limit = (pc.mcu_y0 + pc.mcu_rows) * pc.xmcu;
    int used = pc.mcus_per_group * pc.blocks_in_mcu;

    int mcu_local = 0;
    int b = 0;
    bool live = false;
    if (block_slot < used)
    {
        mcu_local = block_slot / pc.blocks_in_mcu;
        b = block_slot - mcu_local * pc.blocks_in_mcu;
        live = (group_mcu0 + mcu_local) < mcu_limit;
    }

    if (live)
    {
        int global_mcu = pc.mcu_y0 * pc.xmcu + group_mcu0 + mcu_local;
        int base = (global_mcu * pc.blocks_in_mcu + b) * 64;
        int dst = block_slot * 64;
        for (int i = col; i < 64; i += 8)
            nat[dst + zigzag[i]] = c[base + i];
    }
    barrier();

    if (live)
    {
        int dst = block_slot * 64;
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
            int s0 = n0 * qt[qbase + col];
            int s1 = n1 * qt[qbase + col + 8];
            int s2 = n2 * qt[qbase + col + 16];
            int s3 = n3 * qt[qbase + col + 24];
            int s4 = n4 * qt[qbase + col + 32];
            int s5 = n5 * qt[qbase + col + 40];
            int s6 = n6 * qt[qbase + col + 48];
            int s7 = n7 * qt[qbase + col + 56];
            int x0, x1, x2, x3, y0, y1, y2, y3;
            idct_pass(x0, x1, x2, x3, y0, y1, y2, y3, s0, s1, s2, s3, s4, s5, s6, s7);
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
            tmp[dst + col * 8 + 0] = dc;
            tmp[dst + col * 8 + 1] = dc;
            tmp[dst + col * 8 + 2] = dc;
            tmp[dst + col * 8 + 3] = dc;
            tmp[dst + col * 8 + 4] = dc;
            tmp[dst + col * 8 + 5] = dc;
            tmp[dst + col * 8 + 6] = dc;
            tmp[dst + col * 8 + 7] = dc;
        }
    }
    barrier();

    if (live)
    {
        int dst = block_slot * 64;
        int row = col;
        int s0 = tmp[dst + row];
        int s1 = tmp[dst + row + 8];
        int s2 = tmp[dst + row + 16];
        int s3 = tmp[dst + row + 24];
        int s4 = tmp[dst + row + 32];
        int s5 = tmp[dst + row + 40];
        int s6 = tmp[dst + row + 48];
        int s7 = tmp[dst + row + 56];
        int x0, x1, x2, x3, y0, y1, y2, y3;
        idct_pass(x0, x1, x2, x3, y0, y1, y2, y3, s0, s1, s2, s3, s4, s5, s6, s7);
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
    int span = pc.mcus_per_group * area;
    for (int p = lane; p < span; p += 256)
    {
        int mcu_i = p / area;
        int pix = p - mcu_i * area;
        int global_mcu = pc.mcu_y0 * pc.xmcu + group_mcu0 + mcu_i;
        if (global_mcu >= mcu_limit)
            continue;
        int lx = pix % pc.xblock;
        int ly = pix / pc.xblock;
        int mcu_x = global_mcu % pc.xmcu;
        int mcu_y = global_mcu / pc.xmcu;
        int px = mcu_x * pc.xblock + lx;
        int py = mcu_y * pc.yblock + ly;
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
        imageStore(uOut, ivec2(px, py), vec4(float(R), float(G), float(B), 255.0) / 255.0);
    }
}
)glsl";

void buildTable(const mango::image::jpeg::HuffmanCodeTable& src, TableGpu& dst)
{
    std::memset(&dst, 0, sizeof(dst));
    for (int i = 0; i < 16; ++i)
        dst.maxcode[i] = -1;

    int huffsize[257] = {};
    std::uint32_t huffcode[257] = {};
    int p = 0;
    for (int len = 1; len <= 16; ++len)
    {
        int n = src.bits[len - 1];
        while (n-- > 0 && p < 256)
            huffsize[p++] = len;
    }
    huffsize[p] = 0;

    std::uint32_t code = 0;
    int si = huffsize[0];
    int q = 0;
    while (huffsize[q])
    {
        while (huffsize[q] == si)
            huffcode[q++] = code++;
        code <<= 1;
        ++si;
    }

    p = 0;
    for (int len = 1; len <= 16; ++len)
    {
        int n = src.bits[len - 1];
        if (!n)
            continue;
        dst.valptr[len - 1] = p;
        dst.mincode[len - 1] = std::int32_t(huffcode[p]);
        p += n;
        dst.maxcode[len - 1] = std::int32_t(huffcode[p - 1]);
    }

    const int nvalues = std::min(src.nvalues, 256);
    for (int i = 0; i < nvalues; ++i)
        dst.vals[i] = src.values[i];

    p = 0;
    for (int len = 1; len <= 16; ++len)
    {
        int n = src.bits[len - 1];
        for (int i = 0; i < n && p < 256; ++i, ++p)
        {
            if (len > 8)
                continue;
            const std::uint32_t c = huffcode[p];
            const std::uint32_t sym = src.values[p];
            const int shift = 8 - len;
            const std::uint32_t base = c << shift;
            const std::uint32_t count = 1u << shift;
            for (std::uint32_t k = 0; k < count; ++k)
                dst.lut[base + k] = std::uint32_t(len) | (sym << 8);
        }
    }
}

bool submitAndWait(VkQueue queue, VkCommandBuffer cmd, VkFence fence, VkDevice device,
    std::uint64_t timeout_ns, std::string& error)
{
    vkResetFences(device, 1, &fence);
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS)
    {
        error = "vkQueueSubmit failed";
        return false;
    }
    const VkResult wr = vkWaitForFences(device, 1, &fence, VK_TRUE, timeout_ns);
    if (wr != VK_SUCCESS)
    {
        error = wr == VK_TIMEOUT ? "huffman fence timed out" : "vkWaitForFences failed";
        return false;
    }
    return true;
}

} // namespace

bool MangoGpuDecoder::extract(mango::ConstMemory jpeg, std::string& error)
{
    stream = {};
    extract_us = 0;
    uploaded = false;
    m_jpeg = jpeg;

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

bool MangoGpuDecoder::upload(mango::vulkan::Allocator& allocator, std::string& error)
{
    if (!stream.ok)
    {
        error = "entropy stream is not extracted";
        return false;
    }
    if (uploaded)
        return true;

    if (bits)
        allocator.destroyBuffer(bits);
    bits = {};

    const mango::u64 t0 = nowUs();
    const VkDeviceSize payload = VkDeviceSize(stream.packed_bytes);
    const VkDeviceSize bit_bytes = (payload + 16u + 3u) & ~VkDeviceSize(3);
    bits = allocator.createBuffer(
        bit_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        mango::vulkan::MemoryUsage::Upload,
        true);
    if (!bits || !bits.mapped)
    {
        error = "entropy upload buffer failed";
        release(allocator);
        return false;
    }

    auto* dst = static_cast<std::uint8_t*>(bits.mapped);
    for (const mango::image::jpeg::EntropySegment& seg : stream.segments)
    {
        if (!seg.bytes.address || seg.packed_offset + seg.bytes.size > stream.packed_bytes)
        {
            error = "entropy segment is outside the packed buffer";
            release(allocator);
            return false;
        }
        std::memcpy(dst + seg.packed_offset, seg.bytes.address, seg.bytes.size);
    }
    if (bit_bytes > payload)
        std::memset(dst + payload, 0, std::size_t(bit_bytes - payload));
    allocator.flush(bits.allocation, 0, bit_bytes);
    upload_us = nowUs() - t0;
    uploaded = true;
    return true;
}

void MangoGpuDecoder::release(mango::vulkan::Allocator& allocator)
{
    auto drop = [&](mango::vulkan::BufferAllocation& buf)
    {
        if (buf)
            allocator.destroyBuffer(buf);
        buf = {};
    };
    drop(bits);
    drop(m_segs);
    drop(m_tables);
    drop(m_coeffs);
    drop(m_qt);
    drop(m_stats);
    drop(m_stats_rb);

    if (m_device)
    {
        if (m_pipe)
            vkDestroyPipeline(m_device, m_pipe, nullptr);
        if (m_pipe_layout)
            vkDestroyPipelineLayout(m_device, m_pipe_layout, nullptr);
        if (m_set_layout)
            vkDestroyDescriptorSetLayout(m_device, m_set_layout, nullptr);
        if (m_desc_pool)
            vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
        if (m_idct_pipe)
            vkDestroyPipeline(m_device, m_idct_pipe, nullptr);
        if (m_idct_pipe_layout)
            vkDestroyPipelineLayout(m_device, m_idct_pipe_layout, nullptr);
        if (m_idct_layout)
            vkDestroyDescriptorSetLayout(m_device, m_idct_layout, nullptr);
        if (m_idct_pool)
            vkDestroyDescriptorPool(m_device, m_idct_pool, nullptr);
        if (m_idct_shader)
            vkDestroyShaderModule(m_device, m_idct_shader, nullptr);
        if (m_shader)
            vkDestroyShaderModule(m_device, m_shader, nullptr);
        if (m_pool)
            vkDestroyCommandPool(m_device, m_pool, nullptr);
        if (m_fence)
            vkDestroyFence(m_device, m_fence, nullptr);
    }

    m_pipe = VK_NULL_HANDLE;
    m_pipe_layout = VK_NULL_HANDLE;
    m_set_layout = VK_NULL_HANDLE;
    m_desc_pool = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    m_shader = VK_NULL_HANDLE;
    m_idct_pipe = VK_NULL_HANDLE;
    m_idct_pipe_layout = VK_NULL_HANDLE;
    m_idct_layout = VK_NULL_HANDLE;
    m_idct_pool = VK_NULL_HANDLE;
    m_idct_set = VK_NULL_HANDLE;
    m_idct_shader = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_cmd = VK_NULL_HANDLE;
    m_fence = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_queue = VK_NULL_HANDLE;
    m_pipeline = false;
    uploaded = false;
}

bool MangoGpuDecoder::decodeBlocks(mango::vulkan::VulkanWindow& window,
    mango::vulkan::Allocator& allocator, DecodedTexture& texture, std::string& error)
{
    if (!uploaded)
    {
        error = "entropy buffer is not uploaded";
        return false;
    }

    m_device = window.device();
    m_queue = window.graphicsQueue();
    compile_last_us = 0;
    resolve_us = 0;

    if (!texture.image || !texture.view)
    {
        error = "DecodedTexture missing for mango-gpu";
        return false;
    }

    if (!m_pipeline)
    {
        const mango::u64 t_compile = nowUs();

        mango::vulkan::Compiler compiler;
        mango::vulkan::Shader shader = compiler.compile(kHuffmanSource, mango::vulkan::ShaderStage::Compute);
        if (!shader)
        {
            error = "huffman shader compile failed:\n" + shader.log;
            return false;
        }
        m_shader = mango::vulkan::Compiler::createShaderModule(m_device, shader);

        VkDescriptorSetLayoutBinding bindings[5] = {};
        for (uint32_t i = 0; i < 5; ++i)
            bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo set_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 5,
            .pBindings = bindings,
        };
        if (vkCreateDescriptorSetLayout(m_device, &set_info, nullptr, &m_set_layout) != VK_SUCCESS)
        {
            error = "huffman descriptor layout failed";
            return false;
        }

        VkPushConstantRange push = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushData),
        };
        VkPipelineLayoutCreateInfo pl_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push,
        };
        if (vkCreatePipelineLayout(m_device, &pl_info, nullptr, &m_pipe_layout) != VK_SUCCESS)
        {
            error = "huffman pipeline layout failed";
            return false;
        }

        VkComputePipelineCreateInfo pipe_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_shader,
                .pName = "main",
            },
            .layout = m_pipe_layout,
        };
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &m_pipe) != VK_SUCCESS)
        {
            error = "huffman pipeline failed";
            return false;
        }

        VkCommandPoolCreateInfo pool_ci = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = window.graphicsQueueFamilyIndex(),
        };
        if (vkCreateCommandPool(m_device, &pool_ci, nullptr, &m_pool) != VK_SUCCESS)
        {
            error = "huffman command pool failed";
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
            error = "huffman command buffer failed";
            return false;
        }
        VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(m_device, &fence_ci, nullptr, &m_fence) != VK_SUCCESS)
        {
            error = "huffman fence failed";
            return false;
        }

        std::vector<SegGpu> segs(stream.segments.size());
        for (size_t i = 0; i < stream.segments.size(); ++i)
        {
            const auto& seg = stream.segments[i];
            segs[i].offset = std::uint32_t(seg.packed_offset);
            segs[i].size = std::uint32_t(seg.bytes.size);
            segs[i].y0 = std::uint32_t(seg.mcu_y0);
            segs[i].y1 = std::uint32_t(seg.mcu_y1);
        }
        m_segs = allocator.createBuffer(segs.size() * sizeof(SegGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mango::vulkan::MemoryUsage::Upload, true);
        if (!m_segs || !m_segs.mapped)
        {
            error = "segment buffer failed";
            return false;
        }
        std::memcpy(m_segs.mapped, segs.data(), segs.size() * sizeof(SegGpu));
        allocator.flush(m_segs.allocation, 0, segs.size() * sizeof(SegGpu));

        TablesFlat flat = {};
        for (int i = 0; i < 128; ++i)
            flat.maxcode[i] = -1;
        for (const auto& table : stream.huffman)
        {
            const int index = table.cls * 4 + table.id;
            if (index < 0 || index >= 8)
            {
                error = "Huffman table id is outside 0..3";
                return false;
            }
            TableGpu built = {};
            buildTable(table, built);
            for (int i = 0; i < 16; ++i)
            {
                flat.mincode[index * 16 + i] = built.mincode[i];
                flat.maxcode[index * 16 + i] = built.maxcode[i];
                flat.valptr[index * 16 + i] = built.valptr[i];
            }
            std::memcpy(flat.lut + index * 256, built.lut, sizeof(built.lut));
            std::memcpy(flat.vals + index * 256, built.vals, sizeof(built.vals));
        }
        m_tables = allocator.createBuffer(sizeof(flat),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mango::vulkan::MemoryUsage::Upload, true);
        if (!m_tables || !m_tables.mapped)
        {
            error = "Huffman table buffer failed";
            return false;
        }
        std::memcpy(m_tables.mapped, &flat, sizeof(flat));
        allocator.flush(m_tables.allocation, 0, sizeof(flat));

        const std::uint64_t samples = std::uint64_t(stream.ymcu) * std::uint64_t(stream.xmcu)
            * std::uint64_t(stream.blocks_in_mcu) * 64ull;
        m_coeffs = allocator.createBuffer(samples * sizeof(std::int32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            mango::vulkan::MemoryUsage::GpuOnly, false);
        if (!m_coeffs)
        {
            error = "coefficient buffer failed";
            return false;
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
        m_qt = allocator.createBuffer(sizeof(natural_qt),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mango::vulkan::MemoryUsage::Upload, true);
        if (!m_qt || !m_qt.mapped)
        {
            error = "quant table buffer failed";
            return false;
        }
        std::memcpy(m_qt.mapped, natural_qt, sizeof(natural_qt));
        allocator.flush(m_qt.allocation, 0, sizeof(natural_qt));

        const VkDeviceSize stat_bytes = std::max<VkDeviceSize>(stream.segments.size(), 1) * sizeof(CarryGpu);
        m_stats = allocator.createBuffer(stat_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            mango::vulkan::MemoryUsage::GpuOnly, false);
        m_stats_rb = allocator.createBuffer(stat_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            mango::vulkan::MemoryUsage::Upload, true);
        if (!m_stats || !m_stats_rb || !m_stats_rb.mapped)
        {
            error = "sync stats buffer failed";
            return false;
        }

        VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };
        VkDescriptorPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        if (vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_desc_pool) != VK_SUCCESS)
        {
            error = "huffman descriptor pool failed";
            return false;
        }
        VkDescriptorSetAllocateInfo set_ai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = m_desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &m_set_layout,
        };
        if (vkAllocateDescriptorSets(m_device, &set_ai, &m_set) != VK_SUCCESS)
        {
            error = "huffman descriptor set failed";
            return false;
        }

        auto write = [](VkBuffer buffer) {
            VkDescriptorBufferInfo info = { buffer, 0, VK_WHOLE_SIZE };
            return info;
        };
        VkDescriptorBufferInfo infos[5] = {
            write(bits.buffer),
            write(m_segs.buffer),
            write(m_tables.buffer),
            write(m_coeffs.buffer),
            write(m_stats.buffer),
        };
        VkWriteDescriptorSet writes[5] = {};
        for (uint32_t i = 0; i < 5; ++i)
        {
            writes[i] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = m_set,
                .dstBinding = i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[i],
            };
        }
        vkUpdateDescriptorSets(m_device, 5, writes, 0, nullptr);

        if (!texture.image || !texture.view)
        {
            error = "DecodedTexture missing for IDCT";
            return false;
        }
        mango::vulkan::Shader idct_shader = compiler.compile(kIdctSource, mango::vulkan::ShaderStage::Compute);
        if (!idct_shader)
        {
            error = "idct shader compile failed:\n" + idct_shader.log;
            return false;
        }
        m_idct_shader = mango::vulkan::Compiler::createShaderModule(m_device, idct_shader);

        VkDescriptorSetLayoutBinding idct_bindings[3] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        VkDescriptorSetLayoutCreateInfo idct_set_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = idct_bindings,
        };
        if (vkCreateDescriptorSetLayout(m_device, &idct_set_info, nullptr, &m_idct_layout) != VK_SUCCESS)
        {
            error = "idct descriptor layout failed";
            return false;
        }
        VkPushConstantRange idct_push = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(IdctPush),
        };
        VkPipelineLayoutCreateInfo idct_pl = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_idct_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &idct_push,
        };
        if (vkCreatePipelineLayout(m_device, &idct_pl, nullptr, &m_idct_pipe_layout) != VK_SUCCESS)
        {
            error = "idct pipeline layout failed";
            return false;
        }
        VkComputePipelineCreateInfo idct_pipe_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_idct_shader,
                .pName = "main",
            },
            .layout = m_idct_pipe_layout,
        };
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &idct_pipe_info, nullptr, &m_idct_pipe) != VK_SUCCESS)
        {
            error = "idct pipeline failed";
            return false;
        }

        VkDescriptorPoolSize idct_sizes[2] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        };
        VkDescriptorPoolCreateInfo idct_pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 2,
            .pPoolSizes = idct_sizes,
        };
        if (vkCreateDescriptorPool(m_device, &idct_pool_info, nullptr, &m_idct_pool) != VK_SUCCESS)
        {
            error = "idct descriptor pool failed";
            return false;
        }
        VkDescriptorSetAllocateInfo idct_set_ai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = m_idct_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &m_idct_layout,
        };
        if (vkAllocateDescriptorSets(m_device, &idct_set_ai, &m_idct_set) != VK_SUCCESS)
        {
            error = "idct descriptor set failed";
            return false;
        }
        VkDescriptorImageInfo img_info = {
            .sampler = VK_NULL_HANDLE,
            .imageView = texture.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkDescriptorBufferInfo coeff_info = { m_coeffs.buffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo qt_info = { m_qt.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet idct_writes[3] = {};
        idct_writes[0] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_idct_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &img_info,
        };
        idct_writes[1] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_idct_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &coeff_info,
        };
        idct_writes[2] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_idct_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &qt_info,
        };
        vkUpdateDescriptorSets(m_device, 3, idct_writes, 0, nullptr);

        compile_us = nowUs() - t_compile;
        compile_last_us = compile_us;
        m_pipeline = true;
    }

    PushData push = {};
    push.xmcu = stream.xmcu;
    push.blocks = stream.blocks_in_mcu;
    const std::uint32_t chunk = chooseChunkBytes(stream);
    push.chunkBytes = std::int32_t(chunk);
    sync_chunks = std::int32_t(chunk);
    push.segCount = std::int32_t(stream.segments.size());
    const std::uint64_t coeff_limit = std::uint64_t(std::max(stream.ymcu, 0))
        * std::uint64_t(std::max(stream.xmcu, 0))
        * std::uint64_t(std::max(stream.blocks_in_mcu, 0)) * 64ull;
    push.coeffLimit = std::int32_t(std::min<std::uint64_t>(coeff_limit, 0x7fffffffull));
    for (int b = 0; b < stream.blocks_in_mcu; ++b)
    {
        const auto& block = stream.block[b];
        const std::uint32_t dc = std::uint32_t(block.dc);
        const std::uint32_t ac = std::uint32_t(4 + block.ac);
        const std::uint32_t pred = std::uint32_t(block.component);
        push.pack[b] = dc | (ac << 8) | (pred << 16);
    }

    std::uint32_t max_chunks = 1;
    for (const auto& seg : stream.segments)
    {
        const std::uint32_t n = (std::uint32_t(seg.bytes.size) + chunk - 1u) / chunk;
        max_chunks = std::max(max_chunks, std::max(n, 1u));
    }
    if (max_chunks > kSyncLanes)
    {
        error = "entropy segment does not fit in one workgroup";
        return false;
    }
    const uint32_t groups = uint32_t(stream.segments.size());
    push.waveBase = 0;

    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    std::printf("mango-gpu: sync dispatch %u segments, chunk %u bytes\n", groups, chunk);
    std::fflush(stdout);

    const mango::u64 t0 = nowUs();
    vkResetCommandBuffer(m_cmd, 0);
    vkBeginCommandBuffer(m_cmd, &begin);

    VkMemoryBarrier host = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &host, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe);
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_layout,
        0, 1, &m_set, 0, nullptr);
    vkCmdPushConstants(m_cmd, m_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(m_cmd, groups, 1, 1);

    VkMemoryBarrier stats_ready = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &stats_ready, 0, nullptr, 0, nullptr);
    const VkDeviceSize stat_bytes = VkDeviceSize(groups) * sizeof(CarryGpu);
    VkBufferCopy stats_region = { 0, 0, stat_bytes };
    vkCmdCopyBuffer(m_cmd, m_stats.buffer, m_stats_rb.buffer, 1, &stats_region);
    vkEndCommandBuffer(m_cmd);

    if (!submitAndWait(m_queue, m_cmd, m_fence, m_device, 20ull * 1000ull * 1000ull * 1000ull, error))
        return false;
    huffman_us = nowUs() - t0;

    int max_rounds = 0;
    const auto* round_ptr = static_cast<const CarryGpu*>(m_stats_rb.mapped);
    for (uint32_t i = 0; i < groups; ++i)
        max_rounds = std::max(max_rounds, int(round_ptr[i].rounds));
    sync_rounds = max_rounds;
    std::printf("mango-gpu: sync rounds max %d\n", max_rounds);
    std::fflush(stdout);

    mango::image::jpeg::JpegDecoder info_decoder(m_jpeg);
    const mango::image::jpeg::JpegInfo& info = info_decoder.info();
    IdctPush idct = {};
    idct.width = info.width;
    idct.height = info.height;
    idct.xmcu = info.xmcu;
    idct.mcu_y0 = 0;
    idct.mcu_rows = info.ymcu;
    idct.xblock = info.xblock;
    idct.yblock = info.yblock;
    idct.blocks_in_mcu = info.blocks_in_mcu;
    idct.components = info.components;
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
    setComp(0, idct.y_hsf, idct.y_vsf, idct.y_qt, idct.y_off);
    setComp(1, idct.cb_hsf, idct.cb_vsf, idct.cb_qt, idct.cb_off);
    setComp(2, idct.cr_hsf, idct.cr_vsf, idct.cr_qt, idct.cr_off);
    // 256 threads, 8 per block → 32 block slots. Must match the shader.
    const int slots = 32;
    const int bpm = std::max(info.blocks_in_mcu, 1);
    idct.mcus_per_group = std::max(slots / bpm, 1);

    vkResetCommandBuffer(m_cmd, 0);
    vkBeginCommandBuffer(m_cmd, &begin);

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags src_access = 0;
    if (texture.layout != VK_IMAGE_LAYOUT_UNDEFINED)
    {
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        src_access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }
    VkImageMemoryBarrier to_general = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = texture.layout,
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
    vkCmdPipelineBarrier(m_cmd,
        src_stage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &coeffs_ready, 0, nullptr, 1, &to_general);

    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_idct_pipe);
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_idct_pipe_layout,
        0, 1, &m_idct_set, 0, nullptr);
    vkCmdPushConstants(m_cmd, m_idct_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(idct), &idct);
    const uint32_t mcus = uint32_t(std::max(info.xmcu, 0)) * uint32_t(std::max(info.ymcu, 0));
    const uint32_t idct_groups = (mcus + uint32_t(std::max(idct.mcus_per_group, 1)) - 1u)
        / uint32_t(std::max(idct.mcus_per_group, 1));
    vkCmdDispatch(m_cmd, std::max(idct_groups, 1u), 1, 1);

    texture.recordPublishComputeStores(m_cmd);
    vkEndCommandBuffer(m_cmd);

    const mango::u64 t_idct = nowUs();
    if (!submitAndWait(m_queue, m_cmd, m_fence, m_device, 20ull * 1000ull * 1000ull * 1000ull, error))
        return false;
    resolve_us = nowUs() - t_idct;
    texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    return true;
}

} // namespace gpujpeg

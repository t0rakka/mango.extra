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

constexpr std::uint32_t kChunkBytes = 256;
constexpr int kSyncPasses = 4;
constexpr int kSyncRounds = 128;
constexpr int kStateWords = 16;

struct TableGpu
{
    std::int32_t mincode[16];
    std::int32_t maxcode[16];
    std::int32_t valptr[16];
    std::uint32_t lut[256];
    std::uint32_t vals[256];
};

struct TablesFlat
{
    std::int32_t mincode[128];
    std::int32_t maxcode[128];
    std::int32_t valptr[128];
    std::uint32_t lut[2048];
    std::uint32_t vals[2048];
};

struct ChunkGpu
{
    std::uint32_t begin;
    std::uint32_t end;
    std::uint32_t segBegin;
    std::uint32_t segEnd;
    std::uint32_t seg;
    std::uint32_t baseCoeff;
    std::uint32_t coeffEnd;
    std::uint32_t pad;
};

struct SegInfoGpu
{
    std::uint32_t chunk0;
    std::uint32_t nchunks;
    std::uint32_t baseCoeff;
    std::uint32_t pad;
};

struct SyncPush
{
    std::int32_t nchunks;
    std::int32_t pass;
    std::int32_t rounds;
    std::int32_t blocks;
    std::int32_t doWrite;
    std::uint32_t pack[10];
};
static_assert(offsetof(SyncPush, pack) == 20, "sync push layout");

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

constexpr char kSyncSource[] = R"glsl(
#version 450
layout(local_size_x = 128) in;

layout(std430, set = 0, binding = 0) readonly buffer Bits { uint bits[]; };
struct Chunk {
    uint begin;
    uint end;
    uint segBegin;
    uint segEnd;
    uint seg;
    uint baseCoeff;
    uint coeffEnd;
    uint pad;
};
layout(std430, set = 0, binding = 1) readonly buffer Chunks { Chunk chunks[]; };
layout(std430, set = 0, binding = 2) readonly buffer Tables {
    int mincode[128];
    int maxcode[128];
    int valptr[128];
    uint lut[2048];
    uint vals[2048];
} tb;
layout(std430, set = 0, binding = 3) writeonly buffer Coeffs { int coeffs[]; };
layout(std430, set = 0, binding = 4) buffer State { uint st[]; };
struct Entry {
    uint coeff;
    int p0;
    int p1;
    int p2;
    int p3;
};
layout(std430, set = 0, binding = 5) readonly buffer Entries { Entry entries[]; };
struct Cp {
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
layout(std430, set = 0, binding = 6) buffer Cps { Cp cps[]; };

layout(push_constant) uniform Push {
    int nchunks;
    int pass;
    int rounds;
    int blocks;
    int doWrite;
    uint pack[10];
} pc;

struct St {
    uint epos, enbits, ebuf, ez, ebi;
    uint pos, nbits, buf, z, bi, n;
    int d0, d1, d2, d3;
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

bool stateEq(uint pos, uint nbits, uint buf, uint z, uint bi,
             uint pos2, uint nbits2, uint buf2, uint z2, uint bi2)
{
    if (pos != pos2 || nbits != nbits2 || z != z2 || bi != bi2)
        return false;
    if (nbits == 0u)
        return true;
    if (nbits >= 32u)
        return false;
    uint mask = (1u << nbits) - 1u;
    return (buf & mask) == (buf2 & mask);
}

void storeCp(uint idx, uint pos, uint nbits, uint buf, uint z, uint bi,
             uint n, int d0, int d1, int d2, int d3)
{
    Cp o;
    o.pos = pos;
    o.nbits = nbits;
    o.buf = buf;
    o.z = z;
    o.bi = bi;
    o.n = n;
    o.d0 = d0;
    o.d1 = d1;
    o.d2 = d2;
    o.d3 = d3;
    cps[idx] = o;
}

// mode 0: decode. mode 1: also record checkpoints. mode 2: confirm, stop at the
// first checkpoint whose cursor matches and keep that walk's tail.
St walk(St start, Chunk c, bool doWrite, int p0, int p1, int p2, int p3, uint dst, uint endIdx,
        St spec, int mode, uint chunkId, uint useStop,
        uint stopPos, uint stopNbits, uint stopBuf, uint stopZ, uint stopBi)
{
    Reader r;
    r.pos = start.pos;
    r.endp = c.segEnd;
    r.buf = start.buf;
    r.nbits = start.nbits;

    St s;
    s.epos = start.pos;
    s.enbits = start.nbits;
    s.ebuf = start.buf;
    s.ez = start.z;
    s.ebi = start.bi;
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
    uint limit = c.end * 8u;
    uint cur0 = r.pos * 8u - r.nbits;
    int budget = 64;
    if (limit > cur0)
        budget = int(limit - cur0) + 64;
    uint nextK = 0u;

    for (int guard = 0; guard < budget; ++guard)
    {
        if (r.nbits > r.pos * 8u)
            break;
        uint cur = r.pos * 8u - r.nbits;
        if (cur >= limit)
            break;
        if (r.nbits == 0u && r.pos >= r.endp)
            break;
        if (useStop != 0u && stateEq(r.pos, r.nbits, r.buf, s.z, s.bi,
                stopPos, stopNbits, stopBuf, stopZ, stopBi))
            break;

        if (mode != 0)
        {
            while (nextK < 3u)
            {
                uint boundary = c.begin + (nextK + 1u) * 64u;
                if (boundary >= c.end || cur < boundary * 8u)
                    break;
                uint idx = chunkId * 4u + nextK;
                if (mode == 2)
                {
                    Cp saved = cps[idx];
                    if (saved.pos != 0xffffffffu && stateEq(r.pos, r.nbits, r.buf, s.z, s.bi,
                            saved.pos, saved.nbits, saved.buf, saved.z, saved.bi))
                    {
                        int dn = int(s.n) - int(saved.n);
                        int dd0 = s.d0 - saved.d0;
                        int dd1 = s.d1 - saved.d1;
                        int dd2 = s.d2 - saved.d2;
                        int dd3 = s.d3 - saved.d3;
                        for (uint j = nextK; j < 3u; ++j)
                        {
                            uint jdx = chunkId * 4u + j;
                            Cp tail = cps[jdx];
                            if (tail.pos == 0xffffffffu)
                                continue;
                            tail.n = uint(int(tail.n) + dn);
                            tail.d0 += dd0;
                            tail.d1 += dd1;
                            tail.d2 += dd2;
                            tail.d3 += dd3;
                            cps[jdx] = tail;
                        }
                        s.n = uint(int(spec.n) + dn);
                        s.d0 = spec.d0 + dd0;
                        s.d1 = spec.d1 + dd1;
                        s.d2 = spec.d2 + dd2;
                        s.d3 = spec.d3 + dd3;
                        s.pos = spec.pos;
                        s.nbits = spec.nbits;
                        s.buf = spec.buf;
                        s.z = spec.z;
                        s.bi = spec.bi;
                        return s;
                    }
                }
                storeCp(idx, r.pos, r.nbits, r.buf, s.z, s.bi, s.n, s.d0, s.d1, s.d2, s.d3);
                nextK += 1u;
            }
        }

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

bool cursorEq(St a, St b)
{
    if (a.epos != b.pos || a.enbits != b.nbits || a.ez != b.z || a.ebi != b.bi)
        return false;
    if (a.enbits == 0u)
        return true;
    if (a.enbits >= 32u)
        return false;
    uint mask = (1u << a.enbits) - 1u;
    return (a.ebuf & mask) == (b.buf & mask);
}

St loadSt(uint i)
{
    uint o = i * 16u;
    St s;
    s.epos = st[o + 0u];
    s.enbits = st[o + 1u];
    s.ebuf = st[o + 2u];
    s.ez = st[o + 3u];
    s.ebi = st[o + 4u];
    s.pos = st[o + 5u];
    s.nbits = st[o + 6u];
    s.buf = st[o + 7u];
    s.z = st[o + 8u];
    s.bi = st[o + 9u];
    s.n = st[o + 10u];
    s.d0 = int(st[o + 11u]);
    s.d1 = int(st[o + 12u]);
    s.d2 = int(st[o + 13u]);
    s.d3 = int(st[o + 14u]);
    return s;
}

void storeSt(uint i, St s)
{
    uint o = i * 16u;
    st[o + 0u] = s.epos;
    st[o + 1u] = s.enbits;
    st[o + 2u] = s.ebuf;
    st[o + 3u] = s.ez;
    st[o + 4u] = s.ebi;
    st[o + 5u] = s.pos;
    st[o + 6u] = s.nbits;
    st[o + 7u] = s.buf;
    st[o + 8u] = s.z;
    st[o + 9u] = s.bi;
    st[o + 10u] = s.n;
    st[o + 11u] = uint(s.d0);
    st[o + 12u] = uint(s.d1);
    st[o + 13u] = uint(s.d2);
    st[o + 14u] = uint(s.d3);
}

St guessOf(Chunk c)
{
    St s;
    s.pos = c.begin;
    s.nbits = 0u;
    s.buf = 0u;
    s.z = 0u;
    s.bi = 0u;
    s.epos = 0u;
    s.enbits = 0u;
    s.ebuf = 0u;
    s.ez = 0u;
    s.ebi = 0u;
    s.n = 0u;
    s.d0 = 0;
    s.d1 = 0;
    s.d2 = 0;
    s.d3 = 0;
    return s;
}

St exitAsStart(St prev)
{
    St s = prev;
    s.pos = prev.pos;
    s.nbits = prev.nbits;
    s.buf = prev.buf;
    s.z = prev.z;
    s.bi = prev.bi;
    return s;
}

shared St sh[128];
shared uint flag;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    bool live = i < uint(pc.nchunks);

    if (pc.doWrite != 0)
    {
        if (!live)
            return;
        Chunk c = chunks[i];
        St start = guessOf(c);
        if (c.begin != c.segBegin && i > 0u)
            start = exitAsStart(loadSt(i - 1u));
        Entry e = entries[i];
        St none = guessOf(c);
        walk(start, c, true, e.p0, e.p1, e.p2, e.p3, e.coeff, c.coeffEnd,
            none, 0, i, 0u, 0u, 0u, 0u, 0u, 0u);
        return;
    }

    St mine;
    St none = guessOf(chunks[0]);
    if (live && pc.pass == 0)
        mine = walk(guessOf(chunks[i]), chunks[i], false, 0, 0, 0, 0, 0u, 0u,
            none, 1, i, 0u, 0u, 0u, 0u, 0u, 0u);
    else if (live)
        mine = loadSt(i);

    if (lid == 0u)
        flag = 0u;
    barrier();
    sh[lid] = mine;
    barrier();

    St globPrev;
    bool haveGlob = false;
    if (live && lid == 0u && pc.pass > 0 && i > 0u && chunks[i].seg == chunks[i - 1u].seg)
    {
        globPrev = loadSt(i - 1u);
        haveGlob = true;
    }

    for (int round = 0; round < pc.rounds; ++round)
    {
        bool ch = false;
        if (live && chunks[i].begin != chunks[i].segBegin)
        {
            St prev;
            bool have = false;
            if (lid == 0u)
            {
                if (haveGlob)
                {
                    prev = globPrev;
                    have = true;
                }
            }
            else if (chunks[i].seg == chunks[i - 1u].seg)
            {
                prev = sh[lid - 1u];
                have = true;
            }
            if (have && !cursorEq(mine, prev))
            {
                St before = mine;
                mine = walk(exitAsStart(prev), chunks[i], false, 0, 0, 0, 0, 0u, 0u,
                    before, 2, i, 0u, 0u, 0u, 0u, 0u, 0u);
                ch = !stateEq(mine.pos, mine.nbits, mine.buf, mine.z, mine.bi,
                    before.pos, before.nbits, before.buf, before.z, before.bi);
            }
        }
        barrier();
        if (live)
            sh[lid] = mine;
        if (ch)
            atomicOr(flag, 1u);
        barrier();
        if (flag == 0u)
            break;
        barrier();
        if (lid == 0u)
            flag = 0u;
        barrier();
    }
    if (live)
        storeSt(i, mine);
}
)glsl";

constexpr char kPrefixSource[] = R"glsl(
#version 450
layout(local_size_x = 256) in;

struct SegInfo {
    uint chunk0;
    uint nchunks;
    uint baseCoeff;
    uint pad;
};
layout(std430, set = 0, binding = 0) readonly buffer Segs { SegInfo segs[]; };
layout(std430, set = 0, binding = 1) readonly buffer State { uint st[]; };
struct Entry {
    uint coeff;
    int p0;
    int p1;
    int p2;
    int p3;
};
layout(std430, set = 0, binding = 2) writeonly buffer Entries { Entry entries[]; };

shared uint sn[256];
shared int sd0[256];
shared int sd1[256];
shared int sd2[256];
shared int sd3[256];
shared uint run_n;
shared int run0, run1, run2, run3;

void main()
{
    uint li = gl_LocalInvocationID.x;
    SegInfo seg = segs[gl_WorkGroupID.x];
    if (li == 0u)
    {
        run_n = 0u;
        run0 = 0;
        run1 = 0;
        run2 = 0;
        run3 = 0;
    }
    barrier();

    for (uint tile = 0u; tile < seg.nchunks; tile += 256u)
    {
        uint ci = tile + li;
        bool on = ci < seg.nchunks;
        uint n = 0u;
        int d0 = 0;
        int d1 = 0;
        int d2 = 0;
        int d3 = 0;
        if (on)
        {
            uint o = (seg.chunk0 + ci) * 16u;
            n = st[o + 10u];
            d0 = int(st[o + 11u]);
            d1 = int(st[o + 12u]);
            d2 = int(st[o + 13u]);
            d3 = int(st[o + 14u]);
        }
        uint self_n = n;
        int self0 = d0, self1 = d1, self2 = d2, self3 = d3;
        sn[li] = n;
        sd0[li] = d0;
        sd1[li] = d1;
        sd2[li] = d2;
        sd3[li] = d3;
        barrier();

        for (uint off = 1u; off < 256u; off <<= 1u)
        {
            uint nn = sn[li];
            int a0 = sd0[li];
            int a1 = sd1[li];
            int a2 = sd2[li];
            int a3 = sd3[li];
            barrier();
            if (li >= off)
            {
                nn += sn[li - off];
                a0 += sd0[li - off];
                a1 += sd1[li - off];
                a2 += sd2[li - off];
                a3 += sd3[li - off];
            }
            barrier();
            sn[li] = nn;
            sd0[li] = a0;
            sd1[li] = a1;
            sd2[li] = a2;
            sd3[li] = a3;
            barrier();
        }

        if (on)
        {
            Entry e;
            e.coeff = seg.baseCoeff + run_n + (sn[li] - self_n);
            e.p0 = run0 + (sd0[li] - self0);
            e.p1 = run1 + (sd1[li] - self1);
            e.p2 = run2 + (sd2[li] - self2);
            e.p3 = run3 + (sd3[li] - self3);
            entries[seg.chunk0 + ci] = e;
        }
        barrier();
        if (li == 0u)
        {
            uint last = min(255u, seg.nchunks - tile - 1u);
            run_n += sn[last];
            run0 += sd0[last];
            run1 += sd1[last];
            run2 += sd2[last];
            run3 += sd3[last];
        }
        barrier();
    }
}
)glsl";

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
        // group_mcu0 is local to this dispatch. mcu_limit is an absolute MCU index.
        live = (group_mcu0 + mcu_local) < pc.mcu_rows * pc.xmcu;
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
        error = wr == VK_TIMEOUT ? "gpu2 fence timed out" : "vkWaitForFences failed";
        return false;
    }
    return true;
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

} // namespace

bool MangoGpu2Decoder::extract(mango::ConstMemory jpeg, std::string& error)
{
    stream = {};
    extract_us = 0;
    uploaded = false;
    m_jpeg = jpeg;
    m_pipeline = false;

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

    if (m_bits)
        allocator.destroyBuffer(m_bits);
    m_bits = {};

    const mango::u64 t0 = nowUs();
    const VkDeviceSize payload = VkDeviceSize(stream.packed_bytes);
    const VkDeviceSize bit_bytes = (payload + 16u + 3u) & ~VkDeviceSize(3);
    m_bits = allocator.createBuffer(
        bit_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        mango::vulkan::MemoryUsage::Upload,
        true);
    if (!m_bits || !m_bits.mapped)
    {
        error = "entropy upload buffer failed";
        release(allocator);
        return false;
    }
    auto* dst = static_cast<std::uint8_t*>(m_bits.mapped);
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
    allocator.flush(m_bits.allocation, 0, bit_bytes);
    m_bit_bytes = bit_bytes;
    upload_us = nowUs() - t0;
    uploaded = true;
    return true;
}

void MangoGpu2Decoder::release(mango::vulkan::Allocator& allocator)
{
    auto drop = [&](mango::vulkan::BufferAllocation& buf)
    {
        if (buf)
            allocator.destroyBuffer(buf);
        buf = {};
    };
    drop(m_bits);
    drop(m_bits_dev);
    drop(m_chunks);
    drop(m_tables);
    drop(m_coeffs);
    drop(m_state);
    drop(m_entry);
    drop(m_cps);
    drop(m_segs);
    drop(m_qt);

    if (m_device)
    {
        if (m_sync_pipe)
            vkDestroyPipeline(m_device, m_sync_pipe, nullptr);
        if (m_sync_pipe_layout)
            vkDestroyPipelineLayout(m_device, m_sync_pipe_layout, nullptr);
        if (m_sync_layout)
            vkDestroyDescriptorSetLayout(m_device, m_sync_layout, nullptr);
        if (m_sync_pool)
            vkDestroyDescriptorPool(m_device, m_sync_pool, nullptr);
        if (m_sync_shader)
            vkDestroyShaderModule(m_device, m_sync_shader, nullptr);
        if (m_prefix_pipe)
            vkDestroyPipeline(m_device, m_prefix_pipe, nullptr);
        if (m_prefix_pipe_layout)
            vkDestroyPipelineLayout(m_device, m_prefix_pipe_layout, nullptr);
        if (m_prefix_layout)
            vkDestroyDescriptorSetLayout(m_device, m_prefix_layout, nullptr);
        if (m_prefix_pool)
            vkDestroyDescriptorPool(m_device, m_prefix_pool, nullptr);
        if (m_prefix_shader)
            vkDestroyShaderModule(m_device, m_prefix_shader, nullptr);
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
        if (m_pool)
            vkDestroyCommandPool(m_device, m_pool, nullptr);
        if (m_fence)
            vkDestroyFence(m_device, m_fence, nullptr);
    }
    m_sync_pipe = m_prefix_pipe = m_idct_pipe = VK_NULL_HANDLE;
    m_sync_pipe_layout = m_prefix_pipe_layout = m_idct_pipe_layout = VK_NULL_HANDLE;
    m_sync_layout = m_prefix_layout = m_idct_layout = VK_NULL_HANDLE;
    m_sync_pool = m_prefix_pool = m_idct_pool = VK_NULL_HANDLE;
    m_sync_set = m_prefix_set = m_idct_set = VK_NULL_HANDLE;
    m_sync_shader = m_prefix_shader = m_idct_shader = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_cmd = VK_NULL_HANDLE;
    m_fence = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_queue = VK_NULL_HANDLE;
    m_pipeline = false;
    m_bit_bytes = 0;
    uploaded = false;
}

bool MangoGpu2Decoder::decode(mango::vulkan::VulkanWindow& window,
    mango::vulkan::Allocator& allocator, DecodedTexture& texture, std::string& error)
{
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
    decode_us = 0;

    const int bpm = std::max(stream.blocks_in_mcu, 1);
    std::vector<ChunkGpu> chunk_host;
    std::vector<SegInfoGpu> seg_host;
    chunk_host.reserve(stream.packed_bytes / kChunkBytes + stream.segments.size());
    for (size_t s = 0; s < stream.segments.size(); ++s)
    {
        const auto& seg = stream.segments[s];
        if (seg.bytes.size == 0)
            continue;
        SegInfoGpu info = {};
        info.chunk0 = std::uint32_t(chunk_host.size());
        info.baseCoeff = std::uint32_t(seg.mcu_y0) * std::uint32_t(stream.xmcu) * std::uint32_t(bpm) * 64u;
        const std::uint32_t coeff_end = info.baseCoeff
            + std::uint32_t(seg.mcus) * std::uint32_t(bpm) * 64u;
        const std::uint32_t origin = std::uint32_t(seg.packed_offset);
        const std::uint32_t size = std::uint32_t(seg.bytes.size);
        for (std::uint32_t off = 0; off < size; off += kChunkBytes)
        {
            ChunkGpu c = {};
            c.begin = origin + off;
            c.end = c.begin + std::min(kChunkBytes, size - off);
            c.segBegin = origin;
            c.segEnd = origin + size;
            c.seg = std::uint32_t(s);
            c.baseCoeff = info.baseCoeff;
            c.coeffEnd = coeff_end;
            chunk_host.push_back(c);
        }
        info.nchunks = std::uint32_t(chunk_host.size()) - info.chunk0;
        seg_host.push_back(info);
    }
    if (chunk_host.empty() || seg_host.empty())
    {
        error = "no entropy chunks";
        return false;
    }
    chunks = int(chunk_host.size());

    if (!m_pipeline)
    {
        const mango::u64 t_compile = nowUs();
        mango::vulkan::Compiler compiler;
        auto module_from = [&](const char* source, VkShaderModule& out) -> bool
        {
            mango::vulkan::Shader shader = compiler.compile(source, mango::vulkan::ShaderStage::Compute);
            if (!shader)
            {
                error = std::string("gpu2 shader compile failed:\n") + shader.log;
                return false;
            }
            out = mango::vulkan::Compiler::createShaderModule(m_device, shader);
            return out != VK_NULL_HANDLE;
        };
        if (!module_from(kSyncSource, m_sync_shader) || !module_from(kPrefixSource, m_prefix_shader)
            || !module_from(kIdctSource, m_idct_shader))
            return false;

        auto make_layout = [&](VkDescriptorSetLayoutBinding* bindings, uint32_t n, VkDescriptorSetLayout& layout,
                                VkPushConstantRange* push, VkPipelineLayout& pipe_layout) -> bool
        {
            VkDescriptorSetLayoutCreateInfo set_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = n,
                .pBindings = bindings,
            };
            if (vkCreateDescriptorSetLayout(m_device, &set_info, nullptr, &layout) != VK_SUCCESS)
            {
                error = "gpu2 descriptor layout failed";
                return false;
            }
            VkPipelineLayoutCreateInfo pl = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &layout,
                .pushConstantRangeCount = push ? 1u : 0u,
                .pPushConstantRanges = push,
            };
            if (vkCreatePipelineLayout(m_device, &pl, nullptr, &pipe_layout) != VK_SUCCESS)
            {
                error = "gpu2 pipeline layout failed";
                return false;
            }
            return true;
        };
        auto make_pipe = [&](VkShaderModule shader, VkPipelineLayout layout, VkPipeline& pipe) -> bool
        {
            VkComputePipelineCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = shader,
                    .pName = "main",
                },
                .layout = layout,
            };
            if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipe) != VK_SUCCESS)
            {
                error = "gpu2 pipeline failed";
                return false;
            }
            return true;
        };

        VkDescriptorSetLayoutBinding sync_bindings[7] = {};
        for (uint32_t i = 0; i < 7; ++i)
            sync_bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkPushConstantRange sync_push = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(SyncPush),
        };
        if (!make_layout(sync_bindings, 7, m_sync_layout, &sync_push, m_sync_pipe_layout)
            || !make_pipe(m_sync_shader, m_sync_pipe_layout, m_sync_pipe))
            return false;

        VkDescriptorSetLayoutBinding prefix_bindings[3] = {};
        for (uint32_t i = 0; i < 3; ++i)
            prefix_bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        if (!make_layout(prefix_bindings, 3, m_prefix_layout, nullptr, m_prefix_pipe_layout)
            || !make_pipe(m_prefix_shader, m_prefix_pipe_layout, m_prefix_pipe))
            return false;

        VkDescriptorSetLayoutBinding idct_bindings[3] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        VkPushConstantRange idct_push = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(IdctPush),
        };
        if (!make_layout(idct_bindings, 3, m_idct_layout, &idct_push, m_idct_pipe_layout)
            || !make_pipe(m_idct_shader, m_idct_pipe_layout, m_idct_pipe))
            return false;

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

        m_bits_dev = allocator.createBuffer(m_bit_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            mango::vulkan::MemoryUsage::GpuOnly, false);
        m_chunks = allocator.createBuffer(chunk_host.size() * sizeof(ChunkGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mango::vulkan::MemoryUsage::Upload, true);
        m_segs = allocator.createBuffer(seg_host.size() * sizeof(SegInfoGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mango::vulkan::MemoryUsage::Upload, true);
        const std::uint64_t samples = std::uint64_t(stream.ymcu) * std::uint64_t(stream.xmcu)
            * std::uint64_t(bpm) * 64ull;
        m_coeffs = allocator.createBuffer(samples * sizeof(std::int32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            mango::vulkan::MemoryUsage::GpuOnly, false);
        m_state = allocator.createBuffer(VkDeviceSize(chunk_host.size()) * kStateWords * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mango::vulkan::MemoryUsage::GpuOnly, false);
        m_entry = allocator.createBuffer(chunk_host.size() * sizeof(std::uint32_t) * 5u,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mango::vulkan::MemoryUsage::GpuOnly, false);
        m_cps = allocator.createBuffer(chunk_host.size() * 4u * 10u * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            mango::vulkan::MemoryUsage::GpuOnly, false);
        if (!m_bits_dev || !m_chunks || !m_chunks.mapped || !m_segs || !m_segs.mapped
            || !m_coeffs || !m_state || !m_entry || !m_cps)
        {
            error = "gpu2 buffer allocation failed";
            return false;
        }

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

        auto pool_and_set = [&](VkDescriptorType type, uint32_t n, VkDescriptorSetLayout layout,
                                VkDescriptorPool& pool, VkDescriptorSet& set) -> bool
        {
            VkDescriptorPoolSize size = { type, n };
            VkDescriptorPoolCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1,
                .poolSizeCount = 1,
                .pPoolSizes = &size,
            };
            if (vkCreateDescriptorPool(m_device, &info, nullptr, &pool) != VK_SUCCESS)
            {
                error = "gpu2 descriptor pool failed";
                return false;
            }
            VkDescriptorSetAllocateInfo ai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &layout,
            };
            return vkAllocateDescriptorSets(m_device, &ai, &set) == VK_SUCCESS;
        };
        if (!pool_and_set(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, m_sync_layout, m_sync_pool, m_sync_set)
            || !pool_and_set(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, m_prefix_layout, m_prefix_pool, m_prefix_set))
        {
            error = "gpu2 descriptor set failed";
            return false;
        }
        {
            VkDescriptorPoolSize sizes[2] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
            };
            VkDescriptorPoolCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1,
                .poolSizeCount = 2,
                .pPoolSizes = sizes,
            };
            if (vkCreateDescriptorPool(m_device, &info, nullptr, &m_idct_pool) != VK_SUCCESS)
            {
                error = "gpu2 idct descriptor pool failed";
                return false;
            }
            VkDescriptorSetAllocateInfo ai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = m_idct_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &m_idct_layout,
            };
            if (vkAllocateDescriptorSets(m_device, &ai, &m_idct_set) != VK_SUCCESS)
            {
                error = "gpu2 idct descriptor set failed";
                return false;
            }
        }

        auto buf_info = [](VkBuffer buffer) {
            VkDescriptorBufferInfo info = { buffer, 0, VK_WHOLE_SIZE };
            return info;
        };
        VkDescriptorBufferInfo sync_infos[7] = {
            buf_info(m_bits_dev.buffer),
            buf_info(m_chunks.buffer),
            buf_info(m_tables.buffer),
            buf_info(m_coeffs.buffer),
            buf_info(m_state.buffer),
            buf_info(m_entry.buffer),
            buf_info(m_cps.buffer),
        };
        VkWriteDescriptorSet sync_writes[7] = {};
        for (uint32_t i = 0; i < 7; ++i)
        {
            sync_writes[i] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = m_sync_set,
                .dstBinding = i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &sync_infos[i],
            };
        }
        vkUpdateDescriptorSets(m_device, 7, sync_writes, 0, nullptr);

        VkDescriptorBufferInfo prefix_infos[3] = {
            buf_info(m_segs.buffer),
            buf_info(m_state.buffer),
            buf_info(m_entry.buffer),
        };
        VkWriteDescriptorSet prefix_writes[3] = {};
        for (uint32_t i = 0; i < 3; ++i)
        {
            prefix_writes[i] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = m_prefix_set,
                .dstBinding = i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &prefix_infos[i],
            };
        }
        vkUpdateDescriptorSets(m_device, 3, prefix_writes, 0, nullptr);

        VkDescriptorImageInfo img_info = {
            .sampler = VK_NULL_HANDLE,
            .imageView = texture.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkDescriptorBufferInfo coeff_info = buf_info(m_coeffs.buffer);
        VkDescriptorBufferInfo qt_info = buf_info(m_qt.buffer);
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

    std::memcpy(m_chunks.mapped, chunk_host.data(), chunk_host.size() * sizeof(ChunkGpu));
    allocator.flush(m_chunks.allocation, 0, chunk_host.size() * sizeof(ChunkGpu));
    std::memcpy(m_segs.mapped, seg_host.data(), seg_host.size() * sizeof(SegInfoGpu));
    allocator.flush(m_segs.allocation, 0, seg_host.size() * sizeof(SegInfoGpu));

    SyncPush push = {};
    push.nchunks = int(chunk_host.size());
    push.rounds = kSyncRounds;
    push.blocks = stream.blocks_in_mcu;
    for (int b = 0; b < stream.blocks_in_mcu; ++b)
    {
        const auto& block = stream.block[b];
        push.pack[b] = std::uint32_t(block.dc)
            | (std::uint32_t(4 + block.ac) << 8)
            | (std::uint32_t(block.component) << 16);
    }

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
    const int slots = 32;
    idct.mcus_per_group = std::max(slots / bpm, 1);

    vkResetCommandBuffer(m_cmd, 0);
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(m_cmd, &begin);

    memBarrier(m_cmd, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy region = { 0, 0, m_bit_bytes };
    vkCmdCopyBuffer(m_cmd, m_bits.buffer, m_bits_dev.buffer, 1, &region);
    memBarrier(m_cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdFillBuffer(m_cmd, m_coeffs.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(m_cmd, m_cps.buffer, 0, VK_WHOLE_SIZE, 0xffffffffu);
    memBarrier(m_cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    const uint32_t groups = (uint32_t(chunk_host.size()) + 127u) / 128u;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sync_pipe);
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sync_pipe_layout,
        0, 1, &m_sync_set, 0, nullptr);
    for (int pass = 0; pass < kSyncPasses; ++pass)
    {
        push.pass = pass;
        push.doWrite = 0;
        vkCmdPushConstants(m_cmd, m_sync_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(m_cmd, groups, 1, 1);
        memBarrier(m_cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prefix_pipe);
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prefix_pipe_layout,
        0, 1, &m_prefix_set, 0, nullptr);
    vkCmdDispatch(m_cmd, uint32_t(seg_host.size()), 1, 1);
    memBarrier(m_cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    push.doWrite = 1;
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sync_pipe);
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sync_pipe_layout,
        0, 1, &m_sync_set, 0, nullptr);
    vkCmdPushConstants(m_cmd, m_sync_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(m_cmd, groups, 1, 1);

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
    vkCmdPipelineBarrier(m_cmd,
        src_stage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &coeffs_ready, 0, nullptr, 1, &to_general);
    texture.layout = VK_IMAGE_LAYOUT_GENERAL;

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

    const mango::u64 t0 = nowUs();
    if (!submitAndWait(m_queue, m_cmd, m_fence, m_device, 20ull * 1000ull * 1000ull * 1000ull, error))
        return false;
    decode_us = nowUs() - t0;
    return true;
}

} // namespace gpujpeg

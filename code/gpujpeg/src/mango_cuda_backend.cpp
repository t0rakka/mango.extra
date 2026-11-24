// Copyright 2026 Distance Technologies Oy. For internal use only.
//
#include "mango_cuda_backend.hpp"
#include "mango_cuda_launch.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <vector>

namespace gpujpeg
{
namespace
{

int symEntryHost(int sym, int len, bool dc)
{
    int s = sym & 15;
    int r = sym >> 4;
    int za = dc ? 1 : (s ? r + 1 : (r == 15 ? 16 : 64));
    return (len + s) | (s << 5) | (za << 9);
}

void buildSlot(const mango::image::jpeg::HuffmanCodeTable& src, bool dc, CudaTables& dst, int slot)
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
            for (int k = 0; k < nl; ++k)
            {
                int sym = src.values[p + k];
                dst.hv[slot * 256 + p + k] = std::uint8_t(sym);
                if (len <= kCudaLb)
                {
                    int lo = (code + k) << (kCudaLb - len);
                    int n = 1 << (kCudaLb - len);
                    std::uint16_t ent = std::uint16_t(symEntryHost(sym, len, dc));
                    for (int i = 0; i < n; ++i)
                        dst.lut[slot * kCudaLut + lo + i] = ent;
                }
            }
            p += nl;
            code += nl;
            dst.maxc[slot * 18 + len] = code - 1;
        }
        code <<= 1;
    }
}

bool cudaOk(cudaError_t e, const char* what, std::string& error)
{
    if (e == cudaSuccess)
        return true;
    error = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

} // namespace

struct MangoCudaDecoder::Gpu
{
    cudaStream_t stream = nullptr;
    unsigned* words = nullptr;
    unsigned nwords = 0;
    unsigned* hostWords = nullptr;
    CudaChunk* chunks = nullptr;
    int nchunks = 0;
    CudaSeg* segs = nullptr;
    int nsegs = 0;
    CudaTables* tables = nullptr;
    unsigned long long* stIn = nullptr;
    unsigned long long* stOut = nullptr;
    int* cnt = nullptr;
    unsigned long long* cps = nullptr;
    int* cpc = nullptr;
    int* firstb = nullptr;
    short* coef = nullptr;
    int* dcd = nullptr;
    int* qt = nullptr;
    int nblk = 0;
    CudaImg img {};
    cudaEvent_t ev[4] {};
};

void MangoCudaDecoder::freeGpu(Gpu* g)
{
    if (!g)
        return;
    cudaFree(g->words);
    cudaFree(g->chunks);
    cudaFree(g->segs);
    cudaFree(g->tables);
    cudaFree(g->stIn);
    cudaFree(g->stOut);
    cudaFree(g->cnt);
    cudaFree(g->cps);
    cudaFree(g->cpc);
    cudaFree(g->firstb);
    cudaFree(g->coef);
    cudaFree(g->dcd);
    cudaFree(g->qt);
    if (g->hostWords)
        cudaFreeHost(g->hostWords);
    for (int i = 0; i < 4; ++i)
    {
        if (g->ev[i])
            cudaEventDestroy(g->ev[i]);
    }
    if (g->stream)
        cudaStreamDestroy(g->stream);
    *g = {};
}

bool MangoCudaDecoder::extract(mango::ConstMemory jpeg, std::string& error)
{
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    m_jpeg = jpeg;
    release();
    mango::image::jpeg::JpegDecoder decoder(jpeg);
    stream = decoder.extractEntropy();
    if (!stream.ok)
    {
        error = stream.error.empty() ? "extractEntropy failed" : stream.error;
        return false;
    }
    if (stream.blocks_in_mcu <= 0 || stream.blocks_in_mcu > 10)
    {
        error = "unsupported MCU block count";
        return false;
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(stream.packed_bytes);
    std::vector<CudaChunk> chunkHost;
    std::vector<CudaSeg> segHost;
    unsigned bit = 0;
    int block = 0;
    int mcu = 0;
    for (const auto& seg : stream.segments)
    {
        unsigned segBit = bit;
        const auto* p = seg.bytes.address;
        bool ff = false;
        for (size_t i = 0; i < seg.bytes.size; ++i)
        {
            std::uint8_t b = p[i];
            if (ff && b == 0)
            {
                ff = false;
                continue;
            }
            ff = b == 0xff;
            bytes.push_back(b);
            bit += 8;
        }
        unsigned segEnd = bit;
        CudaSeg sg {};
        sg.chunk0 = int(chunkHost.size());
        sg.block0 = block;
        sg.mcu0 = mcu;
        sg.mcus = seg.mcus;
        for (unsigned at = segBit; at < segEnd; at += kCudaChunkBits)
        {
            CudaChunk c {};
            c.bit0 = at;
            c.bit1 = std::min(at + kCudaChunkBits, segEnd);
            c.seg0 = segBit;
            c.seg1 = segEnd;
            c.block0 = block;
            c.block1 = block + seg.mcus * stream.blocks_in_mcu;
            chunkHost.push_back(c);
        }
        sg.nchunks = int(chunkHost.size()) - sg.chunk0;
        segHost.push_back(sg);
        block += seg.mcus * stream.blocks_in_mcu;
        mcu += seg.mcus;
    }
    if (chunkHost.empty())
    {
        error = "no entropy chunks";
        return false;
    }
    chunks = int(chunkHost.size());

    CudaTables tab {};
    for (const auto& table : stream.huffman)
    {
        int slot = table.cls * 4 + table.id;
        if (slot < 0 || slot >= 8)
        {
            error = "Huffman table id is outside 0..3";
            return false;
        }
        buildSlot(table, table.cls == 0, tab, slot);
    }

    CudaImg img {};
    mango::image::jpeg::JpegDecoder infoDec(jpeg);
    const auto& info = infoDec.info();
    img.width = info.width;
    img.height = info.height;
    img.xmcu = info.xmcu;
    img.xblock = info.xblock;
    img.yblock = info.yblock;
    img.bpm = info.blocks_in_mcu;
    img.components = info.components;
    int off = 0;
    auto setComp = [&](int idx, int& hsf, int& vsf, int& qt, int& o)
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
    setComp(0, img.y_hsf, img.y_vsf, img.y_qt, img.y_off);
    setComp(1, img.cb_hsf, img.cb_vsf, img.cb_qt, img.cb_off);
    setComp(2, img.cr_hsf, img.cr_vsf, img.cr_qt, img.cr_off);
    img.mcus_per_group = std::max(32 / std::max(img.bpm, 1), 1);
    for (int b = 0; b < stream.blocks_in_mcu; ++b)
    {
        const auto& blockInfo = stream.block[b];
        img.dc_pack |= unsigned(blockInfo.dc & 7) << (3 * b);
        img.ac_pack |= unsigned((4 + blockInfo.ac) & 7) << (3 * b);
        img.comp_pack |= unsigned(blockInfo.component & 3) << (2 * b);
    }

    int qtHost[4 * 64] = {};
    for (int t = 0; t < 4; ++t)
    {
        if (!info.quant[t])
            continue;
        for (int i = 0; i < 64; ++i)
            qtHost[t * 64 + i] = info.quant[t][i];
    }

    auto gpu = std::make_unique<Gpu>();
    gpu->img = img;
    gpu->nchunks = int(chunkHost.size());
    gpu->nsegs = int(segHost.size());
    gpu->nblk = block;
    size_t nbytes = (bytes.size() + 3u) & ~size_t(3);
    bytes.resize(nbytes + 32, 0);
    gpu->nwords = unsigned((nbytes + 32) / 4);

    auto fail = [&](cudaError_t e, const char* what) {
        error = std::string(what) + ": " + cudaGetErrorString(e);
        freeGpu(gpu.get());
        return false;
    };
    if (cudaFree(0) != cudaSuccess)
        return fail(cudaGetLastError(), "cuda init");
    if (cudaStreamCreate(&gpu->stream) != cudaSuccess)
        return fail(cudaGetLastError(), "cudaStreamCreate");
    if (cudaMallocHost(&gpu->hostWords, gpu->nwords * sizeof(unsigned)) != cudaSuccess)
        return fail(cudaGetLastError(), "cudaMallocHost");
    std::memcpy(gpu->hostWords, bytes.data(), gpu->nwords * sizeof(unsigned));
    if (cudaMalloc(&gpu->words, gpu->nwords * sizeof(unsigned)) != cudaSuccess)
        return fail(cudaGetLastError(), "words");
    if (cudaMalloc(&gpu->chunks, chunkHost.size() * sizeof(CudaChunk)) != cudaSuccess)
        return fail(cudaGetLastError(), "chunks");
    if (cudaMalloc(&gpu->segs, segHost.size() * sizeof(CudaSeg)) != cudaSuccess)
        return fail(cudaGetLastError(), "segs");
    if (cudaMalloc(&gpu->tables, sizeof(CudaTables)) != cudaSuccess)
        return fail(cudaGetLastError(), "tables");
    if (cudaMalloc(&gpu->stIn, chunkHost.size() * sizeof(unsigned long long)) != cudaSuccess)
        return fail(cudaGetLastError(), "stIn");
    if (cudaMalloc(&gpu->stOut, chunkHost.size() * sizeof(unsigned long long)) != cudaSuccess)
        return fail(cudaGetLastError(), "stOut");
    if (cudaMalloc(&gpu->cnt, chunkHost.size() * sizeof(int)) != cudaSuccess)
        return fail(cudaGetLastError(), "cnt");
    if (cudaMalloc(&gpu->cps, chunkHost.size() * kCudaNcp * sizeof(unsigned long long)) != cudaSuccess)
        return fail(cudaGetLastError(), "cps");
    if (cudaMalloc(&gpu->cpc, chunkHost.size() * kCudaNcp * sizeof(int)) != cudaSuccess)
        return fail(cudaGetLastError(), "cpc");
    if (cudaMalloc(&gpu->firstb, chunkHost.size() * sizeof(int)) != cudaSuccess)
        return fail(cudaGetLastError(), "first");
    if (cudaMalloc(&gpu->coef, std::size_t(block) * 64 * sizeof(short)) != cudaSuccess)
        return fail(cudaGetLastError(), "coef");
    if (cudaMalloc(&gpu->dcd, std::size_t(block) * sizeof(int)) != cudaSuccess)
        return fail(cudaGetLastError(), "dcd");
    if (cudaMalloc(&gpu->qt, sizeof(qtHost)) != cudaSuccess)
        return fail(cudaGetLastError(), "qt");
    for (int i = 0; i < 4; ++i)
    {
        if (cudaEventCreate(&gpu->ev[i]) != cudaSuccess)
            return fail(cudaGetLastError(), "event");
    }
    if (cudaMemcpy(gpu->chunks, chunkHost.data(), chunkHost.size() * sizeof(CudaChunk), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(cudaGetLastError(), "upload chunks");
    if (cudaMemcpy(gpu->segs, segHost.data(), segHost.size() * sizeof(CudaSeg), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(cudaGetLastError(), "upload segs");
    if (cudaMemcpy(gpu->tables, &tab, sizeof(tab), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(cudaGetLastError(), "upload tables");
    if (cudaMemcpy(gpu->qt, qtHost, sizeof(qtHost), cudaMemcpyHostToDevice) != cudaSuccess)
        return fail(cudaGetLastError(), "upload qt");

    m_gpu = gpu.release();
    extract_us = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
    return true;
}

bool MangoCudaDecoder::decode(void* rgba, std::string& error)
{
    if (!m_gpu || !rgba)
    {
        error = "mango-cuda is not ready";
        return false;
    }
    Gpu& g = *m_gpu;
    cudaStream_t st = g.stream;
    if (!cudaOk(cudaEventRecord(g.ev[0], st), "event0", error))
        return false;
    if (!cudaOk(cudaMemcpyAsync(g.words, g.hostWords, g.nwords * sizeof(unsigned), cudaMemcpyHostToDevice, st), "copy bits", error))
        return false;
    if (!cudaOk(cudaMemsetAsync(g.coef, 0, std::size_t(g.nblk) * 64 * sizeof(short), st), "clear coef", error))
        return false;
    if (!cudaOk(cudaMemsetAsync(g.dcd, 0, std::size_t(g.nblk) * sizeof(int), st), "clear dc", error))
        return false;

    CudaFrame frame {};
    frame.words = g.words;
    frame.nwords = g.nwords;
    frame.chunks = g.chunks;
    frame.nchunks = g.nchunks;
    frame.segs = g.segs;
    frame.nsegs = g.nsegs;
    frame.tables = g.tables;
    frame.img = g.img;
    frame.stIn = g.stIn;
    frame.stOut = g.stOut;
    frame.cnt = g.cnt;
    frame.cps = g.cps;
    frame.cpc = g.cpc;
    frame.firstb = g.firstb;
    frame.coef = g.coef;
    frame.dcd = g.dcd;
    frame.nblk = g.nblk;
    frame.qt = g.qt;
    frame.rgba = static_cast<unsigned char*>(rgba);
    frame.stream = st;

    mangoCudaHuffman(frame);
    if (!cudaOk(cudaEventRecord(g.ev[1], st), "event1", error))
        return false;
    if (!cudaOk(cudaGetLastError(), "huffman launch", error))
        return false;

    mangoCudaPixels(frame);
    if (!cudaOk(cudaEventRecord(g.ev[2], st), "event2", error))
        return false;
    if (!cudaOk(cudaGetLastError(), "idct launch", error))
        return false;
    if (!cudaOk(cudaEventSynchronize(g.ev[2]), "sync", error))
        return false;
    float huffMs = 0, pixMs = 0;
    cudaEventElapsedTime(&huffMs, g.ev[0], g.ev[1]);
    cudaEventElapsedTime(&pixMs, g.ev[1], g.ev[2]);
    huffman_us = mango::u64(huffMs * 1000.0f);
    pixel_us = mango::u64(pixMs * 1000.0f);
    decode_us = huffman_us + pixel_us;
    copy_us = 0;
    return true;
}

void MangoCudaDecoder::release()
{
    if (!m_gpu)
        return;
    freeGpu(m_gpu);
    delete m_gpu;
    m_gpu = nullptr;
}

} // namespace gpujpeg

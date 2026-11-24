// Copyright 2026 Distance Technologies Oy. For internal use only.
//
// POD layout shared by the CUDA kernels and the host decoder.
// No mango / fmt includes: nvcc 12.3 cannot parse those headers.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gpujpeg
{

constexpr int kCudaLb = 10;
constexpr int kCudaLut = 1 << kCudaLb;
constexpr int kCudaNcp = 7;
constexpr int kCudaPieces = kCudaNcp + 1;
constexpr int kCudaChunkBits = 2048;
constexpr int kCudaSyncThreads = 128;
constexpr int kCudaPasses = 3;
constexpr int kCudaRounds = 128;

struct CudaChunk
{
    unsigned bit0;
    unsigned bit1;
    unsigned seg0;
    unsigned seg1;
    int block0;
    int block1;
};

struct CudaSeg
{
    int chunk0;
    int nchunks;
    int block0;
    int mcu0;
    int mcus;
};

struct CudaTables
{
    std::uint16_t lut[8 * kCudaLut];
    int maxc[8 * 18];
    int voff[8 * 18];
    std::uint8_t hv[8 * 256];
};

struct CudaImg
{
    int width, height, xmcu, xblock, yblock, bpm, components;
    int y_hsf, y_vsf, y_qt, y_off;
    int cb_hsf, cb_vsf, cb_qt, cb_off;
    int cr_hsf, cr_vsf, cr_qt, cr_off;
    int mcus_per_group;
    unsigned dc_pack;
    unsigned ac_pack;
    unsigned comp_pack;
};

struct CudaFrame
{
    unsigned* words;
    unsigned nwords;
    CudaChunk* chunks;
    int nchunks;
    CudaSeg* segs;
    int nsegs;
    CudaTables* tables;
    CudaImg img;
    unsigned long long* stIn;
    unsigned long long* stOut;
    int* cnt;
    unsigned long long* cps;
    int* cpc;
    int* firstb;
    short* coef;
    int* dcd;
    int nblk;
    int* qt;
    unsigned char* rgba;
    cudaStream_t stream;
};

// Huffman sync, prefix, write, and the DC scan. Pixels are a separate launch
// so the host can time the two halves.
void mangoCudaHuffman(const CudaFrame& frame);
void mangoCudaPixels(const CudaFrame& frame);

} // namespace gpujpeg

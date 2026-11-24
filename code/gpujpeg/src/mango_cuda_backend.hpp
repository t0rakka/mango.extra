// Copyright 2026 Distance Technologies Oy. For internal use only.
//
#pragma once

#include "bench.hpp"

#include <cstdint>
#include <string>

#include <mango/core/memory.hpp>
#include <mango/image/jpeg.hpp>

namespace gpujpeg
{

// Wide self-sync Huffman + IDCT + YCbCr, the same shape as the CUDA decoder
// that lands under 2 ms, writing tightly packed RGBA into an exported device pointer.
class MangoCudaDecoder
{
public:
    mango::image::jpeg::EntropyStream stream;
    mango::u64 extract_us = 0;
    mango::u64 decode_us = 0;
    mango::u64 copy_us = 0;
    mango::u64 huffman_us = 0;
    mango::u64 pixel_us = 0;
    int chunks = 0;

    bool extract(mango::ConstMemory jpeg, std::string& error);
    // rgba is a device pointer, tightly packed RGBA8, width * height * 4 bytes.
    bool decode(void* rgba, std::string& error);
    void release();

    MangoCudaDecoder() = default;
    ~MangoCudaDecoder() { release(); }
    MangoCudaDecoder(const MangoCudaDecoder&) = delete;
    MangoCudaDecoder& operator=(const MangoCudaDecoder&) = delete;

private:
    struct Gpu;
    static void freeGpu(Gpu* gpu);
    Gpu* m_gpu = nullptr;
    mango::ConstMemory m_jpeg;
};

} // namespace gpujpeg

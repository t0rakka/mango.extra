#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <mango/core/memory.hpp>
#include <mango/image/jpeg.hpp>

namespace gpujpeg
{

struct MangoHybridHuffmanResult
{
    bool ok = false;
    std::string error;

    int width = 0;
    int height = 0;
    int components = 0;
    int ymcu = 0;
    int bands = 0;
    std::size_t samples = 0; // total s16 coeffs emitted

    bool progressive = false;
    bool mango1 = false;
    bool dri = false;
    std::string chroma;
    std::string encoding;

    // Wall time of decodeCoefficients() with CoeffOrder::Spectral and a no-op sink.
    // This is the Huffman (no zigzag) floor — compute IDCT+color must beat
    // (fullCPU - this) to win on the hybrid path.
    mango::u64 huffman_us = 0;
};

// True when sequential (or progressive) DCT entropy → Spectral coeffs is viable.
// Lossless / UltraHDR / arithmetic / differential are out of scope for hybrid v1.
bool mangoHybridEligible(const mango::image::jpeg::JpegInfo& info, std::string& reason);

// Parse + Spectral Huffman into recycled bands; sink only releases ownership.
MangoHybridHuffmanResult runMangoHybridHuffman(mango::ConstMemory jpeg);

} // namespace gpujpeg

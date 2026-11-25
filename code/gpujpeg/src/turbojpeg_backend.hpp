/*
    MANGO Multimedia Development Platform
    Copyright (C) 2012-2026 Twilight Finland 3D Oy Ltd. All rights reserved.
*/

#pragma once

#include "bench.hpp"

#include <string>

namespace gpujpeg
{

// libjpeg-turbo TurboJPEG API. CPU SIMD, one thread per image.
// JPEG → host RGBA8 → upload → VkImage.

struct TurboJpegRuntimeInfo
{
    bool built = false;
    std::string error;
};

TurboJpegRuntimeInfo queryTurboJpegRuntime();
bool turboJpegAvailable();

struct TurboJpegDecoder
{
    bool ok = false;
    int width = 0;
    int height = 0;
    std::string codec_name;

    mango::u64 init_us = 0;
    mango::u64 warmup_us = 0;

    bool init(int width, int height, std::string& error);
    bool warmup(mango::ConstMemory jpeg, int rounds, std::string& error);
    bool decodeToHost(mango::ConstMemory jpeg, void* dest, size_t dest_bytes,
        mango::u64& decode_us, std::string& error);
    void shutdown();

    TurboJpegDecoder() = default;
    ~TurboJpegDecoder() { shutdown(); }
    TurboJpegDecoder(const TurboJpegDecoder&) = delete;
    TurboJpegDecoder& operator=(const TurboJpegDecoder&) = delete;
    TurboJpegDecoder(TurboJpegDecoder&&) = delete;
    TurboJpegDecoder& operator=(TurboJpegDecoder&&) = delete;

private:
    void* m_handle = nullptr;
};

} // namespace gpujpeg

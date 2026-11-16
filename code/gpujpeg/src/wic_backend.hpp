// Copyright 2026 Distance Technologies Oy. For internal use only.
//

#pragma once

#include "bench.hpp"

#include <string>

namespace gpujpeg
{

// Windows Imaging Component inbox JPEG decoder (Windows only).
// JPEG → WIC → host RGBA8 → upload → VkImage.
// Same traffic class as mango (host→VkImage). Not a hardware ingest path:
// the inbox codec is single-threaded software.

struct WicRuntimeInfo
{
    bool built = false;
    std::string error;
};

WicRuntimeInfo queryWicRuntime();
bool wicAvailable();

struct WicDecoder
{
    bool ok = false;
    int width = 0;
    int height = 0;
    std::string codec_name;

    mango::u64 init_us = 0;
    mango::u64 warmup_us = 0;

    bool init(int width, int height, std::string& error);
    bool warmup(mango::ConstMemory jpeg, int rounds, std::string& error);

    // Bitstream → tightly packed host RGBA8 in `dest` (size >= width*height*4).
    // decode_us is stream open through CopyPixels (no upload).
    bool decodeToHost(mango::ConstMemory jpeg, void* dest, size_t dest_bytes,
        mango::u64& decode_us, std::string& error);

    void shutdown();

    WicDecoder() = default;
    ~WicDecoder() { shutdown(); }
    WicDecoder(const WicDecoder&) = delete;
    WicDecoder& operator=(const WicDecoder&) = delete;
    WicDecoder(WicDecoder&&) = delete;
    WicDecoder& operator=(WicDecoder&&) = delete;

private:
    void* m_impl = nullptr;
};

} // namespace gpujpeg

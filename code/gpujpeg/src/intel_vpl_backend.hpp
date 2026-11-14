#pragma once

#include "bench.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gpujpeg
{

// Intel oneVPL / Quick Sync JPEG (curiosity backend).
// On a discrete-GPU Vulkan workbench (e.g. NVIDIA):
//   JPEG → iGPU QSV (UMA) → map to host RGBA → upload → VkImage
// Path label: igpu→host→VkImage (same traffic class as mango CPU).
// Available = CMake found libvpl ∩ hardware JPEG session creates at runtime.

struct IntelVplRuntimeInfo
{
    bool dispatcher = false;
    bool hardware_jpeg = false;
    std::string impl_name;
    std::string error;
};

IntelVplRuntimeInfo queryIntelVplRuntime();
bool intelVplAvailable();

// Persistent decoder: init once, decode many. Output is host RGBA8.
struct IntelVplDecoder
{
    bool ok = false;
    int width = 0;
    int height = 0;

    mango::u64 init_us = 0;
    mango::u64 warmup_us = 0;

    bool init(int width, int height, std::string& error);
    bool warmup(mango::ConstMemory jpeg, int rounds, std::string& error);

    // Bitstream → tightly packed host RGBA8 in `dest` (size >= width*height*4).
    // decode_us covers QSV submit→sync + NV12/I420→RGBA convert (no upload).
    bool decodeToHost(mango::ConstMemory jpeg, void* dest, size_t dest_bytes,
        mango::u64& decode_us, std::string& error);

    void shutdown();

    IntelVplDecoder() = default;
    ~IntelVplDecoder() { shutdown(); }
    IntelVplDecoder(const IntelVplDecoder&) = delete;
    IntelVplDecoder& operator=(const IntelVplDecoder&) = delete;

private:
    void* m_impl = nullptr;
};

} // namespace gpujpeg

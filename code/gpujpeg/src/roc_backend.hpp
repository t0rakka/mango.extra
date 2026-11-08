#pragma once

#include "bench.hpp"

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace gpujpeg
{

// rocJPEG backends (rocjpeg.h, rocJPEG 1.4 — the API in ROCm 7.2):
//   Hardware = ROCJPEG_BACKEND_HARDWARE — VCN JPEG ASIC (full decode on the media engine)
//   Hybrid   = ROCJPEG_BACKEND_HYBRID   — declared by the SDK; 1.4 returns not-implemented
// Available at runtime only if rocJpegCreate() succeeds on the Vulkan device's HIP peer.
enum class RocJpegMode
{
    Hardware,
    Hybrid,
};

inline const char* rocJpegModeName(RocJpegMode mode)
{
    switch (mode)
    {
        case RocJpegMode::Hardware: return "hardware";
        case RocJpegMode::Hybrid:   return "hybrid";
    }
    return "?";
}

// One HIP device, after a rocJpegCreate probe (handle is destroyed again).
struct RocDeviceStatus
{
    int index = -1;
    std::string name;
    std::string arch;
    bool hardware = false;
    bool hybrid = false;
    std::string hardware_error;
    std::string hybrid_error;
};

struct RocRuntimeInfo
{
    bool hip_present = false;
    bool hardware = false; // any HIP device accepted HARDWARE
    bool hybrid = false;
    std::string error;
    std::vector<RocDeviceStatus> devices;
};

RocRuntimeInfo queryRocRuntime();
bool rocAvailable();

// File → HIP RGB (interleaved). No D2H.
// rocJPEG 1.4 decode is one blocking VCN submit. inflight sizes RGB slots;
// a second submit does not add a second JPEG core on Navi.
struct RocJpegDecoder
{
    bool ok = false;
    RocJpegMode mode = RocJpegMode::Hardware;
    int width = 0;
    int height = 0;
    int inflight = 1;
    int hip_device = -1;
    std::string device_name;
    std::string arch_name;

    mango::u64 init_us = 0;
    mango::u64 warmup_us = 0;

    // physical: Vulkan device that will own the destination VkImage. HIP peer is matched here.
    bool init(RocJpegMode mode, int width, int height, int inflight,
        VkPhysicalDevice physical, std::string& error);
    bool warmup(mango::ConstMemory jpeg, int rounds, std::string& error);

    // Bitstream → HIP RGB. Blocks until the VCN/HIP write is complete.
    bool decodeToDevice(mango::ConstMemory jpeg, mango::u64& decode_us, std::string& error);

    // Device RGB → tightly packed RGBA8 in an already-mapped HIP pointer (the Vk export).
    // Synchronizes the device before return so a following Vulkan copy sees the bytes.
    bool copyRgbToRgba(void* dst_rgba, int slot, std::string& error);

    const char* subsampling() const; // "4:2:0", or "" before the first successful parse

    void shutdown();

    RocJpegDecoder() = default;
    ~RocJpegDecoder() { shutdown(); }
    RocJpegDecoder(const RocJpegDecoder&) = delete;
    RocJpegDecoder& operator=(const RocJpegDecoder&) = delete;

private:
    void* m_impl = nullptr;
    const char* m_subsampling = "";
};

} // namespace gpujpeg

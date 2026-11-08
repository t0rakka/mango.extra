#pragma once

#include "bench.hpp"
#include <cstdint>
#include <mango/core/memory.hpp>

namespace gpujpeg
{

// nvJPEG backends (nvjpeg.h):
//   Hybrid = NVJPEG_BACKEND_HYBRID     — Huffman on CPU ("some on GPU")
//   Gpu    = NVJPEG_BACKEND_GPU_HYBRID — GPU-assisted Huffman ("more/all on GPU")
enum class NvJpegMode
{
    Hybrid,
    Gpu,
};

inline const char* nvJpegModeName(NvJpegMode mode)
{
    switch (mode)
    {
        case NvJpegMode::Hybrid: return "hybrid";
        case NvJpegMode::Gpu:    return "gpu";
    }
    return "?";
}

bool nvjpegAvailable();

// Persistent decoder: init + warmup once, then decode / pipeline many times.
// Design path: file (host bitstream) → GPU pixel buffer. No host round-trip.
struct NvJpegDecoder
{
    bool ok = false;
    NvJpegMode mode = NvJpegMode::Hybrid;
    int width = 0;
    int height = 0;
    int inflight = 1;

    mango::u64 init_us = 0;
    mango::u64 warmup_us = 0;

    bool init(NvJpegMode mode, int width, int height, int inflight, std::string& error);
    bool warmup(mango::ConstMemory jpeg, int rounds, std::string& error);

    // File → GPU memory ready (nvjpegDecode + stream sync). No D2H. Uses slot 0.
    bool decodeToDevice(mango::ConstMemory jpeg, mango::u64& decode_us, std::string& error);

    // Multi-slot async API (depth = inflight).
    bool decodeLaunch(mango::ConstMemory jpeg, int slot, std::string& error); // no sync
    bool syncSlot(int slot, std::string& error);

    std::uint8_t* deviceRgbi(int slot = 0) const;
    int rgbiPitch() const; // bytes per row (= width * 3)
    void* decodeStream(int slot = 0) const; // cudaStream_t

    // Overlap `count` decodes of the same bitstream across `inflight` streams.
    bool decodePipeline(mango::ConstMemory jpeg, int count,
        mango::u64& total_us, mango::u64& avg_us, std::string& error);

    void shutdown();

    NvJpegDecoder() = default;
    ~NvJpegDecoder() { shutdown(); }
    NvJpegDecoder(const NvJpegDecoder&) = delete;
    NvJpegDecoder& operator=(const NvJpegDecoder&) = delete;

private:
    void* m_impl = nullptr;
};

} // namespace gpujpeg

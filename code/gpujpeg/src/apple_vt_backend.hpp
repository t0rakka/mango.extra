#pragma once

#if defined(GPUJPEG_HAS_APPLE_VT)

#include <cstdint>
#include <string>

#include <mango/core/core.hpp>

struct __CVBuffer;
typedef struct __CVBuffer* CVPixelBufferRef;

namespace gpujpeg
{

struct AppleVtAvailability
{
    bool ok = false;
    bool hardware_jpeg = false;
    std::string error;
};

struct AppleVtDecodeTimings
{
    mango::u64 total_us = 0;
    mango::u64 probe_us = 0;    // mango JpegDecoder eligibility (0 when skipped)
    mango::u64 session_us = 0;  // VTDecompressionSessionCreate (0 when cached)
    mango::u64 vt_wait_us = 0;  // DecodeFrame + WaitForAsynchronousFrames
    mango::u64 resolve_us = 0;  // GPU CI YCbCr→BGRA (0 when VT emits BGRA)
};

struct AppleVtDecodeResult
{
    bool ok = false;
    std::string error;
    mango::u64 decode_us = 0; // sum of timings below (compat)
    AppleVtDecodeTimings detail;
    int width = 0;
    int height = 0;
    bool hardware_capable = false; // VTIsHardwareDecodeSupported(JPEG)
    bool hardware = false;         // UsingHardwareAcceleratedVideoDecoder, or inferred from Require*
    bool hardware_property_read = false; // false if VT did not expose the readback key (JPEG)
    std::string path; // "videotoolbox" (VT → IOSurface; no host pixel path)
    CVPixelBufferRef pixel_buffer = nullptr; // retained; release with AppleVtReleasePixelBuffer
};

AppleVtAvailability probeAppleVt();
void AppleVtReleasePixelBuffer(CVPixelBufferRef buffer);

// dimensions optional: if >0, skips redundant mango probe on hot path (eligibility checked in warmup).
AppleVtDecodeResult appleVtDecodeJpeg(mango::ConstMemory jpeg, int width = 0, int height = 0);
bool appleVtValidateJpeg(mango::ConstMemory jpeg, std::string& error);
bool appleVtWarmup(mango::ConstMemory jpeg, int width, int height, int rounds, mango::u64& out_us);
void appleVtDiscardDecodeCache();

} // namespace gpujpeg

#endif // GPUJPEG_HAS_APPLE_VT

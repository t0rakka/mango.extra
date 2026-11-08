#pragma once

#if defined(GPUJPEG_HAS_APPLE_VT)

#include <string>

#include <mango/core/core.hpp>

struct __CVBuffer;
typedef struct __CVBuffer* CVPixelBufferRef;

namespace gpujpeg
{

struct AppleImageIoTimings
{
    mango::u64 decode_us = 0;  // CGImageSource decode
    mango::u64 render_us = 0;  // CI render into BGRA IOSurface
};

struct AppleImageIoResult
{
    bool ok = false;
    std::string error;
    int width = 0;
    int height = 0;
    mango::u64 decode_us = 0; // decode + render
    AppleImageIoTimings detail;
    CVPixelBufferRef pixel_buffer = nullptr; // retained; release with appleImageIoRelease
};

AppleImageIoResult appleImageIoDecodeJpeg(mango::ConstMemory jpeg);
void appleImageIoRelease(CVPixelBufferRef buffer);
bool appleImageIoWarmup(mango::ConstMemory jpeg, int rounds, mango::u64& out_us);

} // namespace gpujpeg

#endif

#include "apple_vt_backend.hpp"

#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#include <cstring>
#include <vector>

#if defined(GPUJPEG_HAS_APPLE_VT)

#include <VideoToolbox/VideoToolbox.h>
#include <VideoToolbox/VTDecompressionProperties.h>
#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>

#include <mango/image/image.hpp>
#include <mango/image/jpeg.hpp>

namespace gpujpeg
{
namespace
{

using namespace mango;
using namespace mango::image;
using namespace mango::image::jpeg;

struct DecodeCache
{
    int width = 0;
    int height = 0;
    CVPixelBufferRef pixel = nullptr; // IOSurface BGRA pool target (CI resolve only)

    VTDecompressionSessionRef vt_session = nullptr;
    CMVideoFormatDescriptionRef vt_format = nullptr;
};

static DecodeCache g_cache;

struct VtOutputContext
{
    CVPixelBufferRef buffer = nullptr;
};

static VtOutputContext g_vt_out;

void vtDecompressionOutput(void* decompressionOutputRefCon,
    void* /*sourceFrameRefCon*/,
    OSStatus status,
    VTDecodeInfoFlags /*infoFlags*/,
    CVImageBufferRef imageBuffer,
    CMTime /*presentationTimeStamp*/,
    CMTime /*presentationDuration*/)
{
    auto* ctx = static_cast<VtOutputContext*>(decompressionOutputRefCon);
    if (status != noErr || !imageBuffer)
        return;
    if (ctx->buffer)
        CVPixelBufferRelease(ctx->buffer);
    ctx->buffer = CVPixelBufferRetain(imageBuffer);
}

static CFDictionaryRef makeBgraIosurfaceAttrs(int width, int height)
{
    const void* keys[] = {
        kCVPixelBufferPixelFormatTypeKey,
        kCVPixelBufferWidthKey,
        kCVPixelBufferHeightKey,
        kCVPixelBufferIOSurfacePropertiesKey,
        kCVPixelBufferMetalCompatibilityKey,
    };
    const void* vals[] = {
        (__bridge const void*)@(kCVPixelFormatType_32BGRA),
        (__bridge const void*)@(width),
        (__bridge const void*)@(height),
        (__bridge const void*)@{},
        kCFBooleanTrue,
    };
    return CFDictionaryCreate(
        kCFAllocatorDefault, keys, vals, 5,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
}

static bool ensureCachePixelBuffer(int width, int height, std::string& error)
{
    if (g_cache.pixel && g_cache.width == width && g_cache.height == height)
        return true;

    if (g_cache.pixel)
    {
        CVPixelBufferRelease(g_cache.pixel);
        g_cache.pixel = nullptr;
        g_cache.width = 0;
        g_cache.height = 0;
    }

    CFDictionaryRef attrs = makeBgraIosurfaceAttrs(width, height);
    CVPixelBufferRef buffer = nullptr;
    const CVReturn cr = CVPixelBufferCreate(
        kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &buffer);
    CFRelease(attrs);
    if (cr != kCVReturnSuccess || !buffer)
    {
        error = "CVPixelBufferCreate (BGRA IOSurface) failed";
        return false;
    }
    if (!CVPixelBufferGetIOSurface(buffer))
    {
        CVPixelBufferRelease(buffer);
        error = "BGRA pixel buffer is not IOSurface-backed";
        return false;
    }

    g_cache.pixel = buffer;
    g_cache.width = width;
    g_cache.height = height;
    return true;
}

static void resetDecodeCache()
{
    if (g_vt_out.buffer)
    {
        CVPixelBufferRelease(g_vt_out.buffer);
        g_vt_out.buffer = nullptr;
    }
    if (g_cache.vt_session)
    {
        VTDecompressionSessionInvalidate(g_cache.vt_session);
        CFRelease(g_cache.vt_session);
        g_cache.vt_session = nullptr;
    }
    if (g_cache.vt_format)
    {
        CFRelease(g_cache.vt_format);
        g_cache.vt_format = nullptr;
    }
    if (g_cache.pixel)
    {
        CVPixelBufferRelease(g_cache.pixel);
        g_cache.pixel = nullptr;
    }
    g_cache.width = 0;
    g_cache.height = 0;
}

static CVPixelBufferRef borrowBgraIosurfaceBuffer(int width, int height, std::string& error)
{
    if (!ensureCachePixelBuffer(width, height, error))
        return nullptr;
    return CVPixelBufferRetain(g_cache.pixel);
}

static CFDictionaryRef makeRequireHardwareDecoderSpec()
{
    const void* keys[] = { kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder };
    const void* vals[] = { kCFBooleanTrue };
    return CFDictionaryCreate(
        kCFAllocatorDefault, keys, vals, 1,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
}

// Returns false only on unexpected errors. property_known=false when VT does not expose the key (JPEG).
static bool copyVtSessionUsesHardwareDecoder(VTDecompressionSessionRef session,
    bool& using_hw, bool& property_known)
{
    using_hw = false;
    property_known = false;
    if (!session)
        return false;

    CFTypeRef value = nullptr;
    const OSStatus status = VTSessionCopyProperty(
        session,
        kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
        kCFAllocatorDefault,
        &value);
    if (status == kVTPropertyNotSupportedErr)
        return true;

    if (status != noErr)
        return false;

    property_known = true;
    if (value && CFGetTypeID(value) == CFBooleanGetTypeID())
        using_hw = CFBooleanGetValue(static_cast<CFBooleanRef>(value));

    if (value)
        CFRelease(value);
    return true;
}

static bool pixelBufferIsGpuBgraIosurface(CVPixelBufferRef buffer, std::string& error)
{
    if (!buffer)
    {
        error = "null pixel buffer";
        return false;
    }
    if (CVPixelBufferGetPixelFormatType(buffer) != kCVPixelFormatType_32BGRA)
    {
        error = "VT output is not BGRA";
        return false;
    }
    if (!CVPixelBufferGetIOSurface(buffer))
    {
        error = "VT BGRA buffer is not IOSurface-backed";
        return false;
    }
    return true;
}

static CIContext* sharedMetalCiContext()
{
    static CIContext* ctx = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        ctx = [[CIContext alloc] initWithOptions:@{
            kCIContextUseSoftwareRenderer : @NO,
            kCIContextCacheIntermediates : @NO,
        }];
    });
    return ctx;
}

// VT sometimes delivers YCbCr IOSurfaces; resolve on GPU into our BGRA IOSurface (no host map).
static bool gpuResolveToBgraIosurface(CVImageBufferRef src, int width, int height,
    CVPixelBufferRef dst, std::string& error)
{
    CIImage* src_image = [CIImage imageWithCVPixelBuffer:src];
    if (!src_image)
    {
        error = "CIImage from VT buffer failed";
        return false;
    }

    CIContext* ctx = sharedMetalCiContext();
    const CGRect bounds = CGRectMake(0, 0, width, height);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    [ctx render:src_image toCVPixelBuffer:dst bounds:bounds colorSpace:cs];
    CGColorSpaceRelease(cs);
    return pixelBufferIsGpuBgraIosurface(dst, error);
}

static bool ensureVtSession(int width, int height, std::string& error, mango::u64* session_create_us = nullptr)
{
    if (g_cache.vt_session && g_cache.width == width && g_cache.height == height)
        return true;
    if (g_cache.vt_session)
        resetDecodeCache();

    const u64 t_session = Time::us();
    OSStatus status = CMVideoFormatDescriptionCreate(
        kCFAllocatorDefault,
        kCMVideoCodecType_JPEG,
        int32_t(width),
        int32_t(height),
        nullptr,
        &g_cache.vt_format);
    if (status != noErr || !g_cache.vt_format)
    {
        error = "CMVideoFormatDescriptionCreate failed";
        return false;
    }

    CFDictionaryRef dest_attrs = makeBgraIosurfaceAttrs(width, height);
    CFDictionaryRef decoder_spec = makeRequireHardwareDecoderSpec();

    VTDecompressionOutputCallbackRecord cb = {};
    cb.decompressionOutputCallback = vtDecompressionOutput;
    cb.decompressionOutputRefCon = &g_vt_out;

    status = VTDecompressionSessionCreate(
        kCFAllocatorDefault,
        g_cache.vt_format,
        decoder_spec,
        dest_attrs,
        &cb,
        &g_cache.vt_session);
    CFRelease(decoder_spec);
    CFRelease(dest_attrs);

    if (status != noErr || !g_cache.vt_session)
    {
        error = fmt::format(
            "VTDecompressionSessionCreate failed ({}) — hardware JPEG required",
            int(status));
        return false;
    }

    if (!ensureCachePixelBuffer(width, height, error))
    {
        resetDecodeCache();
        return false;
    }
    if (session_create_us)
        *session_create_us = Time::us() - t_session;
    return true;
}

static bool decodeVideoToolbox(ConstMemory jpeg, int width, int height,
    CVPixelBufferRef& out_buffer, bool& hardware_active, bool& hardware_property_read,
    std::string& error, AppleVtDecodeTimings* detail = nullptr)
{
    mango::u64 session_us = 0;
    if (!ensureVtSession(width, height, error, &session_us))
        return false;
    if (detail)
        detail->session_us = session_us;

    CMBlockBufferRef block = nullptr;
    OSStatus status = CMBlockBufferCreateWithMemoryBlock(
        kCFAllocatorDefault,
        const_cast<void*>(static_cast<const void*>(jpeg.address)),
        jpeg.size,
        kCFAllocatorNull,
        nullptr,
        0,
        jpeg.size,
        0,
        &block);
    if (status != noErr || !block)
    {
        error = "CMBlockBufferCreateWithMemoryBlock failed";
        return false;
    }

    CMSampleBufferRef sample = nullptr;
    const size_t sample_size = jpeg.size;
    status = CMSampleBufferCreate(
        kCFAllocatorDefault,
        block,
        true,
        nullptr,
        nullptr,
        g_cache.vt_format,
        1,
        0,
        nullptr,
        1,
        &sample_size,
        &sample);
    if (status != noErr || !sample)
    {
        CFRelease(block);
        error = "CMSampleBufferCreate failed";
        return false;
    }

    if (g_vt_out.buffer)
    {
        CVPixelBufferRelease(g_vt_out.buffer);
        g_vt_out.buffer = nullptr;
    }

    const u64 t_vt = Time::us();
    status = VTDecompressionSessionDecodeFrame(
        g_cache.vt_session, sample, 0, nullptr, nullptr);
    CFRelease(sample);
    CFRelease(block);

    if (status != noErr)
    {
        error = fmt::format("VTDecompressionSessionDecodeFrame failed ({})", int(status));
        return false;
    }
    VTDecompressionSessionWaitForAsynchronousFrames(g_cache.vt_session);
    if (detail)
        detail->vt_wait_us = Time::us() - t_vt;

    hardware_active = false;
    bool hw_property_known = false;
    if (!copyVtSessionUsesHardwareDecoder(g_cache.vt_session, hardware_active, hw_property_known))
    {
        error = "VTSessionCopyProperty(UsingHardwareAcceleratedVideoDecoder) failed";
        return false;
    }
    if (hw_property_known && !hardware_active)
    {
        error = "VT session reports software decoder (UsingHardwareAcceleratedVideoDecoder=false)";
        return false;
    }
    hardware_property_read = hw_property_known;
    if (!hw_property_known)
    {
        // Session was created with RequireHardwareAcceleratedVideoDecoder; property unreadable on JPEG.
        hardware_active = true;
    }

    if (!g_vt_out.buffer)
    {
        error = "VT decode produced no pixel buffer";
        return false;
    }

    if (pixelBufferIsGpuBgraIosurface(g_vt_out.buffer, error))
    {
        out_buffer = g_vt_out.buffer;
        g_vt_out.buffer = nullptr;
        return true;
    }

    CVPixelBufferRef bgra = borrowBgraIosurfaceBuffer(width, height, error);
    if (!bgra)
    {
        CVPixelBufferRelease(g_vt_out.buffer);
        g_vt_out.buffer = nullptr;
        return false;
    }

    const u64 t_resolve = Time::us();
    std::string resolve_err;
    if (!gpuResolveToBgraIosurface(g_vt_out.buffer, width, height, bgra, resolve_err))
    {
        CVPixelBufferRelease(bgra);
        CVPixelBufferRelease(g_vt_out.buffer);
        g_vt_out.buffer = nullptr;
        error = resolve_err;
        return false;
    }

    CVPixelBufferRelease(g_vt_out.buffer);
    g_vt_out.buffer = nullptr;
    if (detail)
        detail->resolve_us = Time::us() - t_resolve;
    out_buffer = bgra;
    return true;
}

static bool validateJpegBaseline(ConstMemory jpeg, int& width, int& height, std::string& error)
{
    JpegDecoder probe(jpeg);
    const JpegInfo& info = probe.info();
    if (!info.valid)
    {
        error = info.error.empty() ? "invalid JPEG" : info.error;
        return false;
    }
    if (info.lossless || info.arithmetic || info.progressive || info.precision != 8 || info.components < 1)
    {
        error = "apple-vt: baseline 8-bit Huffman interleaved only";
        return false;
    }
    width = info.width;
    height = info.height;
    return true;
}

} // namespace

AppleVtAvailability probeAppleVt()
{
    AppleVtAvailability a;
    if (@available(macOS 10.13, *))
    {
        a.hardware_jpeg = VTIsHardwareDecodeSupported(kCMVideoCodecType_JPEG);
        a.ok = true;
        if (!a.hardware_jpeg)
            a.error = "VTIsHardwareDecodeSupported(JPEG) returned false";
    }
    else
    {
        a.error = "VideoToolbox JPEG requires macOS 10.13+";
    }
    return a;
}

void AppleVtReleasePixelBuffer(CVPixelBufferRef buffer)
{
    if (buffer)
        CVPixelBufferRelease(buffer);
}

void appleVtDiscardDecodeCache()
{
    resetDecodeCache();
}

bool appleVtValidateJpeg(ConstMemory jpeg, std::string& error)
{
    int w = 0;
    int h = 0;
    return validateJpegBaseline(jpeg, w, h, error);
}

bool appleVtWarmup(ConstMemory jpeg, int width, int height, int rounds, u64& out_us)
{
    out_us = 0;
    if (rounds <= 0 || width <= 0 || height <= 0)
        return true;

    const u64 t0 = Time::us();
    for (int i = 0; i < rounds; ++i)
    {
        auto r = appleVtDecodeJpeg(jpeg, width, height);
        if (!r.ok)
            return false;
        AppleVtReleasePixelBuffer(r.pixel_buffer);
    }
    out_us = Time::us() - t0;
    return true;
}

AppleVtDecodeResult appleVtDecodeJpeg(ConstMemory jpeg, int width, int height)
{
    AppleVtDecodeResult r;
    const u64 t0 = Time::us();

    if (width > 0 && height > 0)
    {
        r.width = width;
        r.height = height;
    }
    else
    {
        const u64 t_probe = Time::us();
        std::string probe_err;
        if (!validateJpegBaseline(jpeg, r.width, r.height, probe_err))
        {
            r.error = probe_err;
            return r;
        }
        r.detail.probe_us = Time::us() - t_probe;
    }

    r.hardware_capable = VTIsHardwareDecodeSupported(kCMVideoCodecType_JPEG);

    CVPixelBufferRef buffer = nullptr;
    bool hardware_active = false;
    bool hardware_property_read = false;
    std::string vt_err;

    if (decodeVideoToolbox(jpeg, r.width, r.height, buffer, hardware_active,
            hardware_property_read, vt_err, &r.detail))
    {
        r.hardware = hardware_active;
        r.hardware_property_read = hardware_property_read;
        r.detail.total_us = Time::us() - t0;
        r.decode_us = r.detail.total_us;
        r.path = "videotoolbox";
        r.pixel_buffer = buffer;
        r.ok = true;
        return r;
    }

    r.error = vt_err;
    return r;
}

} // namespace gpujpeg

#endif

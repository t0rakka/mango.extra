#include "apple_imageio_backend.hpp"

#if defined(GPUJPEG_HAS_APPLE_VT)

#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreVideo/CoreVideo.h>

namespace gpujpeg
{
namespace
{

using namespace mango;

struct Cache
{
    int width = 0;
    int height = 0;
    CVPixelBufferRef pixel = nullptr;
};

static Cache g_cache;

static CIContext* metalContext()
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

static bool ensurePixel(int width, int height, std::string& error)
{
    if (g_cache.pixel && g_cache.width == width && g_cache.height == height)
        return true;
    if (g_cache.pixel)
    {
        CVPixelBufferRelease(g_cache.pixel);
        g_cache.pixel = nullptr;
    }

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
    CFDictionaryRef attrs = CFDictionaryCreate(
        kCFAllocatorDefault, keys, vals, 5,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CVPixelBufferRef buffer = nullptr;
    const CVReturn cr = CVPixelBufferCreate(
        kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &buffer);
    CFRelease(attrs);
    if (cr != kCVReturnSuccess || !buffer || !CVPixelBufferGetIOSurface(buffer))
    {
        if (buffer)
            CVPixelBufferRelease(buffer);
        error = "ImageIO BGRA IOSurface create failed";
        return false;
    }
    g_cache.pixel = buffer;
    g_cache.width = width;
    g_cache.height = height;
    return true;
}

} // namespace

void appleImageIoRelease(CVPixelBufferRef buffer)
{
    if (buffer)
        CVPixelBufferRelease(buffer);
}

bool appleImageIoWarmup(ConstMemory jpeg, int rounds, u64& out_us)
{
    out_us = 0;
    if (rounds <= 0)
        return true;
    const u64 t0 = Time::us();
    for (int i = 0; i < rounds; ++i)
    {
        AppleImageIoResult r = appleImageIoDecodeJpeg(jpeg);
        if (!r.ok)
            return false;
        appleImageIoRelease(r.pixel_buffer);
    }
    out_us = Time::us() - t0;
    return true;
}

AppleImageIoResult appleImageIoDecodeJpeg(ConstMemory jpeg)
{
    AppleImageIoResult r;
    if (!jpeg.address || jpeg.size == 0)
    {
        r.error = "empty JPEG";
        return r;
    }

    const u64 t_all = Time::us();
    CFDataRef data = CFDataCreateWithBytesNoCopy(
        kCFAllocatorDefault, jpeg.address, CFIndex(jpeg.size), kCFAllocatorNull);
    CGImageSourceRef source = CGImageSourceCreateWithData(data, nullptr);
    CFRelease(data);
    if (!source)
    {
        r.error = "CGImageSourceCreateWithData failed";
        return r;
    }

    const void* opt_keys[] = { kCGImageSourceShouldCacheImmediately };
    const void* opt_vals[] = { kCFBooleanTrue };
    CFDictionaryRef opts = CFDictionaryCreate(
        kCFAllocatorDefault, opt_keys, opt_vals, 1,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    const u64 t_dec = Time::us();
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, opts);
    r.detail.decode_us = Time::us() - t_dec;
    CFRelease(opts);
    CFRelease(source);
    if (!image)
    {
        r.error = "CGImageSourceCreateImageAtIndex failed";
        return r;
    }

    r.width = int(CGImageGetWidth(image));
    r.height = int(CGImageGetHeight(image));
    std::string error;
    if (!ensurePixel(r.width, r.height, error))
    {
        CGImageRelease(image);
        r.error = error;
        return r;
    }

    CIImage* ci = [CIImage imageWithCGImage:image];
    CGImageRelease(image);
    if (!ci)
    {
        r.error = "CIImage from CGImage failed";
        return r;
    }

    const u64 t_render = Time::us();
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    [metalContext() render:ci toCVPixelBuffer:g_cache.pixel
        bounds:CGRectMake(0, 0, r.width, r.height) colorSpace:cs];
    CGColorSpaceRelease(cs);
    r.detail.render_us = Time::us() - t_render;

    // Includes CGImage→CIImage, which sits between the two detail timers.
    r.decode_us = Time::us() - t_all;
    r.pixel_buffer = CVPixelBufferRetain(g_cache.pixel);
    r.ok = true;
    return r;
}

} // namespace gpujpeg

#endif

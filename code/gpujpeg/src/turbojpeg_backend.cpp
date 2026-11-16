// Copyright 2026 Distance Technologies Oy. For internal use only.
//

#include "turbojpeg_backend.hpp"

#include <string>
#include <vector>

#if defined(GPUJPEG_HAS_TURBOJPEG)
#include <jconfig.h>
#include <turbojpeg.h>
#endif

namespace gpujpeg
{

#if !defined(GPUJPEG_HAS_TURBOJPEG)

TurboJpegRuntimeInfo queryTurboJpegRuntime()
{
    TurboJpegRuntimeInfo info;
    info.error = "not built (vcpkg install libjpeg-turbo)";
    return info;
}

bool turboJpegAvailable()
{
    return false;
}

bool TurboJpegDecoder::init(int, int, std::string& error)
{
    error = "libjpeg-turbo not built";
    return false;
}

bool TurboJpegDecoder::warmup(mango::ConstMemory, int, std::string& error)
{
    error = "libjpeg-turbo not built";
    return false;
}

bool TurboJpegDecoder::decodeToHost(mango::ConstMemory, void*, size_t,
    mango::u64&, std::string& error)
{
    error = "libjpeg-turbo not built";
    return false;
}

void TurboJpegDecoder::shutdown()
{
    m_handle = nullptr;
    ok = false;
}

#else

namespace
{

std::string turboError(tjhandle handle, const char* what)
{
    const char* msg = tjGetErrorStr2(handle);
    if (!msg || !msg[0])
        msg = "unknown";
    return std::string(what) + ": " + msg;
}

} // namespace

TurboJpegRuntimeInfo queryTurboJpegRuntime()
{
    TurboJpegRuntimeInfo info;
    info.built = true;
    return info;
}

bool turboJpegAvailable()
{
    return true;
}

bool TurboJpegDecoder::init(int w, int h, std::string& error)
{
    shutdown();
    if (w <= 0 || h <= 0)
    {
        error = "libjpeg-turbo: invalid dimensions";
        return false;
    }

    const mango::u64 t0 = nowUs();
    tjhandle handle = tjInitDecompress();
    if (!handle)
    {
        error = turboError(nullptr, "tjInitDecompress");
        return false;
    }

    m_handle = handle;
    width = w;
    height = h;
    codec_name = "libjpeg-turbo " + std::to_string(LIBJPEG_TURBO_VERSION_NUMBER / 1000000)
        + "." + std::to_string((LIBJPEG_TURBO_VERSION_NUMBER / 1000) % 1000)
        + "." + std::to_string(LIBJPEG_TURBO_VERSION_NUMBER % 1000);
    init_us = nowUs() - t0;
    ok = true;
    return true;
}

bool TurboJpegDecoder::warmup(mango::ConstMemory jpeg, int rounds, std::string& error)
{
    if (!ok || !m_handle)
    {
        error = "libjpeg-turbo decoder not initialized";
        return false;
    }
    if (rounds <= 0)
    {
        warmup_us = 0;
        return true;
    }

    std::vector<mango::u8> scratch(size_t(width) * size_t(height) * 4);
    const mango::u64 t0 = nowUs();
    for (int i = 0; i < rounds; ++i)
    {
        mango::u64 ignore = 0;
        if (!decodeToHost(jpeg, scratch.data(), scratch.size(), ignore, error))
            return false;
    }
    warmup_us = nowUs() - t0;
    return true;
}

bool TurboJpegDecoder::decodeToHost(mango::ConstMemory jpeg, void* dest, size_t dest_bytes,
    mango::u64& decode_us, std::string& error)
{
    decode_us = 0;
    if (!ok || !m_handle)
    {
        error = "libjpeg-turbo decoder not initialized";
        return false;
    }
    if (!jpeg.address || !jpeg.size || !dest)
    {
        error = "libjpeg-turbo: empty bitstream or null dest";
        return false;
    }
    if (jpeg.size > 0xffffffffu)
    {
        error = "libjpeg-turbo: JPEG larger than 4GB";
        return false;
    }

    auto* handle = static_cast<tjhandle>(m_handle);
    const mango::u64 t0 = nowUs();

    int w = 0;
    int h = 0;
    int subsamp = 0;
    int colorspace = 0;
    if (tjDecompressHeader3(handle,
            reinterpret_cast<const unsigned char*>(jpeg.address),
            static_cast<unsigned long>(jpeg.size),
            &w, &h, &subsamp, &colorspace) != 0)
    {
        error = turboError(handle, "tjDecompressHeader3");
        return false;
    }
    if (w != width || h != height)
    {
        error = "libjpeg-turbo: decoded size does not match init dimensions";
        return false;
    }

    const std::uint64_t nbytes = std::uint64_t(w) * std::uint64_t(h) * 4;
    if (nbytes > dest_bytes)
    {
        error = "libjpeg-turbo: output buffer too small";
        return false;
    }

    // flags 0: library default IDCT (accurate), not TJFLAG_FASTDCT.
    if (tjDecompress2(handle,
            reinterpret_cast<const unsigned char*>(jpeg.address),
            static_cast<unsigned long>(jpeg.size),
            static_cast<unsigned char*>(dest),
            w, 0, h, TJPF_RGBA, 0) != 0)
    {
        error = turboError(handle, "tjDecompress2");
        return false;
    }

    decode_us = nowUs() - t0;
    return true;
}

void TurboJpegDecoder::shutdown()
{
    if (m_handle)
    {
        tjDestroy(static_cast<tjhandle>(m_handle));
        m_handle = nullptr;
    }
    ok = false;
    width = 0;
    height = 0;
    codec_name.clear();
}

#endif // GPUJPEG_HAS_TURBOJPEG

} // namespace gpujpeg

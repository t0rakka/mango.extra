// Copyright 2026 Distance Technologies Oy. For internal use only.
//

#include "wic_backend.hpp"

#include <cstdio>
#include <string>
#include <vector>

#if defined(GPUJPEG_HAS_WIC)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <initguid.h>
#include <wincodec.h>
#include <wrl/client.h>
#endif

namespace gpujpeg
{

#if !defined(GPUJPEG_HAS_WIC)

WicRuntimeInfo queryWicRuntime()
{
    WicRuntimeInfo info;
    info.error = "not built (Windows only)";
    return info;
}

bool wicAvailable()
{
    return false;
}

bool WicDecoder::init(int, int, std::string& error)
{
    error = "WIC not built (Windows only)";
    return false;
}

bool WicDecoder::warmup(mango::ConstMemory, int, std::string& error)
{
    error = "WIC not built";
    return false;
}

bool WicDecoder::decodeToHost(mango::ConstMemory, void*, size_t,
    mango::u64&, std::string& error)
{
    error = "WIC not built";
    return false;
}

void WicDecoder::shutdown()
{
    m_impl = nullptr;
    ok = false;
}

#else

namespace
{

using Microsoft::WRL::ComPtr;

struct Impl
{
    ComPtr<IWICImagingFactory> factory;
};

std::string hresultText(const char* what, HRESULT hr)
{
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s (0x%08lX)", what, static_cast<unsigned long>(hr));
    return buf;
}

bool failHr(HRESULT hr, const char* what, std::string& error)
{
    if (SUCCEEDED(hr))
        return false;
    error = hresultText(what, hr);
    return true;
}

// One COM init per thread that touches WIC. The factory itself is free-threaded.
bool ensureCom(std::string& error)
{
    thread_local bool ready = false;
    if (ready)
        return true;
    const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE)
    {
        error = hresultText("CoInitializeEx", hr);
        return false;
    }
    ready = true;
    return true;
}

std::string wideToUtf8(const wchar_t* ws)
{
    if (!ws || !ws[0])
        return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, ws, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1)
        return {};
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string inboxCodecName()
{
    ComPtr<IWICBitmapDecoder> decoder;
    const HRESULT hr = CoCreateInstance(CLSID_WICJpegDecoder, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&decoder));
    if (FAILED(hr) || !decoder)
        return "JPEG Decoder";

    ComPtr<IWICBitmapDecoderInfo> info;
    if (FAILED(decoder->GetDecoderInfo(&info)) || !info)
        return "JPEG Decoder";

    wchar_t name[256] = {};
    UINT got = 0;
    if (FAILED(info->GetFriendlyName(static_cast<UINT>(std::size(name)), name, &got)) || !name[0])
        return "JPEG Decoder";
    const std::string utf8 = wideToUtf8(name);
    return utf8.empty() ? "JPEG Decoder" : utf8;
}

} // namespace

WicRuntimeInfo queryWicRuntime()
{
    WicRuntimeInfo info;
    info.built = true;
    return info;
}

bool wicAvailable()
{
    return true;
}

bool WicDecoder::init(int w, int h, std::string& error)
{
    shutdown();
    if (w <= 0 || h <= 0)
    {
        error = "WIC: invalid dimensions";
        return false;
    }

    const mango::u64 t0 = nowUs();
    if (!ensureCom(error))
        return false;

    auto* impl = new Impl;
    const HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&impl->factory));
    if (FAILED(hr) || !impl->factory)
    {
        delete impl;
        error = hresultText("CoCreateInstance(IWICImagingFactory)", hr);
        return false;
    }

    m_impl = impl;
    width = w;
    height = h;
    codec_name = inboxCodecName();
    init_us = nowUs() - t0;
    ok = true;
    return true;
}

bool WicDecoder::warmup(mango::ConstMemory jpeg, int rounds, std::string& error)
{
    if (!ok || !m_impl)
    {
        error = "WIC decoder not initialized";
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

bool WicDecoder::decodeToHost(mango::ConstMemory jpeg, void* dest, size_t dest_bytes,
    mango::u64& decode_us, std::string& error)
{
    decode_us = 0;
    if (!ok || !m_impl)
    {
        error = "WIC decoder not initialized";
        return false;
    }
    if (!jpeg.address || !jpeg.size || !dest)
    {
        error = "WIC: empty bitstream or null dest";
        return false;
    }
    if (jpeg.size > 0xffffffffu)
    {
        error = "WIC: JPEG larger than 4GB";
        return false;
    }

    if (!ensureCom(error))
        return false;

    auto* impl = static_cast<Impl*>(m_impl);
    const mango::u64 t0 = nowUs();

    // Inbox JPEG codec, not whichever component registered for .jpg.
    ComPtr<IWICStream> stream;
    HRESULT hr = impl->factory->CreateStream(&stream);
    if (failHr(hr, "IWICImagingFactory::CreateStream", error))
        return false;

    hr = stream->InitializeFromMemory(
        const_cast<BYTE*>(reinterpret_cast<const BYTE*>(jpeg.address)),
        static_cast<DWORD>(jpeg.size));
    if (failHr(hr, "IWICStream::InitializeFromMemory", error))
        return false;

    ComPtr<IWICBitmapDecoder> decoder;
    hr = CoCreateInstance(CLSID_WICJpegDecoder, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&decoder));
    if (failHr(hr, "CoCreateInstance(CLSID_WICJpegDecoder)", error))
        return false;

    hr = decoder->Initialize(stream.Get(), WICDecodeMetadataCacheOnDemand);
    if (failHr(hr, "IWICBitmapDecoder::Initialize", error))
        return false;

    ComPtr<IWICBitmapFrameDecode> frame;
    hr = decoder->GetFrame(0, &frame);
    if (failHr(hr, "IWICBitmapDecoder::GetFrame", error))
        return false;

    UINT w = 0;
    UINT h = 0;
    hr = frame->GetSize(&w, &h);
    if (failHr(hr, "IWICBitmapFrameDecode::GetSize", error))
        return false;
    if (int(w) != width || int(h) != height)
    {
        error = "WIC: decoded size does not match init dimensions";
        return false;
    }

    ComPtr<IWICFormatConverter> converter;
    hr = impl->factory->CreateFormatConverter(&converter);
    if (failHr(hr, "IWICImagingFactory::CreateFormatConverter", error))
        return false;

    // R,G,B,A matches VK_FORMAT_R8G8B8A8_UNORM. JPEG has no alpha; WIC writes 255.
    hr = converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
    if (failHr(hr, "IWICFormatConverter::Initialize", error))
        return false;

    const std::uint64_t nbytes = std::uint64_t(w) * std::uint64_t(h) * 4;
    if (nbytes > dest_bytes || nbytes > 0xffffffffu)
    {
        error = "WIC: output buffer too small";
        return false;
    }

    const UINT stride = w * 4;
    hr = converter->CopyPixels(nullptr, stride, static_cast<UINT>(nbytes),
        static_cast<BYTE*>(dest));
    if (failHr(hr, "IWICFormatConverter::CopyPixels", error))
        return false;

    decode_us = nowUs() - t0;
    return true;
}

void WicDecoder::shutdown()
{
    if (m_impl)
    {
        delete static_cast<Impl*>(m_impl);
        m_impl = nullptr;
    }
    ok = false;
    width = 0;
    height = 0;
    codec_name.clear();
}

#endif // GPUJPEG_HAS_WIC

} // namespace gpujpeg

#include "intel_vpl_backend.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#if defined(GPUJPEG_HAS_INTEL_VPL)
#include <vpl/mfx.h>
#include <vpl/mfxjpeg.h>
#include <vpl/mfxdispatcher.h>
#endif

namespace gpujpeg
{

#if !defined(GPUJPEG_HAS_INTEL_VPL)

IntelVplRuntimeInfo queryIntelVplRuntime()
{
    IntelVplRuntimeInfo info;
    info.error = "not built (install libvpl and reconfigure CMake)";
    return info;
}

bool intelVplAvailable()
{
    return false;
}

bool IntelVplDecoder::init(int, int, std::string& error)
{
    error = "Intel VPL not built (install libvpl and reconfigure CMake)";
    return false;
}

bool IntelVplDecoder::warmup(mango::ConstMemory, int, std::string& error)
{
    error = "Intel VPL not built";
    return false;
}

bool IntelVplDecoder::decodeToHost(mango::ConstMemory, void*, size_t,
    mango::u64&, std::string& error)
{
    error = "Intel VPL not built";
    return false;
}

void IntelVplDecoder::shutdown()
{
    m_impl = nullptr;
    ok = false;
}

#else

namespace
{

const char* mfxStatusString(mfxStatus s)
{
    switch (s)
    {
        case MFX_ERR_NONE:                    return "MFX_ERR_NONE";
        case MFX_ERR_UNKNOWN:                 return "MFX_ERR_UNKNOWN";
        case MFX_ERR_NULL_PTR:                return "MFX_ERR_NULL_PTR";
        case MFX_ERR_UNSUPPORTED:             return "MFX_ERR_UNSUPPORTED";
        case MFX_ERR_MEMORY_ALLOC:            return "MFX_ERR_MEMORY_ALLOC";
        case MFX_ERR_NOT_ENOUGH_BUFFER:       return "MFX_ERR_NOT_ENOUGH_BUFFER";
        case MFX_ERR_INVALID_HANDLE:          return "MFX_ERR_INVALID_HANDLE";
        case MFX_ERR_LOCK_MEMORY:             return "MFX_ERR_LOCK_MEMORY";
        case MFX_ERR_NOT_INITIALIZED:         return "MFX_ERR_NOT_INITIALIZED";
        case MFX_ERR_NOT_FOUND:               return "MFX_ERR_NOT_FOUND";
        case MFX_ERR_MORE_DATA:               return "MFX_ERR_MORE_DATA";
        case MFX_ERR_MORE_SURFACE:            return "MFX_ERR_MORE_SURFACE";
        case MFX_ERR_ABORTED:                 return "MFX_ERR_ABORTED";
        case MFX_ERR_DEVICE_LOST:             return "MFX_ERR_DEVICE_LOST";
        case MFX_ERR_INCOMPATIBLE_VIDEO_PARAM:return "MFX_ERR_INCOMPATIBLE_VIDEO_PARAM";
        case MFX_ERR_INVALID_VIDEO_PARAM:     return "MFX_ERR_INVALID_VIDEO_PARAM";
        case MFX_ERR_UNDEFINED_BEHAVIOR:      return "MFX_ERR_UNDEFINED_BEHAVIOR";
        case MFX_ERR_DEVICE_FAILED:           return "MFX_ERR_DEVICE_FAILED";
        case MFX_ERR_MORE_BITSTREAM:          return "MFX_ERR_MORE_BITSTREAM";
        case MFX_ERR_GPU_HANG:                return "MFX_ERR_GPU_HANG";
        case MFX_ERR_REALLOC_SURFACE:         return "MFX_ERR_REALLOC_SURFACE";
        case MFX_WRN_IN_EXECUTION:            return "MFX_WRN_IN_EXECUTION";
        case MFX_WRN_DEVICE_BUSY:             return "MFX_WRN_DEVICE_BUSY";
        case MFX_WRN_VIDEO_PARAM_CHANGED:     return "MFX_WRN_VIDEO_PARAM_CHANGED";
        case MFX_WRN_PARTIAL_ACCELERATION:    return "MFX_WRN_PARTIAL_ACCELERATION";
        default:                              return "mfxStatus(?)";
    }
}

mfxStatus createJpegHwSession(mfxLoader loader, mfxSession* session)
{
    mfxConfig cfg_impl = MFXCreateConfig(loader);
    if (!cfg_impl)
        return MFX_ERR_NULL_PTR;

    mfxVariant val {};
    val.Type = MFX_VARIANT_TYPE_U32;
    val.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    mfxStatus sts = MFXSetConfigFilterProperty(cfg_impl,
        reinterpret_cast<const mfxU8*>("mfxImplDescription.Impl"), val);
    if (sts != MFX_ERR_NONE)
        return sts;

    mfxConfig cfg_codec = MFXCreateConfig(loader);
    if (!cfg_codec)
        return MFX_ERR_NULL_PTR;

    val.Data.U32 = MFX_CODEC_JPEG;
    sts = MFXSetConfigFilterProperty(cfg_codec,
        reinterpret_cast<const mfxU8*>(
            "mfxImplDescription.mfxDecoderDescription.decoder.CodecID"),
        val);
    if (sts != MFX_ERR_NONE)
        return sts;

    return MFXCreateSession(loader, 0, session);
}

// BT.601 limited-range-ish fast path (good enough for a latency bench).
void nv12ToRgba(const mfxU8* y, int y_pitch,
    const mfxU8* uv, int uv_pitch,
    int width, int height, mfxU8* rgba)
{
    for (int row = 0; row < height; ++row)
    {
        const mfxU8* yrow = y + row * y_pitch;
        const mfxU8* uvrow = uv + (row / 2) * uv_pitch;
        mfxU8* out = rgba + size_t(row) * size_t(width) * 4;

        for (int col = 0; col < width; ++col)
        {
            const int Y = int(yrow[col]);
            const int U = int(uvrow[(col & ~1) + 0]) - 128;
            const int V = int(uvrow[(col & ~1) + 1]) - 128;

            int r = Y + ((359 * V) >> 8);
            int g = Y - ((88 * U + 183 * V) >> 8);
            int b = Y + ((454 * U) >> 8);

            out[0] = mfxU8(std::clamp(r, 0, 255));
            out[1] = mfxU8(std::clamp(g, 0, 255));
            out[2] = mfxU8(std::clamp(b, 0, 255));
            out[3] = 255;
            out += 4;
        }
    }
}

void i420ToRgba(const mfxU8* y, int y_pitch,
    const mfxU8* u, int u_pitch,
    const mfxU8* v, int v_pitch,
    int width, int height, mfxU8* rgba)
{
    for (int row = 0; row < height; ++row)
    {
        const mfxU8* yrow = y + row * y_pitch;
        const mfxU8* urow = u + (row / 2) * u_pitch;
        const mfxU8* vrow = v + (row / 2) * v_pitch;
        mfxU8* out = rgba + size_t(row) * size_t(width) * 4;

        for (int col = 0; col < width; ++col)
        {
            const int Y = int(yrow[col]);
            const int U = int(urow[col / 2]) - 128;
            const int V = int(vrow[col / 2]) - 128;

            int r = Y + ((359 * V) >> 8);
            int g = Y - ((88 * U + 183 * V) >> 8);
            int b = Y + ((454 * U) >> 8);

            out[0] = mfxU8(std::clamp(r, 0, 255));
            out[1] = mfxU8(std::clamp(g, 0, 255));
            out[2] = mfxU8(std::clamp(b, 0, 255));
            out[3] = 255;
            out += 4;
        }
    }
}

bool surfaceToRgba(mfxFrameSurface1* surface, int width, int height,
    mfxU8* dest, size_t dest_bytes, std::string& error)
{
    if (!surface || !surface->FrameInterface || !dest)
    {
        error = "Intel VPL: null surface/dest";
        return false;
    }
    const size_t need = size_t(width) * size_t(height) * 4;
    if (dest_bytes < need)
    {
        error = "Intel VPL: destination buffer too small";
        return false;
    }

    mfxStatus sts = surface->FrameInterface->Map(surface, MFX_MAP_READ);
    if (sts != MFX_ERR_NONE)
    {
        error = std::string("Intel VPL Map failed: ") + mfxStatusString(sts);
        return false;
    }

    const mfxFrameInfo& info = surface->Info;
    const mfxFrameData& data = surface->Data;
    bool ok = false;

    switch (info.FourCC)
    {
        case MFX_FOURCC_NV12:
            if (data.Y && data.UV)
            {
                nv12ToRgba(data.Y, data.Pitch, data.UV, data.Pitch,
                    width, height, dest);
                ok = true;
            }
            else
                error = "Intel VPL: NV12 plane pointers null";
            break;

        case MFX_FOURCC_I420: // same FourCC value as IYUV in VPL headers
            if (data.Y && data.U && data.V)
            {
                i420ToRgba(data.Y, data.Pitch,
                    data.U, data.Pitch / 2,
                    data.V, data.Pitch / 2,
                    width, height, dest);
                ok = true;
            }
            else
                error = "Intel VPL: I420 plane pointers null";
            break;

        case MFX_FOURCC_RGB4:
            if (data.B)
            {
                // BGRA packed in Pitch rows → tightly packed RGBA
                for (int row = 0; row < height; ++row)
                {
                    const mfxU8* src = data.B + row * data.Pitch;
                    mfxU8* out = dest + size_t(row) * size_t(width) * 4;
                    for (int col = 0; col < width; ++col)
                    {
                        out[0] = src[2];
                        out[1] = src[1];
                        out[2] = src[0];
                        out[3] = 255;
                        src += 4;
                        out += 4;
                    }
                }
                ok = true;
            }
            else
                error = "Intel VPL: RGB4 pointer null";
            break;

        default:
            error = "Intel VPL: unsupported FourCC for host convert";
            break;
    }

    surface->FrameInterface->Unmap(surface);
    return ok;
}

struct Impl
{
    mfxLoader loader = nullptr;
    mfxSession session = nullptr;
    mfxVideoParam params {};
    int width = 0;
    int height = 0;
    bool decode_inited = false;

    void closeDecode()
    {
        if (session && decode_inited)
        {
            MFXVideoDECODE_Close(session);
            decode_inited = false;
        }
    }

    void destroy()
    {
        closeDecode();
        if (session)
        {
            MFXClose(session);
            session = nullptr;
        }
        if (loader)
        {
            MFXUnload(loader);
            loader = nullptr;
        }
    }

};

} // namespace

IntelVplRuntimeInfo queryIntelVplRuntime()
{
    IntelVplRuntimeInfo info;

    mfxLoader loader = MFXLoad();
    if (!loader)
    {
        info.error = "MFXLoad failed (libvpl dispatcher missing from PATH?)";
        return info;
    }
    info.dispatcher = true;

    mfxSession session = nullptr;
    mfxStatus sts = createJpegHwSession(loader, &session);
    if (sts != MFX_ERR_NONE || !session)
    {
        info.error = std::string("no hardware JPEG impl (") + mfxStatusString(sts)
            + ") — enable Intel iGPU + install Intel graphics driver";
        MFXUnload(loader);
        return info;
    }

    mfxImplDescription* desc = nullptr;
    if (MFXEnumImplementations(loader, 0, MFX_IMPLCAPS_IMPLDESCSTRUCTURE,
            reinterpret_cast<mfxHDL*>(&desc)) == MFX_ERR_NONE && desc)
    {
        if (desc->ImplName[0])
            info.impl_name = reinterpret_cast<const char*>(desc->ImplName);
        MFXDispReleaseImplDescription(loader, desc);
    }

    info.hardware_jpeg = true;
    MFXClose(session);
    MFXUnload(loader);
    return info;
}

bool intelVplAvailable()
{
    return queryIntelVplRuntime().hardware_jpeg;
}

bool IntelVplDecoder::init(int w, int h, std::string& error)
{
    shutdown();
    if (w <= 0 || h <= 0)
    {
        error = "Intel VPL: invalid dimensions";
        return false;
    }

    const mango::u64 t0 = nowUs();
    auto* impl = new Impl();

    impl->loader = MFXLoad();
    if (!impl->loader)
    {
        delete impl;
        error = "MFXLoad failed";
        return false;
    }

    mfxStatus sts = createJpegHwSession(impl->loader, &impl->session);
    if (sts != MFX_ERR_NONE || !impl->session)
    {
        error = std::string("Intel VPL session create failed: ") + mfxStatusString(sts)
            + " (need Intel iGPU + graphics driver for QSV JPEG)";
        impl->destroy();
        delete impl;
        return false;
    }

    impl->width = w;
    impl->height = h;
    m_impl = impl;
    width = w;
    height = h;
    init_us = nowUs() - t0;
    ok = true;
    return true;
}

bool IntelVplDecoder::warmup(mango::ConstMemory jpeg, int rounds, std::string& error)
{
    if (!ok || !m_impl)
    {
        error = "Intel VPL decoder not initialized";
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

bool IntelVplDecoder::decodeToHost(mango::ConstMemory jpeg, void* dest, size_t dest_bytes,
    mango::u64& decode_us, std::string& error)
{
    decode_us = 0;
    if (!ok || !m_impl)
    {
        error = "Intel VPL decoder not initialized";
        return false;
    }
    if (!jpeg.address || !jpeg.size || !dest)
    {
        error = "Intel VPL: empty bitstream or null dest";
        return false;
    }

    auto* impl = static_cast<Impl*>(m_impl);
    const mango::u64 t0 = nowUs();

    mfxBitstream bitstream {};
    bitstream.Data = const_cast<mfxU8*>(reinterpret_cast<const mfxU8*>(jpeg.address));
    bitstream.DataLength = mfxU32(jpeg.size);
    bitstream.DataOffset = 0;
    bitstream.MaxLength = mfxU32(jpeg.size);
    bitstream.CodecId = MFX_CODEC_JPEG;

    // Known-good still-JPEG path on this driver: Close+DecodeHeader+Init per call,
    // sync via FrameInterface only. Warm Reset/SyncOperation both returned
    // MFX_ERR_INVALID_VIDEO_PARAM here — revisit once we have a working baseline again.
    impl->closeDecode();

    mfxVideoParam params {};
    params.mfx.CodecId = MFX_CODEC_JPEG;
    params.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

    mfxStatus sts = MFXVideoDECODE_DecodeHeader(impl->session, &bitstream, &params);
    if (sts != MFX_ERR_NONE)
    {
        error = std::string("Intel VPL DecodeHeader failed: ") + mfxStatusString(sts);
        return false;
    }

    params.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

    sts = MFXVideoDECODE_Init(impl->session, &params);
    if (sts < MFX_ERR_NONE)
    {
        error = std::string("Intel VPL DECODE_Init failed: ") + mfxStatusString(sts);
        return false;
    }
    impl->decode_inited = true;
    impl->params = params;

    const int out_w = params.mfx.FrameInfo.CropW
        ? int(params.mfx.FrameInfo.CropW)
        : int(params.mfx.FrameInfo.Width);
    const int out_h = params.mfx.FrameInfo.CropH
        ? int(params.mfx.FrameInfo.CropH)
        : int(params.mfx.FrameInfo.Height);

    if (out_w != width || out_h != height)
    {
        error = "Intel VPL: decoded size does not match init dimensions";
        return false;
    }

    bitstream.DataOffset = 0;
    bitstream.DataLength = mfxU32(jpeg.size);

    mfxFrameSurface1* out_surface = nullptr;
    mfxSyncPoint syncp = nullptr;
    bool got_frame = false;

    bool draining = false;
    for (int attempt = 0; attempt < 64 && !got_frame; ++attempt)
    {
        sts = MFXVideoDECODE_DecodeFrameAsync(impl->session,
            draining ? nullptr : &bitstream,
            nullptr,
            &out_surface,
            &syncp);

        if (sts == MFX_ERR_NONE && out_surface)
        {
            if (!out_surface->FrameInterface)
            {
                error = "Intel VPL: surface has no FrameInterface";
                return false;
            }

            do
            {
                sts = out_surface->FrameInterface->Synchronize(out_surface, 1000);
            } while (sts == MFX_WRN_IN_EXECUTION);

            if (sts != MFX_ERR_NONE)
            {
                out_surface->FrameInterface->Release(out_surface);
                error = std::string("Intel VPL Synchronize failed: ") + mfxStatusString(sts);
                return false;
            }

            if (!surfaceToRgba(out_surface, width, height,
                    static_cast<mfxU8*>(dest), dest_bytes, error))
            {
                out_surface->FrameInterface->Release(out_surface);
                return false;
            }
            out_surface->FrameInterface->Release(out_surface);
            got_frame = true;
            break;
        }

        if (sts == MFX_ERR_MORE_DATA)
        {
            if (draining)
                break;
            draining = true;
            continue;
        }

        if (sts == MFX_WRN_DEVICE_BUSY)
            continue;

        if (sts == MFX_ERR_MORE_SURFACE)
            continue;

        error = std::string("Intel VPL DecodeFrameAsync failed: ") + mfxStatusString(sts);
        return false;
    }

    if (!got_frame)
    {
        error = "Intel VPL: no frame produced";
        return false;
    }

    decode_us = nowUs() - t0;
    return true;
}

void IntelVplDecoder::shutdown()
{
    if (m_impl)
    {
        auto* impl = static_cast<Impl*>(m_impl);
        impl->destroy();
        delete impl;
        m_impl = nullptr;
    }
    ok = false;
    width = 0;
    height = 0;
}

#endif // GPUJPEG_HAS_INTEL_VPL

} // namespace gpujpeg

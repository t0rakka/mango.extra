#include "nvjpeg_backend.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#if defined(GPUJPEG_HAS_NVJPEG)
#include <cuda_runtime.h>
#include <nvjpeg.h>
#endif

namespace gpujpeg
{

#if !defined(GPUJPEG_HAS_NVJPEG)

bool nvjpegAvailable()
{
    return false;
}

bool NvJpegDecoder::init(NvJpegMode, int, int, int, std::string& error)
{
    error = "nvJPEG not built (install CUDA Toolkit and reconfigure CMake)";
    return false;
}

bool NvJpegDecoder::warmup(mango::ConstMemory, int, std::string& error)
{
    error = "nvJPEG not built";
    return false;
}

bool NvJpegDecoder::decodeToDevice(mango::ConstMemory, mango::u64&, std::string& error)
{
    error = "nvJPEG not built";
    return false;
}

bool NvJpegDecoder::decodeLaunch(mango::ConstMemory, int, std::string& error)
{
    error = "nvJPEG not built";
    return false;
}

bool NvJpegDecoder::syncSlot(int, std::string& error)
{
    error = "nvJPEG not built";
    return false;
}

std::uint8_t* NvJpegDecoder::deviceRgbi(int) const
{
    return nullptr;
}

int NvJpegDecoder::rgbiPitch() const
{
    return 0;
}

void* NvJpegDecoder::decodeStream(int) const
{
    return nullptr;
}

bool NvJpegDecoder::decodePipeline(mango::ConstMemory, int,
    mango::u64&, mango::u64&, std::string& error)
{
    error = "nvJPEG not built";
    return false;
}

void NvJpegDecoder::shutdown()
{
    m_impl = nullptr;
    ok = false;
}

#else

namespace
{

struct Slot
{
    nvjpegJpegState_t state = nullptr;
    cudaStream_t stream = nullptr;
    nvjpegImage_t image {};
    bool submitted = false;
};

struct Impl
{
    nvjpegHandle_t handle = nullptr;
    std::vector<Slot> slots;
    size_t rgb_bytes = 0;
};

const char* nvjpegStatusString(nvjpegStatus_t s)
{
    switch (s)
    {
        case NVJPEG_STATUS_SUCCESS: return "SUCCESS";
        case NVJPEG_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case NVJPEG_STATUS_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case NVJPEG_STATUS_BAD_JPEG: return "BAD_JPEG";
        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "JPEG_NOT_SUPPORTED";
        case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "ALLOCATOR_FAILURE";
        case NVJPEG_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case NVJPEG_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case NVJPEG_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        default: return "UNKNOWN";
    }
}

bool checkNv(nvjpegStatus_t s, const char* what, std::string& error)
{
    if (s == NVJPEG_STATUS_SUCCESS)
        return true;
    error = fmt::format("{} failed: {}", what, nvjpegStatusString(s));
    return false;
}

bool checkCuda(cudaError_t e, const char* what, std::string& error)
{
    if (e == cudaSuccess)
        return true;
    error = fmt::format("{} failed: {}", what, cudaGetErrorString(e));
    return false;
}

nvjpegBackend_t toNvBackend(NvJpegMode mode)
{
    switch (mode)
    {
        case NvJpegMode::Hybrid: return NVJPEG_BACKEND_HYBRID;
        case NvJpegMode::Gpu:    return NVJPEG_BACKEND_GPU_HYBRID;
    }
    return NVJPEG_BACKEND_HYBRID;
}

Impl* asImpl(void* p)
{
    return static_cast<Impl*>(p);
}

} // namespace

bool nvjpegAvailable()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return false;
    return true;
}

void NvJpegDecoder::shutdown()
{
    Impl* impl = asImpl(m_impl);
    if (!impl)
    {
        ok = false;
        return;
    }

    for (Slot& slot : impl->slots)
    {
        if (slot.image.channel[0])
            cudaFree(slot.image.channel[0]);
        if (slot.state)
            nvjpegJpegStateDestroy(slot.state);
        if (slot.stream)
            cudaStreamDestroy(slot.stream);
    }
    if (impl->handle)
        nvjpegDestroy(impl->handle);

    delete impl;
    m_impl = nullptr;
    ok = false;
}

bool NvJpegDecoder::init(NvJpegMode mode_in, int w, int h, int inflight_in, std::string& err)
{
    shutdown();

    mode = mode_in;
    width = w;
    height = h;
    inflight = std::max(1, inflight_in);
    init_us = 0;
    warmup_us = 0;

    if (width <= 0 || height <= 0)
    {
        err = "invalid dimensions";
        return false;
    }

    const mango::u64 t0 = nowUs();

    // Force CUDA context creation up front.
    if (!checkCuda(cudaFree(0), "cuda context init", err))
        return false;

    auto* impl = new Impl();
    m_impl = impl;

    if (!checkNv(nvjpegCreateEx(toNvBackend(mode), nullptr, nullptr, 0, &impl->handle),
            "nvjpegCreateEx", err))
    {
        shutdown();
        return false;
    }

    impl->rgb_bytes = size_t(width) * size_t(height) * 3;
    impl->slots.resize(size_t(inflight));

    for (Slot& slot : impl->slots)
    {
        if (!checkNv(nvjpegJpegStateCreate(impl->handle, &slot.state), "nvjpegJpegStateCreate", err) ||
            !checkCuda(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking), "cudaStreamCreate", err) ||
            !checkCuda(cudaMalloc(&slot.image.channel[0], impl->rgb_bytes), "cudaMalloc RGBI", err))
        {
            shutdown();
            return false;
        }
        slot.image.pitch[0] = width * 3;
    }

    init_us = nowUs() - t0;
    ok = true;
    return true;
}

bool NvJpegDecoder::warmup(mango::ConstMemory jpeg, int rounds, std::string& err)
{
    if (!ok)
    {
        err = "decoder not initialized";
        return false;
    }

    rounds = std::max(0, rounds);
    if (!rounds)
        return true;

    const mango::u64 t0 = nowUs();
    for (int i = 0; i < rounds; ++i)
    {
        mango::u64 decode_us = 0;
        if (!decodeToDevice(jpeg, decode_us, err))
            return false;
    }
    warmup_us = nowUs() - t0;
    return true;
}

bool NvJpegDecoder::decodeLaunch(mango::ConstMemory jpeg, int slot_index, std::string& err)
{
    Impl* impl = asImpl(m_impl);
    if (!impl || !ok)
    {
        err = "decoder not initialized";
        return false;
    }
    if (!jpeg.address || !jpeg.size)
    {
        err = "empty JPEG buffer";
        return false;
    }
    if (slot_index < 0 || slot_index >= inflight)
    {
        err = "slot out of range";
        return false;
    }

    Slot& slot = impl->slots[size_t(slot_index)];
    nvjpegStatus_t st = nvjpegDecode(impl->handle, slot.state,
        jpeg.address, jpeg.size,
        NVJPEG_OUTPUT_RGBI, &slot.image, slot.stream);
    return checkNv(st, "nvjpegDecode", err);
}

bool NvJpegDecoder::syncSlot(int slot_index, std::string& err)
{
    Impl* impl = asImpl(m_impl);
    if (!impl || !ok)
    {
        err = "decoder not initialized";
        return false;
    }
    if (slot_index < 0 || slot_index >= inflight)
    {
        err = "slot out of range";
        return false;
    }

    Slot& slot = impl->slots[size_t(slot_index)];
    return checkCuda(cudaStreamSynchronize(slot.stream), "cudaStreamSynchronize", err);
}

bool NvJpegDecoder::decodeToDevice(mango::ConstMemory jpeg, mango::u64& decode_us, std::string& err)
{
    decode_us = 0;
    const mango::u64 t_dec0 = nowUs();
    if (!decodeLaunch(jpeg, 0, err) || !syncSlot(0, err))
        return false;
    decode_us = nowUs() - t_dec0;
    return true;
}

std::uint8_t* NvJpegDecoder::deviceRgbi(int slot_index) const
{
    Impl* impl = asImpl(m_impl);
    if (!impl || !ok || slot_index < 0 || slot_index >= inflight)
        return nullptr;
    return impl->slots[size_t(slot_index)].image.channel[0];
}

int NvJpegDecoder::rgbiPitch() const
{
    return width * 3;
}

void* NvJpegDecoder::decodeStream(int slot_index) const
{
    Impl* impl = asImpl(m_impl);
    if (!impl || !ok || slot_index < 0 || slot_index >= inflight)
        return nullptr;
    return impl->slots[size_t(slot_index)].stream;
}

bool NvJpegDecoder::decodePipeline(mango::ConstMemory jpeg, int count,
    mango::u64& total_us, mango::u64& avg_us, std::string& err)
{
    total_us = 0;
    avg_us = 0;

    Impl* impl = asImpl(m_impl);
    if (!impl || !ok)
    {
        err = "decoder not initialized";
        return false;
    }
    if (!jpeg.address || !jpeg.size)
    {
        err = "empty JPEG buffer";
        return false;
    }
    if (count <= 0)
    {
        err = "count must be > 0";
        return false;
    }

    for (Slot& slot : impl->slots)
        slot.submitted = false;

    const mango::u64 t0 = nowUs();

    for (int i = 0; i < count; ++i)
    {
        const int si = i % inflight;
        Slot& slot = impl->slots[size_t(si)];

        if (slot.submitted)
        {
            if (!syncSlot(si, err))
                return false;
            slot.submitted = false;
        }

        if (!decodeLaunch(jpeg, si, err))
            return false;
        slot.submitted = true;
    }

    for (int si = 0; si < inflight; ++si)
    {
        Slot& slot = impl->slots[size_t(si)];
        if (!slot.submitted)
            continue;
        if (!syncSlot(si, err))
            return false;
        slot.submitted = false;
    }

    total_us = nowUs() - t0;
    avg_us = total_us / mango::u64(count);
    return true;
}

#endif // GPUJPEG_HAS_NVJPEG

} // namespace gpujpeg

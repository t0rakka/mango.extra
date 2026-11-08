#include "roc_backend.hpp"

#include "decoded_texture.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#if defined(GPUJPEG_HAS_SOURCE_ROC)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hiprtc.h>
#include <rocjpeg/rocjpeg.h>
#endif

namespace gpujpeg
{

#if !defined(GPUJPEG_HAS_SOURCE_ROC)

RocRuntimeInfo queryRocRuntime()
{
    RocRuntimeInfo info;
    info.error = "rocJPEG not built";
    return info;
}

bool rocAvailable()
{
    return false;
}

bool RocJpegDecoder::init(RocJpegMode, int, int, int, VkPhysicalDevice, std::string& error)
{
    error = "rocJPEG not built (install rocjpeg / HIP and reconfigure CMake)";
    return false;
}

bool RocJpegDecoder::warmup(mango::ConstMemory, int, std::string& error)
{
    error = "rocJPEG not built";
    return false;
}

bool RocJpegDecoder::decodeToDevice(mango::ConstMemory, mango::u64&, std::string& error)
{
    error = "rocJPEG not built";
    return false;
}

bool RocJpegDecoder::copyRgbToRgba(void*, int, std::string& error)
{
    error = "rocJPEG not built";
    return false;
}

const char* RocJpegDecoder::subsampling() const
{
    return "";
}

void RocJpegDecoder::shutdown()
{
    m_impl = nullptr;
    ok = false;
    m_subsampling = "";
}

#else

namespace
{

// hiprtc source. Interleaved RGB (pitch = bytes/row) → tightly packed RGBA8.
constexpr const char* kRgbToRgbaSource = R"hip(
extern "C" __global__ void gpujpegRgbToRgba(
    const unsigned char* __restrict__ rgb,
    int rgbPitch,
    unsigned char* __restrict__ rgba,
    int width,
    int height)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;
    const unsigned char* s = rgb + (size_t)y * (size_t)rgbPitch + (size_t)x * 3u;
    unsigned char* d = rgba + ((size_t)y * (size_t)width + (size_t)x) * 4u;
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
    d[3] = 255;
}
)hip";

struct Slot
{
    std::uint8_t* rgb = nullptr;
    std::uint32_t pitch = 0;
};

struct Impl
{
    RocJpegHandle handle = nullptr;
    RocJpegStreamHandle stream = nullptr;
    hipModule_t module = nullptr;
    hipFunction_t kernel = nullptr;
    std::vector<Slot> slots;
    RocJpegChromaSubsampling subsampling = ROCJPEG_CSS_UNKNOWN;
};

const char* chromaName(RocJpegChromaSubsampling s)
{
    switch (s)
    {
        case ROCJPEG_CSS_444: return "4:4:4";
        case ROCJPEG_CSS_440: return "4:4:0";
        case ROCJPEG_CSS_422: return "4:2:2";
        case ROCJPEG_CSS_420: return "4:2:0";
        case ROCJPEG_CSS_411: return "4:1:1";
        case ROCJPEG_CSS_400: return "4:0:0";
        default:              return "unknown";
    }
}

std::string statusText(RocJpegStatus st)
{
    const char* name = rocJpegGetErrorName(st);
    if (name && name[0])
        return name;
    return fmt::format("status {}", int(st));
}

bool checkRoc(RocJpegStatus st, const char* what, std::string& error)
{
    if (st == ROCJPEG_STATUS_SUCCESS)
        return true;
    error = fmt::format("{} failed: {}", what, statusText(st));
    return false;
}

bool checkHip(hipError_t e, const char* what, std::string& error)
{
    if (e == hipSuccess)
        return true;
    const char* msg = hipGetErrorString(e);
    error = fmt::format("{} failed: {}", what, msg ? msg : "hip error");
    return false;
}

RocJpegBackend toRocBackend(RocJpegMode mode)
{
    switch (mode)
    {
        case RocJpegMode::Hardware: return ROCJPEG_BACKEND_HARDWARE;
        case RocJpegMode::Hybrid:   return ROCJPEG_BACKEND_HYBRID;
    }
    return ROCJPEG_BACKEND_HARDWARE;
}

std::string stripArch(const char* gcn)
{
    std::string arch = gcn ? gcn : "";
    const auto colon = arch.find(':');
    if (colon != std::string::npos)
        arch.resize(colon);
    return arch;
}

std::string hiprtcLog(hiprtcProgram prog)
{
    size_t log_size = 0;
    if (hiprtcGetProgramLogSize(prog, &log_size) != HIPRTC_SUCCESS || log_size <= 1)
        return {};
    std::string log(log_size, '\0');
    hiprtcGetProgramLog(prog, log.data());
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n'))
        log.pop_back();
    return log;
}

bool compileRgbKernel(const std::string& arch, hipModule_t& module, hipFunction_t& kernel, std::string& error)
{
    hiprtcProgram prog = nullptr;
    hiprtcResult cr = hiprtcCreateProgram(&prog, kRgbToRgbaSource, "gpujpeg_rgb_to_rgba.cpp",
        0, nullptr, nullptr);
    if (cr != HIPRTC_SUCCESS || !prog)
    {
        error = "hiprtcCreateProgram failed";
        return false;
    }

    auto compile = [&](const std::vector<const char*>& opts) -> hiprtcResult
    {
        return hiprtcCompileProgram(prog, int(opts.size()), opts.empty() ? nullptr : opts.data());
    };

    std::string arch_opt;
    std::vector<const char*> opts;
    if (!arch.empty())
    {
        arch_opt = "--offload-arch=" + arch;
        opts.push_back(arch_opt.c_str());
    }

    cr = compile(opts);
    if (cr != HIPRTC_SUCCESS && !opts.empty())
    {
        // Recreate: hiprtc doesn't reset a failed program cleanly on every version.
        hiprtcDestroyProgram(&prog);
        prog = nullptr;
        cr = hiprtcCreateProgram(&prog, kRgbToRgbaSource, "gpujpeg_rgb_to_rgba.cpp",
            0, nullptr, nullptr);
        if (cr != HIPRTC_SUCCESS || !prog)
        {
            error = "hiprtcCreateProgram failed";
            return false;
        }
        cr = compile({});
    }

    if (cr != HIPRTC_SUCCESS)
    {
        const std::string log = hiprtcLog(prog);
        hiprtcDestroyProgram(&prog);
        error = log.empty() ? "hiprtcCompileProgram failed" : fmt::format("hiprtcCompileProgram failed: {}", log);
        return false;
    }

    size_t code_size = 0;
    cr = hiprtcGetCodeSize(prog, &code_size);
    if (cr != HIPRTC_SUCCESS || code_size == 0)
    {
        hiprtcDestroyProgram(&prog);
        error = "hiprtcGetCodeSize failed";
        return false;
    }

    std::vector<char> code(code_size);
    cr = hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);
    if (cr != HIPRTC_SUCCESS)
    {
        error = "hiprtcGetCode failed";
        return false;
    }

    if (!checkHip(hipModuleLoadData(&module, code.data()), "hipModuleLoadData", error))
        return false;
    if (!checkHip(hipModuleGetFunction(&kernel, module, "gpujpegRgbToRgba"),
            "hipModuleGetFunction(gpujpegRgbToRgba)", error))
    {
        (void)hipModuleUnload(module);
        module = nullptr;
        return false;
    }
    return true;
}

bool probeBackend(int device, RocJpegBackend backend, std::string& error)
{
    if (!checkHip(hipSetDevice(device), "hipSetDevice", error))
        return false;

    RocJpegHandle handle = nullptr;
    const RocJpegStatus st = rocJpegCreate(backend, device, &handle);
    if (st == ROCJPEG_STATUS_SUCCESS)
    {
        rocJpegDestroy(handle);
        return true;
    }
    error = statusText(st);
    if (handle)
        rocJpegDestroy(handle);
    return false;
}

Impl* asImpl(void* p)
{
    return static_cast<Impl*>(p);
}

RocRuntimeInfo probeRuntime()
{
    RocRuntimeInfo info;
    int count = 0;
    const hipError_t ce = hipGetDeviceCount(&count);
    if (ce != hipSuccess || count <= 0)
    {
        info.error = (ce != hipSuccess)
            ? fmt::format("hipGetDeviceCount failed: {}", hipGetErrorString(ce))
            : "no HIP devices";
        return info;
    }

    info.hip_present = true;
    info.devices.reserve(size_t(count));
    for (int i = 0; i < count; ++i)
    {
        RocDeviceStatus dev;
        dev.index = i;
        hipDeviceProp_t prop {};
        if (hipGetDeviceProperties(&prop, i) == hipSuccess)
        {
            dev.name = prop.name;
            dev.arch = stripArch(prop.gcnArchName);
        }
        else
        {
            dev.name = "unknown";
        }

        dev.hardware = probeBackend(i, ROCJPEG_BACKEND_HARDWARE, dev.hardware_error);
        dev.hybrid = probeBackend(i, ROCJPEG_BACKEND_HYBRID, dev.hybrid_error);
        info.hardware = info.hardware || dev.hardware;
        info.hybrid = info.hybrid || dev.hybrid;
        info.devices.push_back(std::move(dev));
    }
    return info;
}

} // namespace

RocRuntimeInfo queryRocRuntime()
{
    static const RocRuntimeInfo cached = probeRuntime();
    return cached;
}

bool rocAvailable()
{
    const RocRuntimeInfo info = queryRocRuntime();
    return info.hardware || info.hybrid;
}

void RocJpegDecoder::shutdown()
{
    Impl* impl = asImpl(m_impl);
    if (!impl)
    {
        ok = false;
        m_subsampling = "";
        return;
    }

    for (Slot& slot : impl->slots)
    {
        if (slot.rgb)
            (void)hipFree(slot.rgb);
    }
    if (impl->module)
        (void)hipModuleUnload(impl->module);
    if (impl->stream)
        rocJpegStreamDestroy(impl->stream);
    if (impl->handle)
        rocJpegDestroy(impl->handle);

    delete impl;
    m_impl = nullptr;
    ok = false;
    m_subsampling = "";
}

bool RocJpegDecoder::init(RocJpegMode mode_in, int w, int h, int inflight_in,
    VkPhysicalDevice physical, std::string& err)
{
    shutdown();

    mode = mode_in;
    width = w;
    height = h;
    inflight = std::max(1, inflight_in);
    hip_device = -1;
    device_name.clear();
    arch_name.clear();
    init_us = 0;
    warmup_us = 0;

    if (width <= 0 || height <= 0)
    {
        err = "invalid dimensions";
        return false;
    }
    if (!physical)
    {
        err = "Vulkan physical device missing";
        return false;
    }

    const mango::u64 t0 = nowUs();

    int device = -1;
    if (!selectHipDeviceForVulkan(physical, &device, err))
        return false;
    hip_device = device;

    hipDeviceProp_t prop {};
    if (!checkHip(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties", err))
        return false;
    device_name = prop.name;
    arch_name = stripArch(prop.gcnArchName);

    // Touch the context on this device before rocJPEG / hiprtc.
    if (!checkHip(hipFree(nullptr), "hip context init", err))
        return false;

    auto* impl = new Impl();
    m_impl = impl;

    if (!checkRoc(rocJpegCreate(toRocBackend(mode), device, &impl->handle), "rocJpegCreate", err) ||
        !checkRoc(rocJpegStreamCreate(&impl->stream), "rocJpegStreamCreate", err))
    {
        shutdown();
        return false;
    }

    if (!compileRgbKernel(arch_name, impl->module, impl->kernel, err))
    {
        shutdown();
        return false;
    }

    const std::uint32_t pitch = std::uint32_t(width) * 3u;
    const size_t bytes = size_t(pitch) * size_t(height);
    impl->slots.resize(size_t(inflight));
    for (Slot& slot : impl->slots)
    {
        slot.pitch = pitch;
        if (!checkHip(hipMalloc(reinterpret_cast<void**>(&slot.rgb), bytes), "hipMalloc RGB", err))
        {
            shutdown();
            return false;
        }
    }

    init_us = nowUs() - t0;
    ok = true;
    return true;
}

bool RocJpegDecoder::warmup(mango::ConstMemory jpeg, int rounds, std::string& err)
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

bool RocJpegDecoder::decodeToDevice(mango::ConstMemory jpeg, mango::u64& decode_us, std::string& err)
{
    decode_us = 0;
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

    if (hip_device >= 0 && !checkHip(hipSetDevice(hip_device), "hipSetDevice", err))
        return false;

    const mango::u64 t0 = nowUs();

    // rocJPEG does not copy the bitstream. `jpeg` must stay alive until decode returns.
    if (!checkRoc(rocJpegStreamParse(jpeg.address, jpeg.size, impl->stream), "rocJpegStreamParse", err))
        return false;

    std::uint8_t components = 0;
    RocJpegChromaSubsampling subsampling = ROCJPEG_CSS_UNKNOWN;
    std::uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
    std::uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
    if (!checkRoc(rocJpegGetImageInfo(impl->handle, impl->stream, &components, &subsampling, widths, heights),
            "rocJpegGetImageInfo", err))
        return false;

    if (int(widths[0]) != width || int(heights[0]) != height)
    {
        err = fmt::format("rocJPEG size {}x{} does not match header {}x{}",
            widths[0], heights[0], width, height);
        return false;
    }
    impl->subsampling = subsampling;
    m_subsampling = chromaName(subsampling);

    RocJpegDecodeParams params {};
    params.output_format = ROCJPEG_OUTPUT_RGB;

    Slot& slot = impl->slots[0];
    RocJpegImage image {};
    image.channel[0] = slot.rgb;
    image.pitch[0] = slot.pitch;

    if (!checkRoc(rocJpegDecode(impl->handle, impl->stream, &params, &image), "rocJpegDecode", err))
        return false;

    decode_us = nowUs() - t0;
    return true;
}

bool RocJpegDecoder::copyRgbToRgba(void* dst_rgba, int slot_index, std::string& err)
{
    Impl* impl = asImpl(m_impl);
    if (!impl || !ok || !impl->kernel)
    {
        err = "decoder not initialized";
        return false;
    }
    if (!dst_rgba)
    {
        err = "HIP RGBA export pointer missing";
        return false;
    }
    if (slot_index < 0 || slot_index >= inflight)
    {
        err = "slot out of range";
        return false;
    }

    if (hip_device >= 0 && !checkHip(hipSetDevice(hip_device), "hipSetDevice", err))
        return false;

    Slot& slot = impl->slots[size_t(slot_index)];
    int w = width;
    int h = height;
    int pitch = int(slot.pitch);
    std::uint8_t* src = slot.rgb;
    auto* dst = static_cast<std::uint8_t*>(dst_rgba);
    void* args[] = { &src, &pitch, &dst, &w, &h };

    const unsigned bx = 16;
    const unsigned by = 16;
    const unsigned gx = (unsigned(width) + bx - 1) / bx;
    const unsigned gy = (unsigned(height) + by - 1) / by;
    if (!checkHip(hipModuleLaunchKernel(impl->kernel, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr),
            "hipModuleLaunchKernel", err))
        return false;
    return checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize", err);
}

const char* RocJpegDecoder::subsampling() const
{
    return m_subsampling ? m_subsampling : "";
}

#endif // GPUJPEG_HAS_SOURCE_ROC

} // namespace gpujpeg

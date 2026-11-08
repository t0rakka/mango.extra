#pragma once

// Support matrix (workbench → later mango plugin):
//   Sources:  MANGO | NVJPEG | ROC  (+ Intel later)
//   Outputs:  VK | GL | D3D11 | D3D12
// Available = CMake enabled && SDK detected && runtime present.

namespace gpujpeg
{

enum class DecodeSource
{
    Mango,
    MangoHybrid, // CPU Spectral Huffman → (later) GPU compute
    NvJpeg,
    Roc,
    AppleVt,
};

enum class OutputApi
{
    Vulkan,
    OpenGL,
    D3D11,
    D3D12,
};

inline const char* decodeSourceName(DecodeSource s)
{
    switch (s)
    {
        case DecodeSource::Mango:       return "mango";
        case DecodeSource::MangoHybrid: return "mango-hybrid";
        case DecodeSource::NvJpeg:      return "nvjpeg";
        case DecodeSource::Roc:         return "roc";
        case DecodeSource::AppleVt:    return "apple-vt";
    }
    return "?";
}

inline const char* outputApiName(OutputApi a)
{
    switch (a)
    {
        case OutputApi::Vulkan: return "vulkan";
        case OutputApi::OpenGL: return "opengl";
        case OutputApi::D3D11:  return "d3d11";
        case OutputApi::D3D12:  return "d3d12";
    }
    return "?";
}

inline bool sourceBuilt(DecodeSource s)
{
    switch (s)
    {
        case DecodeSource::Mango:
#if defined(GPUJPEG_HAS_SOURCE_MANGO)
            return true;
#else
            return false;
#endif
        case DecodeSource::MangoHybrid:
#if defined(GPUJPEG_HAS_SOURCE_MANGO_HYBRID)
            return true;
#else
            return false;
#endif
        case DecodeSource::NvJpeg:
#if defined(GPUJPEG_HAS_SOURCE_NVJPEG)
            return true;
#else
            return false;
#endif
        case DecodeSource::AppleVt:
#if defined(GPUJPEG_HAS_SOURCE_APPLE_VT)
            return true;
#else
            return false;
#endif
        case DecodeSource::Roc:
#if defined(GPUJPEG_HAS_SOURCE_ROC)
            return true;
#else
            return false;
#endif
    }
    return false;
}

inline bool outputBuilt(OutputApi a)
{
    switch (a)
    {
        case OutputApi::Vulkan:
#if defined(GPUJPEG_HAS_OUTPUT_VULKAN)
            return true;
#else
            return false;
#endif
        case OutputApi::OpenGL:
#if defined(GPUJPEG_HAS_OUTPUT_OPENGL)
            return true;
#else
            return false;
#endif
        case OutputApi::D3D11:
#if defined(GPUJPEG_HAS_OUTPUT_D3D11)
            return true;
#else
            return false;
#endif
        case OutputApi::D3D12:
#if defined(GPUJPEG_HAS_OUTPUT_D3D12)
            return true;
#else
            return false;
#endif
    }
    return false;
}

} // namespace gpujpeg

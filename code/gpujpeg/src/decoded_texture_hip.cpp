#include "decoded_texture.hpp"

#include <mango/core/core.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string_view>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif

#if defined(GPUJPEG_HAS_SOURCE_ROC)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>
#endif

namespace gpujpeg
{

namespace
{

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits kOpaqueExportHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits kOpaqueExportHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

bool containsCi(std::string_view hay, std::string_view needle)
{
    if (needle.size() < 6 || hay.size() < needle.size())
        return false;
    const auto lower = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };
    return std::search(hay.begin(), hay.end(), needle.begin(), needle.end(),
        [&](char a, char b) { return lower(static_cast<unsigned char>(a)) == lower(static_cast<unsigned char>(b)); })
        != hay.end();
}

} // namespace

bool importHipExternalMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size,
    size_t map_bytes, void*& out_rgba, void*& out_ext, std::string& error)
{
#if !defined(GPUJPEG_HAS_SOURCE_ROC)
    MANGO_UNREFERENCED(device);
    MANGO_UNREFERENCED(memory);
    MANGO_UNREFERENCED(size);
    MANGO_UNREFERENCED(map_bytes);
    MANGO_UNREFERENCED(out_rgba);
    MANGO_UNREFERENCED(out_ext);
    error = "rocJPEG/HIP not built";
    return false;
#else
#if defined(_WIN32)
    auto getHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (!getHandle)
    {
        error = "vkGetMemoryWin32HandleKHR missing (enable VK_KHR_external_memory_win32)";
        return false;
    }

    VkMemoryGetWin32HandleInfoKHR get_info =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .memory = memory,
        .handleType = kOpaqueExportHandleType,
    };
    HANDLE handle = nullptr;
    if (getHandle(device, &get_info, &handle) != VK_SUCCESS || !handle)
    {
        error = "vkGetMemoryWin32HandleKHR failed";
        return false;
    }

    hipExternalMemoryHandleDesc desc {};
    desc.type = hipExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;

    hipExternalMemory_t ext = nullptr;
    hipError_t he = hipImportExternalMemory(&ext, &desc);
    if (he != hipSuccess)
        CloseHandle(handle);
#else
    auto getHandle = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
    if (!getHandle)
    {
        error = "vkGetMemoryFdKHR missing (enable VK_KHR_external_memory_fd)";
        return false;
    }

    VkMemoryGetFdInfoKHR get_info =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memory,
        .handleType = kOpaqueExportHandleType,
    };
    int fd = -1;
    if (getHandle(device, &get_info, &fd) != VK_SUCCESS || fd < 0)
    {
        error = "vkGetMemoryFdKHR failed";
        return false;
    }

    hipExternalMemoryHandleDesc desc {};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;

    hipExternalMemory_t ext = nullptr;
    hipError_t he = hipImportExternalMemory(&ext, &desc);
    // On success HIP owns the fd. On failure we close it.
    if (he != hipSuccess)
        ::close(fd);
#endif

    if (he != hipSuccess)
    {
        const char* msg = hipGetErrorString(he);
        error = fmt::format("hipImportExternalMemory: {}", msg ? msg : "hip error");
        return false;
    }

    hipExternalMemoryBufferDesc buf_desc {};
    buf_desc.offset = 0;
    buf_desc.size = map_bytes;
    buf_desc.flags = 0;
    void* mapped = nullptr;
    he = hipExternalMemoryGetMappedBuffer(&mapped, ext, &buf_desc);
    if (he != hipSuccess)
    {
        (void)hipDestroyExternalMemory(ext);
        const char* msg = hipGetErrorString(he);
        error = fmt::format("hipExternalMemoryGetMappedBuffer: {}", msg ? msg : "hip error");
        return false;
    }

    out_rgba = mapped;
    out_ext = ext;
    return true;
#endif
}

void releaseHipExternalMemory(void*& hip_rgba, void*& hip_ext)
{
    hip_rgba = nullptr;
#if defined(GPUJPEG_HAS_SOURCE_ROC)
    if (hip_ext)
        (void)hipDestroyExternalMemory(static_cast<hipExternalMemory_t>(hip_ext));
#endif
    hip_ext = nullptr;
}

bool selectHipDeviceForVulkan(VkPhysicalDevice physical, int* out_index, std::string& error)
{
    if (out_index)
        *out_index = -1;
#if !defined(GPUJPEG_HAS_SOURCE_ROC)
    MANGO_UNREFERENCED(physical);
    error = "rocJPEG/HIP not built";
    return false;
#else
    VkPhysicalDeviceProperties vk_props {};
    vkGetPhysicalDeviceProperties(physical, &vk_props);

    VkPhysicalDeviceIDProperties id_props =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props2 =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &id_props,
    };
    vkGetPhysicalDeviceProperties2(physical, &props2);

    int count = 0;
    const hipError_t ce = hipGetDeviceCount(&count);
    if (ce != hipSuccess || count <= 0)
    {
        error = (ce != hipSuccess)
            ? fmt::format("hipGetDeviceCount: {}", hipGetErrorString(ce))
            : "no HIP devices";
        return false;
    }

    int uuid_match = -1;
    int name_match = -1;
    int name_matches = 0;
    const std::string_view vk_name(vk_props.deviceName);

    for (int i = 0; i < count; ++i)
    {
        hipUUID uuid {};
        if (hipDeviceGetUuid(&uuid, i) == hipSuccess &&
            std::memcmp(uuid.bytes, id_props.deviceUUID, VK_UUID_SIZE) == 0)
        {
            uuid_match = i;
            break;
        }

        hipDeviceProp_t prop {};
        if (hipGetDeviceProperties(&prop, i) != hipSuccess)
            continue;
        std::string arch = prop.gcnArchName;
        const auto colon = arch.find(':');
        if (colon != std::string::npos)
            arch.resize(colon);
        if (containsCi(vk_name, prop.name) || containsCi(vk_name, arch))
        {
            name_match = i;
            ++name_matches;
        }
    }

    int chosen = uuid_match;
    const char* how = "UUID";
    if (chosen < 0 && name_matches == 1)
    {
        chosen = name_match;
        how = "name";
    }
    else if (chosen < 0 && count == 1)
    {
        chosen = 0;
        how = "only-device";
    }

    if (chosen < 0)
    {
        error = fmt::format("no HIP device matches Vulkan \"{}\" ({} HIP device(s))",
            vk_props.deviceName, count);
        return false;
    }

    if (hipSetDevice(chosen) != hipSuccess)
    {
        error = "hipSetDevice failed";
        return false;
    }
    if (out_index)
        *out_index = chosen;

    hipDeviceProp_t chosen_prop {};
    if (hipGetDeviceProperties(&chosen_prop, chosen) == hipSuccess)
    {
        mango::printLine("HIP device {} \"{}\" ({}) selected for Vulkan \"{}\" ({})",
            chosen, chosen_prop.name, chosen_prop.gcnArchName, vk_props.deviceName, how);
    }
    return true;
#endif
}

} // namespace gpujpeg

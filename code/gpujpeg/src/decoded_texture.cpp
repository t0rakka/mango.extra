#include "decoded_texture.hpp"

#include <mango/core/core.hpp>
#include <mango/vulkan/vulkan.hpp>

#include <cstring>
#include <cstdint>

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

#if defined(GPUJPEG_HAS_NVJPEG)
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_data_exchange_and_initialization.h>
#endif

namespace gpujpeg
{

void appendCudaVkInteropDeviceExtensions(mango::vulkan::VulkanDeviceConfig& config)
{
    // String forms avoid pulling vulkan_win32.h / vulkan_fuchsia.h into callers.
#if defined(_WIN32)
    config.deviceExtensions.push_back("VK_KHR_external_memory_win32");
#else
    config.deviceExtensions.push_back("VK_KHR_external_memory_fd");
#endif
}

namespace
{

using mango::vulkan::Allocator;
using mango::vulkan::BufferAllocation;
using mango::vulkan::ImageAllocation;
using mango::vulkan::MemoryUsage;
using mango::vulkan::VulkanWindow;

void imageBarrier(VkCommandBuffer cmd, VkImage image,
    VkImageLayout old_layout, VkImageLayout new_layout,
    VkAccessFlags src_access, VkAccessFlags dst_access,
    VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
    VkImageMemoryBarrier barrier =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits kOpaqueExportHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits kOpaqueExportHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

#if defined(GPUJPEG_HAS_NVJPEG)
bool fillNppContext(cudaStream_t stream, NppStreamContext& ctx, std::string& error)
{
    std::memset(&ctx, 0, sizeof(ctx));
    ctx.hStream = stream;
    if (cudaGetDevice(&ctx.nCudaDeviceId) != cudaSuccess)
    {
        error = "cudaGetDevice failed";
        return false;
    }
    cudaDeviceProp props {};
    if (cudaGetDeviceProperties(&props, ctx.nCudaDeviceId) != cudaSuccess)
    {
        error = "cudaGetDeviceProperties failed";
        return false;
    }
    ctx.nMultiProcessorCount = props.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = props.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = props.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = props.minor;
    cudaStreamGetFlags(stream, &ctx.nStreamFlags);
    return true;
}

bool importCudaExternalMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size,
    size_t map_bytes, void*& out_rgba, void*& out_ext, std::string& error)
{
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

    cudaExternalMemoryHandleDesc desc {};
    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;

    cudaExternalMemory_t ext = nullptr;
    cudaError_t ce = cudaImportExternalMemory(&ext, &desc);
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

    cudaExternalMemoryHandleDesc desc {};
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;

    cudaExternalMemory_t ext = nullptr;
    cudaError_t ce = cudaImportExternalMemory(&ext, &desc);
    // On success CUDA owns the fd; on failure we must close it.
    if (ce != cudaSuccess)
        ::close(fd);
#endif

    if (ce != cudaSuccess)
    {
        error = fmt::format("cudaImportExternalMemory: {}", cudaGetErrorString(ce));
        return false;
    }

    cudaExternalMemoryBufferDesc buf_desc {};
    buf_desc.offset = 0;
    buf_desc.size = map_bytes;
    void* mapped = nullptr;
    ce = cudaExternalMemoryGetMappedBuffer(&mapped, ext, &buf_desc);
    if (ce != cudaSuccess)
    {
        cudaDestroyExternalMemory(ext);
        error = fmt::format("cudaExternalMemoryGetMappedBuffer: {}", cudaGetErrorString(ce));
        return false;
    }

    out_rgba = mapped;
    out_ext = ext;
    return true;
}
#endif

} // namespace

// HIP entry points live in decoded_texture_hip.cpp so this TU never includes hip_runtime.h
// (CUDA and HIP vector types conflict when both SDKs are enabled).
bool importHipExternalMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size,
    size_t map_bytes, void*& out_rgba, void*& out_ext, std::string& error);
void releaseHipExternalMemory(void*& hip_rgba, void*& hip_ext);

bool selectCudaDeviceForVulkan(VkPhysicalDevice physical, std::string& error)
{
#if !defined(GPUJPEG_HAS_NVJPEG)
    error = "CUDA not built";
    return false;
#else
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
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
    {
        error = "no CUDA devices";
        return false;
    }

    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp prop {};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
            continue;
        if (std::memcmp(prop.uuid.bytes, id_props.deviceUUID, VK_UUID_SIZE) == 0)
        {
            if (cudaSetDevice(i) != cudaSuccess)
            {
                error = "cudaSetDevice failed";
                return false;
            }
            return true;
        }
    }

    error = "no CUDA device UUID matches Vulkan physical device";
    return false;
#endif
}

bool DecodedTexture::createCudaExportBuffer(std::string& error)
{
#if !defined(GPUJPEG_HAS_NVJPEG)
    error = "CUDA not built";
    return false;
#else
    if (!selectCudaDeviceForVulkan(physical, error))
        return false;

    VkExternalMemoryBufferCreateInfo ext_buf =
    {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = kOpaqueExportHandleType,
    };
    VkBufferCreateInfo cuda_info =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext_buf,
        .size = rgba_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(device, &cuda_info, nullptr, &cuda_buffer) != VK_SUCCESS)
    {
        error = "cuda export buffer create failed";
        return false;
    }

    VkMemoryRequirements cuda_req {};
    vkGetBufferMemoryRequirements(device, cuda_buffer, &cuda_req);

    VkExportMemoryAllocateInfo export_info =
    {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = kOpaqueExportHandleType,
    };

    // Prefer dedicated allocation for exportable memory (driver-friendly).
    VkMemoryDedicatedAllocateInfo dedicated =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &export_info,
        .image = VK_NULL_HANDLE,
        .buffer = cuda_buffer,
    };

    VkPhysicalDeviceMemoryProperties props {};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    uint32_t type_index = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    {
        if ((cuda_req.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        {
            type_index = i;
            break;
        }
    }
    if (type_index == UINT32_MAX)
    {
        error = "no device-local memory for cuda buffer";
        return false;
    }

    VkMemoryAllocateInfo cuda_alloc =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .allocationSize = cuda_req.size,
        .memoryTypeIndex = type_index,
    };
    if (vkAllocateMemory(device, &cuda_alloc, nullptr, &cuda_buffer_memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, cuda_buffer, cuda_buffer_memory, 0) != VK_SUCCESS)
    {
        error = "cuda buffer memory failed";
        return false;
    }

    if (!importCudaExternalMemory(device, cuda_buffer_memory, cuda_req.size,
            rgba_bytes, cuda_rgba, cuda_ext_memory, error))
        return false;

    cuda_ready = true;
    return true;
#endif
}

bool DecodedTexture::createHipExportBuffer(std::string& error)
{
#if !defined(GPUJPEG_HAS_SOURCE_ROC)
    error = "rocJPEG/HIP not built";
    return false;
#else
    if (!selectHipDeviceForVulkan(physical, nullptr, error))
        return false;

    VkExternalMemoryBufferCreateInfo ext_buf =
    {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = kOpaqueExportHandleType,
    };
    VkBufferCreateInfo hip_info =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext_buf,
        .size = rgba_bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(device, &hip_info, nullptr, &hip_buffer) != VK_SUCCESS)
    {
        error = "HIP export buffer create failed";
        return false;
    }

    VkMemoryRequirements hip_req {};
    vkGetBufferMemoryRequirements(device, hip_buffer, &hip_req);

    VkExportMemoryAllocateInfo export_info =
    {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = kOpaqueExportHandleType,
    };
    VkMemoryDedicatedAllocateInfo dedicated =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &export_info,
        .image = VK_NULL_HANDLE,
        .buffer = hip_buffer,
    };

    VkPhysicalDeviceMemoryProperties props {};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    uint32_t type_index = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    {
        if ((hip_req.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        {
            type_index = i;
            break;
        }
    }
    if (type_index == UINT32_MAX)
    {
        error = "no device-local memory for HIP buffer";
        return false;
    }

    VkMemoryAllocateInfo hip_alloc =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .allocationSize = hip_req.size,
        .memoryTypeIndex = type_index,
    };
    if (vkAllocateMemory(device, &hip_alloc, nullptr, &hip_buffer_memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, hip_buffer, hip_buffer_memory, 0) != VK_SUCCESS)
    {
        error = "HIP buffer memory failed";
        return false;
    }

    if (!importHipExternalMemory(device, hip_buffer_memory, hip_req.size,
            rgba_bytes, hip_rgba, hip_ext_memory, error))
        return false;

    hip_ready = true;
    return true;
#endif
}

bool DecodedTexture::create(VulkanWindow& window, Allocator& alloc,
    int w, int h, bool enable_cuda_export, std::string& error, bool enable_hip_export)
{
    destroy();

    if (w <= 0 || h <= 0)
    {
        error = "invalid texture size";
        return false;
    }
    if (!alloc.valid())
    {
        error = "Vulkan Allocator not ready";
        return false;
    }

    width = w;
    height = h;
    allocator = &alloc;
    device = window.device();
    physical = window.physicalDevice();
    rgba_bytes = size_t(w) * size_t(h) * 4;

    VkImageCreateInfo image_info =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = { uint32_t(w), uint32_t(h), 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT
               | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
               | VK_IMAGE_USAGE_SAMPLED_BIT
               | VK_IMAGE_USAGE_STORAGE_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    // Compute imageStore does not maintain AMD DCC metadata. A later blit can
    // sample that metadata while a buffer copy still sees the raw texels.
    // VK_IMAGE_COMPRESSION_DISABLED_EXT keeps the two views of the image the same.
    VkImageCompressionControlEXT compression =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT,
        .flags = VK_IMAGE_COMPRESSION_DISABLED_EXT,
    };
    mango::vulkan::DeviceExtensionProperties device_ext(physical);
    if (device_ext.contains(VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME))
    {
        VkPhysicalDeviceImageCompressionControlFeaturesEXT compression_feature =
        {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT,
        };
        VkPhysicalDeviceFeatures2 features2 =
        {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &compression_feature,
        };
        vkGetPhysicalDeviceFeatures2(physical, &features2);
        if (compression_feature.imageCompressionControl)
            image_info.pNext = &compression;
    }

    image_alloc = alloc.createImage(image_info, MemoryUsage::GpuOnly);
    if (!image_alloc)
    {
        error = "Allocator::createImage failed";
        destroy();
        return false;
    }
    image = image_alloc.image;

    VkImageViewCreateInfo view_info =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    if (vkCreateImageView(device, &view_info, nullptr, &view) != VK_SUCCESS)
    {
        error = "vkCreateImageView failed";
        destroy();
        return false;
    }

    layout = VK_IMAGE_LAYOUT_UNDEFINED;

    // SRC feeds the upload copy. DST is the GPUJPEG_CHECK image readback.
    host_alloc = alloc.createBuffer(rgba_bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        MemoryUsage::Upload, true);
    if (!host_alloc || !host_alloc.mapped)
    {
        error = "Allocator::createBuffer (host staging) failed";
        destroy();
        return false;
    }
    host_mapped = host_alloc.mapped;

    // Compression control is off on this device. Keep a device-local buffer so
    // compute stores can be rewritten through vkCmdCopyBufferToImage.
    if (!image_info.pNext)
    {
        publish_alloc = alloc.createBuffer(rgba_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            MemoryUsage::GpuOnly, false);
        if (!publish_alloc)
        {
            error = "Allocator::createBuffer (publish) failed";
            destroy();
            return false;
        }
    }

    if (enable_cuda_export)
    {
        if (!createCudaExportBuffer(error))
        {
            destroy();
            return false;
        }
    }
    if (enable_hip_export)
    {
        if (!createHipExportBuffer(error))
        {
            destroy();
            return false;
        }
    }

    return true;
}

void DecodedTexture::destroy()
{
#if defined(GPUJPEG_HAS_NVJPEG)
    if (cuda_rgba)
    {
        cudaFree(cuda_rgba); // mapped external buffer freed with cudaFree per CUDA docs
        cuda_rgba = nullptr;
    }
    if (cuda_ext_memory)
    {
        cudaDestroyExternalMemory(static_cast<cudaExternalMemory_t>(cuda_ext_memory));
        cuda_ext_memory = nullptr;
    }
#endif
    cuda_ready = false;

    releaseHipExternalMemory(hip_rgba, hip_ext_memory);
    hip_ready = false;

    if (device && cuda_buffer)
        vkDestroyBuffer(device, cuda_buffer, nullptr);
    cuda_buffer = VK_NULL_HANDLE;
    if (device && cuda_buffer_memory)
        vkFreeMemory(device, cuda_buffer_memory, nullptr);
    cuda_buffer_memory = VK_NULL_HANDLE;

    if (device && hip_buffer)
        vkDestroyBuffer(device, hip_buffer, nullptr);
    hip_buffer = VK_NULL_HANDLE;
    if (device && hip_buffer_memory)
        vkFreeMemory(device, hip_buffer_memory, nullptr);
    hip_buffer_memory = VK_NULL_HANDLE;

    releaseUpload();

    if (device && view)
        vkDestroyImageView(device, view, nullptr);
    view = VK_NULL_HANDLE;

    if (allocator)
    {
        if (host_alloc)
            allocator->destroyBuffer(host_alloc);
        if (publish_alloc)
            allocator->destroyBuffer(publish_alloc);
        if (image_alloc)
            allocator->destroyImage(image_alloc);
    }
    host_alloc = {};
    publish_alloc = {};
    image_alloc = {};
    host_mapped = nullptr;
    image = VK_NULL_HANDLE;

    layout = VK_IMAGE_LAYOUT_UNDEFINED;
    allocator = nullptr;
    device = VK_NULL_HANDLE;
    physical = VK_NULL_HANDLE;
    width = height = 0;
    rgba_bytes = 0;
}

bool DecodedTexture::writeHostRgba(const std::uint8_t* rgba, size_t bytes, std::string& error)
{
    if (!host_mapped || bytes > rgba_bytes)
    {
        error = "host staging not ready";
        return false;
    }
    std::memcpy(host_mapped, rgba, bytes);
    return true;
}

#if defined(GPUJPEG_HAS_NVJPEG)
bool DecodedTexture::writeDeviceRgbiToCudaRgba(const std::uint8_t* d_rgbi, int rgbi_pitch,
    cudaStream_t stream, std::string& error)
{
    if (!cuda_ready || !cuda_rgba || !d_rgbi)
    {
        error = "CUDA RGBA export buffer not ready";
        return false;
    }

    NppStreamContext npp {};
    if (!fillNppContext(stream, npp, error))
        return false;

    const int order[4] = { 0, 1, 2, 3 }; // RGB + nValue alpha
    NppiSize roi { width, height };
    NppStatus st = nppiSwapChannels_8u_C3C4R_Ctx(
        d_rgbi, rgbi_pitch,
        static_cast<Npp8u*>(cuda_rgba), width * 4,
        roi, order, 255, npp);
    if (st != NPP_SUCCESS)
    {
        error = fmt::format("nppiSwapChannels_8u_C3C4R failed ({})", int(st));
        return false;
    }
    return true;
}
#endif

void DecodedTexture::recordCopyHostToImage(VkCommandBuffer cmd)
{
    imageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkBufferImageCopy region =
    {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { uint32_t(width), uint32_t(height), 1 },
    };
    vkCmdCopyBufferToImage(cmd, host_alloc.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    imageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void DecodedTexture::releaseUpload()
{
    if (device && upload_pending && upload_fence)
        vkWaitForFences(device, 1, &upload_fence, VK_TRUE, UINT64_MAX);
    upload_pending = false;
    if (device && upload_fence)
        vkDestroyFence(device, upload_fence, nullptr);
    upload_fence = VK_NULL_HANDLE;
    upload_cmd = VK_NULL_HANDLE;
    if (device && upload_pool)
        vkDestroyCommandPool(device, upload_pool, nullptr);
    upload_pool = VK_NULL_HANDLE;
    upload_family = UINT32_MAX;
}

bool DecodedTexture::ensureUpload(uint32_t queue_family, std::string& error)
{
    if (upload_pool && upload_family == queue_family)
        return true;

    releaseUpload();

    VkCommandPoolCreateInfo pool_info =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family,
    };
    if (vkCreateCommandPool(device, &pool_info, nullptr, &upload_pool) != VK_SUCCESS)
    {
        error = "vkCreateCommandPool failed";
        return false;
    }

    VkCommandBufferAllocateInfo alloc_info =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = upload_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(device, &alloc_info, &upload_cmd) != VK_SUCCESS)
    {
        error = "vkAllocateCommandBuffers failed";
        releaseUpload();
        return false;
    }

    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(device, &fence_info, nullptr, &upload_fence) != VK_SUCCESS)
    {
        error = "vkCreateFence failed";
        releaseUpload();
        return false;
    }

    upload_family = queue_family;
    return true;
}

bool DecodedTexture::waitCopy(std::string& error)
{
    if (!upload_pending)
        return true;
    if (!device || !upload_fence)
    {
        error = "upload fence missing";
        return false;
    }
    if (vkWaitForFences(device, 1, &upload_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
    {
        error = "vkWaitForFences (texture upload) failed";
        return false;
    }
    vkResetFences(device, 1, &upload_fence);
    upload_pending = false;
    return true;
}

bool DecodedTexture::submitCopy(VkQueue queue, uint32_t queue_family, int source, std::string& error)
{
    // source: 0 host, 1 cuda, 2 hip
    if (!device || !image)
    {
        error = "texture not created";
        return false;
    }
    if (source == 1 && !cuda_buffer)
    {
        error = "cuda buffer missing";
        return false;
    }
    if (source == 2 && !hip_buffer)
    {
        error = "HIP buffer missing";
        return false;
    }
    if (source == 0 && !host_alloc)
    {
        error = "host buffer missing";
        return false;
    }
    if (!ensureUpload(queue_family, error) || !waitCopy(error))
        return false;

    if (source == 0 && allocator)
        allocator->flush(host_alloc.allocation, 0, rgba_bytes);

    vkResetCommandBuffer(upload_cmd, 0);
    VkCommandBufferBeginInfo begin =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(upload_cmd, &begin);
    if (source == 1)
        recordCopyCudaToImage(upload_cmd);
    else if (source == 2)
        recordCopyBufferToImage(upload_cmd, hip_buffer);
    else
        recordCopyHostToImage(upload_cmd);
    vkEndCommandBuffer(upload_cmd);

    VkSubmitInfo submit =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &upload_cmd,
    };
    if (vkQueueSubmit(queue, 1, &submit, upload_fence) != VK_SUCCESS)
    {
        error = "vkQueueSubmit (texture upload) failed";
        return false;
    }
    upload_pending = true;
    return true;
}

bool DecodedTexture::flushCopy(VkQueue queue, uint32_t queue_family, int source, std::string& error)
{
    return submitCopy(queue, queue_family, source, error) && waitCopy(error);
}

bool DecodedTexture::submitCopyHostToImage(VkQueue queue, uint32_t queue_family, std::string& error)
{
    return submitCopy(queue, queue_family, 0, error);
}

bool DecodedTexture::submitCopyCudaToImage(VkQueue queue, uint32_t queue_family, std::string& error)
{
    return submitCopy(queue, queue_family, 1, error);
}

bool DecodedTexture::submitCopyHipToImage(VkQueue queue, uint32_t queue_family, std::string& error)
{
    return submitCopy(queue, queue_family, 2, error);
}

bool DecodedTexture::flushCopyHostToImage(VkQueue queue, uint32_t queue_family, std::string& error)
{
    return flushCopy(queue, queue_family, 0, error);
}

bool DecodedTexture::flushCopyCudaToImage(VkQueue queue, uint32_t queue_family, std::string& error)
{
    return flushCopy(queue, queue_family, 1, error);
}

bool DecodedTexture::flushCopyHipToImage(VkQueue queue, uint32_t queue_family, std::string& error)
{
    return flushCopy(queue, queue_family, 2, error);
}

void DecodedTexture::recordCopyBufferToImage(VkCommandBuffer cmd, VkBuffer src)
{
    imageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkBufferImageCopy region =
    {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { uint32_t(width), uint32_t(height), 1 },
    };
    vkCmdCopyBufferToImage(cmd, src, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    imageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void DecodedTexture::recordCopyCudaToImage(VkCommandBuffer cmd)
{
    recordCopyBufferToImage(cmd, cuda_buffer);
}

void DecodedTexture::recordTransitionToShaderRead(VkCommandBuffer cmd)
{
    if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        return;
    imageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void DecodedTexture::recordPublishComputeStores(VkCommandBuffer cmd)
{
    if (!image)
        return;

    if (!publish_alloc)
    {
        imageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return;
    }

    imageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region =
    {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { uint32_t(width), uint32_t(height), 1 },
    };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        publish_alloc.buffer, 1, &region);

    VkBufferMemoryBarrier buffer_ready =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = publish_alloc.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    VkImageMemoryBarrier to_dst =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &buffer_ready, 1, &to_dst);

    vkCmdCopyBufferToImage(cmd, publish_alloc.buffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    imageBarrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void DecodedTexture::recordBlitTo(VkCommandBuffer cmd, VkImage dst_image,
    int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1)
{
    if (!image || dst_x1 <= dst_x0 || dst_y1 <= dst_y0)
        return;

    imageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkImageBlit blit =
    {
        .srcSubresource =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .srcOffsets = { { 0, 0, 0 }, { width, height, 1 } },
        .dstSubresource =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .dstOffsets = { { dst_x0, dst_y0, 0 }, { dst_x1, dst_y1, 1 } },
    };

    vkCmdBlitImage(cmd,
        image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit, VK_FILTER_LINEAR);

    imageBarrier(cmd, image, layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

} // namespace gpujpeg

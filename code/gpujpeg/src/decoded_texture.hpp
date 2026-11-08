#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <mango/vulkan/allocator.hpp>
#include <vulkan/vulkan.h>

#if defined(GPUJPEG_HAS_NVJPEG)
struct CUstream_st;
typedef CUstream_st* cudaStream_t;
#endif

namespace mango::vulkan
{
    class VulkanWindow;
    struct VulkanDeviceConfig;
}

namespace gpujpeg
{

// Opaque external-memory device extension (Win32 handle or POSIX fd).
// CUDA (nvJPEG) and HIP (rocJPEG) both import a VkBuffer exported this way.
// Call before creating VulkanWindow. Keeps Win32/FD knowledge out of app main.
void appendCudaVkInteropDeviceExtensions(mango::vulkan::VulkanDeviceConfig& config);

// Intermediate decoded asset: a sampled VkImage (not the swapchain).
// CUDA path writes RGBA into an exported device buffer, then copies into the image.
// Host path uploads via a host-visible staging buffer into the same image.
struct DecodedTexture
{
    int width = 0;
    int height = 0;

    mango::vulkan::Allocator* allocator = nullptr;
    mango::vulkan::ImageAllocation image_alloc {};
    mango::vulkan::BufferAllocation host_alloc {};

    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;

    // CUDA-exported transfer buffer (manual alloc; VMA path has no export hook).
    VkBuffer cuda_buffer = VK_NULL_HANDLE;
    VkDeviceMemory cuda_buffer_memory = VK_NULL_HANDLE;
    void* cuda_rgba = nullptr; // device pointer (imported), tightly packed RGBA8
    size_t rgba_bytes = 0;
    bool cuda_ready = false;

    // HIP-exported transfer buffer. Same layout as the CUDA one: tightly packed RGBA8.
    VkBuffer hip_buffer = VK_NULL_HANDLE;
    VkDeviceMemory hip_buffer_memory = VK_NULL_HANDLE;
    void* hip_rgba = nullptr;
    bool hip_ready = false;

    void* host_mapped = nullptr;

    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;

    // cudaExternalMemory_t / hipExternalMemory_t stored opaquely.
    void* cuda_ext_memory = nullptr;
    void* hip_ext_memory = nullptr;

    bool create(mango::vulkan::VulkanWindow& window, mango::vulkan::Allocator& alloc,
        int w, int h, bool enable_cuda_export, std::string& error, bool enable_hip_export = false);
    void destroy();

    // Host RGBA8 → staging (CPU). Call recordCopyHostToImage after.
    bool writeHostRgba(const std::uint8_t* rgba, size_t bytes, std::string& error);

#if defined(GPUJPEG_HAS_NVJPEG)
    // Device RGBI → CUDA-exported RGBA buffer (NPP). Call recordCopyCudaToImage after sync.
    bool writeDeviceRgbiToCudaRgba(const std::uint8_t* d_rgbi, int rgbi_pitch,
        cudaStream_t stream, std::string& error);
#endif

    void recordCopyHostToImage(VkCommandBuffer cmd);
    void recordCopyCudaToImage(VkCommandBuffer cmd);

    // GPU upload into the intermediate image (no swapchain acquire).
    // submit* returns after the queue accepts the copy; waitCopy blocks until the image is usable.
    // flush* is submit + wait, for a single-frame READY.
    bool submitCopyHostToImage(VkQueue queue, uint32_t queue_family, std::string& error);
    bool submitCopyCudaToImage(VkQueue queue, uint32_t queue_family, std::string& error);
    bool submitCopyHipToImage(VkQueue queue, uint32_t queue_family, std::string& error);
    bool waitCopy(std::string& error);
    bool flushCopyHostToImage(VkQueue queue, uint32_t queue_family, std::string& error);
    bool flushCopyCudaToImage(VkQueue queue, uint32_t queue_family, std::string& error);
    bool flushCopyHipToImage(VkQueue queue, uint32_t queue_family, std::string& error);

    // Blit full image into [dst_x0,dst_y0)→[dst_x1,dst_y1) of an already TRANSFER_DST dst.
    // Leaves this image as SHADER_READ_ONLY; leaves dst as TRANSFER_DST.
    void recordBlitTo(VkCommandBuffer cmd, VkImage dst_image,
        int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1);

    // Resting layout after upload/blit helpers: ready to sample / feed compute.
    void recordTransitionToShaderRead(VkCommandBuffer cmd);

private:
    bool createCudaExportBuffer(std::string& error);
    bool createHipExportBuffer(std::string& error);
    bool ensureUpload(uint32_t queue_family, std::string& error);
    bool submitCopy(VkQueue queue, uint32_t queue_family, int source, std::string& error);
    bool flushCopy(VkQueue queue, uint32_t queue_family, int source, std::string& error);
    void recordCopyBufferToImage(VkCommandBuffer cmd, VkBuffer src);
    void releaseUpload();

    VkCommandPool upload_pool = VK_NULL_HANDLE;
    VkCommandBuffer upload_cmd = VK_NULL_HANDLE;
    VkFence upload_fence = VK_NULL_HANDLE;
    uint32_t upload_family = UINT32_MAX;
    bool upload_pending = false;
};

// Match a HIP device to this Vulkan physical device (UUID, then product-name / arch).
// Sets the current HIP device. out_index may be null.
bool selectHipDeviceForVulkan(VkPhysicalDevice physical, int* out_index, std::string& error);

bool selectCudaDeviceForVulkan(VkPhysicalDevice physical, std::string& error);

} // namespace gpujpeg

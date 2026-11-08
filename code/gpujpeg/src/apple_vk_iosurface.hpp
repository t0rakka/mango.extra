#pragma once

#if defined(GPUJPEG_HAS_APPLE_VT)

#include <string>

#include <mango/vulkan/vulkan.hpp>
#include <vulkan/vulkan.h>

struct __CVBuffer;
typedef struct __CVBuffer* CVPixelBufferRef;

namespace gpujpeg
{

struct AppleImportedImage
{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    int width = 0;
    int height = 0;
    bool imported_iosurface = false;
    CVPixelBufferRef pixel_buffer = nullptr; // retains IOSurface until destroy

    void destroy(VkDevice device);
};

bool appleCreateVkImageFromPixelBuffer(VkPhysicalDevice physical, VkDevice device,
    CVPixelBufferRef pixel, AppleImportedImage& out, std::string& error);

void appleRecordBlitTo(VkCommandBuffer cmd, AppleImportedImage& src, VkImage dst_image,
    VkImageLayout dst_layout, int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1);

bool applePrepareImportedForPresent(VkDevice device, VkQueue queue, uint32_t queue_family,
    AppleImportedImage& image, std::string& error);

void appendAppleVkInteropDeviceExtensions(mango::vulkan::VulkanDeviceConfig& config);

} // namespace gpujpeg

#endif

#include "apple_vk_iosurface.hpp"

#if defined(GPUJPEG_HAS_APPLE_VT)

#include <IOSurface/IOSurface.h>
#include <CoreVideo/CoreVideo.h>

#include <vulkan/vulkan_metal.h>

namespace gpujpeg
{
namespace
{

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

} // namespace

void appendAppleVkInteropDeviceExtensions(mango::vulkan::VulkanDeviceConfig& config)
{
    config.deviceExtensions.push_back(VK_EXT_METAL_OBJECTS_EXTENSION_NAME);
}

void AppleImportedImage::destroy(VkDevice device)
{
    if (device == VK_NULL_HANDLE)
        return;
    if (view != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    if (image != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, image, nullptr);
        image = VK_NULL_HANDLE;
    }
    if (pixel_buffer)
    {
        CVPixelBufferRelease(pixel_buffer);
        pixel_buffer = nullptr;
    }
    layout = VK_IMAGE_LAYOUT_UNDEFINED;
    imported_iosurface = false;
}

bool appleCreateVkImageFromPixelBuffer(VkPhysicalDevice /*physical*/, VkDevice device,
    CVPixelBufferRef pixel, AppleImportedImage& out, std::string& error)
{
    out.destroy(device);

    IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixel);
    if (!surface)
    {
        error = "CVPixelBuffer has no IOSurface backing";
        return false;
    }

    const int w = int(CVPixelBufferGetWidth(pixel));
    const int h = int(CVPixelBufferGetHeight(pixel));

    VkExportMetalObjectCreateInfoEXT export_info =
    {
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT,
        .exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_IOSURFACE_BIT_EXT,
    };
    VkImportMetalIOSurfaceInfoEXT import_info =
    {
        .sType = VK_STRUCTURE_TYPE_IMPORT_METAL_IO_SURFACE_INFO_EXT,
        .ioSurface = surface,
    };
    export_info.pNext = &import_info;

    VkImageCreateInfo image_info =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &export_info,
        .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = { uint32_t(w), uint32_t(h), 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    if (vkCreateImage(device, &image_info, nullptr, &out.image) != VK_SUCCESS)
    {
        error = "vkCreateImage (IOSurface import) failed";
        return false;
    }

    VkImageViewCreateInfo view_info =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = out.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .subresourceRange =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    if (vkCreateImageView(device, &view_info, nullptr, &out.view) != VK_SUCCESS)
    {
        error = "vkCreateImageView failed";
        out.destroy(device);
        return false;
    }

    if (out.pixel_buffer)
        CVPixelBufferRelease(out.pixel_buffer);
    out.pixel_buffer = CVPixelBufferRetain(pixel);

    out.width = w;
    out.height = h;
    out.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    out.imported_iosurface = true;
    return true;
}

bool applePrepareImportedForPresent(VkDevice device, VkQueue queue, uint32_t queue_family,
    AppleImportedImage& image, std::string& error)
{
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkCommandPoolCreateInfo pool_info =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family,
    };
    if (vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS)
    {
        error = "command pool failed";
        return false;
    }

    VkCommandBufferAllocateInfo alloc_info =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(device, &alloc_info, &cmd) != VK_SUCCESS)
    {
        vkDestroyCommandPool(device, pool, nullptr);
        error = "command buffer alloc failed";
        return false;
    }

    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, pool, 1, &cmd);
        vkDestroyCommandPool(device, pool, nullptr);
        error = "fence create failed";
        return false;
    }

    VkCommandBufferBeginInfo begin =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin);
    // IOSurface content is produced outside Vulkan (ImageIO / VT). Treat as foreign fill,
    // then make the image visible to the transfer stage for the swapchain blit.
    imageBarrier(cmd, image.image,
        image.layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        0, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    image.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    if (vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS)
    {
        error = "queue submit failed";
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, pool, 1, &cmd);
        vkDestroyCommandPool(device, pool, nullptr);
        return false;
    }
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyCommandPool(device, pool, nullptr);
    return true;
}

void appleRecordBlitTo(VkCommandBuffer cmd, AppleImportedImage& src, VkImage dst_image,
    VkImageLayout dst_layout, int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1)
{
    if (!src.image || dst_x1 <= dst_x0 || dst_y1 <= dst_y0)
        return;

    if (src.layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        imageBarrier(cmd, src.image,
            src.layout,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        src.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }

    VkImageBlit blit =
    {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .srcOffsets =
        {
            { 0, 0, 0 },
            { src.width, src.height, 1 },
        },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstOffsets =
        {
            { dst_x0, dst_y0, 0 },
            { dst_x1, dst_y1, 0 },
        },
    };
    vkCmdBlitImage(cmd,
        src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst_image, dst_layout,
        1, &blit, VK_FILTER_LINEAR);
}

} // namespace gpujpeg

#endif

// Copyright 2026 Distance Technologies Oy. For internal use only.
//
#pragma once

#include "bench.hpp"

#include <cstdint>

#include <mango/image/jpeg.hpp>
#include <mango/vulkan/vulkan.hpp>

namespace gpujpeg
{

struct DecodedTexture;

// Wide self-synchronising Huffman on Vulkan compute, then the same IDCT path
// as mango-gpu. Chunks are 256 bytes and do not cross a restart segment, so a
// serial scan and a restart scan dispatch the same way.
class MangoGpu2Decoder
{
public:
    mango::image::jpeg::EntropyStream stream;
    mango::u64 extract_us = 0;
    mango::u64 upload_us = 0;
    mango::u64 compile_us = 0;
    mango::u64 compile_last_us = 0;
    mango::u64 decode_us = 0;
    int chunks = 0;
    bool uploaded = false;

    bool extract(mango::ConstMemory jpeg, std::string& error);
    bool upload(mango::vulkan::Allocator& allocator, std::string& error);
    bool decode(mango::vulkan::VulkanWindow& window,
        mango::vulkan::Allocator& allocator, DecodedTexture& texture, std::string& error);
    void release(mango::vulkan::Allocator& allocator);

    MangoGpu2Decoder() = default;
    ~MangoGpu2Decoder() = default;
    MangoGpu2Decoder(const MangoGpu2Decoder&) = delete;
    MangoGpu2Decoder& operator=(const MangoGpu2Decoder&) = delete;

private:
    mango::ConstMemory m_jpeg;
    bool m_pipeline = false;
    VkDeviceSize m_bit_bytes = 0;

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;

    VkShaderModule m_sync_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_sync_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_sync_pipe_layout = VK_NULL_HANDLE;
    VkPipeline m_sync_pipe = VK_NULL_HANDLE;
    VkDescriptorPool m_sync_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_sync_set = VK_NULL_HANDLE;

    VkShaderModule m_prefix_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_prefix_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_prefix_pipe_layout = VK_NULL_HANDLE;
    VkPipeline m_prefix_pipe = VK_NULL_HANDLE;
    VkDescriptorPool m_prefix_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_prefix_set = VK_NULL_HANDLE;

    VkShaderModule m_idct_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_idct_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_idct_pipe_layout = VK_NULL_HANDLE;
    VkPipeline m_idct_pipe = VK_NULL_HANDLE;
    VkDescriptorPool m_idct_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_idct_set = VK_NULL_HANDLE;

    mango::vulkan::BufferAllocation m_bits;
    mango::vulkan::BufferAllocation m_bits_dev;
    mango::vulkan::BufferAllocation m_chunks;
    mango::vulkan::BufferAllocation m_tables;
    mango::vulkan::BufferAllocation m_coeffs;
    mango::vulkan::BufferAllocation m_state;
    mango::vulkan::BufferAllocation m_entry;
    mango::vulkan::BufferAllocation m_cps;
    mango::vulkan::BufferAllocation m_segs;
    mango::vulkan::BufferAllocation m_qt;
};
}

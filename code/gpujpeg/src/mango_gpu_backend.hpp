// Copyright 2026 Distance Technologies Oy. For internal use only.
//
#pragma once

#include "bench.hpp"

#include <mango/image/jpeg.hpp>
#include <mango/vulkan/vulkan.hpp>

namespace gpujpeg
{

struct DecodedTexture;

// Full-GPU decode ingest (Mango1 offsets, DRI restart scan, or one serial scan).
// Step 1: entropy views from jpeg::JpegDecoder, copied segment-by-segment into
// one host-visible VkBuffer. No spectral coefficient buffer, no second host copy.
// Huffman + IDCT compute comes after this buffer is in place.
//
// Dispatch shape (CPU decides, GPU runs): one segment is rows_per_segment MCU
// rows. Huffman inside a segment is serial (DC predictors chain across those
// rows). IDCT is one work item per MCU once the segment's blocks exist.
class MangoGpuDecoder
{
public:
    mango::image::jpeg::EntropyStream stream;
    mango::vulkan::BufferAllocation bits;
    mango::u64 extract_us = 0;
    mango::u64 upload_us = 0;
    mango::u64 compile_us = 0;
    mango::u64 huffman_us = 0;
    mango::u64 resolve_us = 0; // IDCT + YCbCr → texture
    mango::u64 compile_last_us = 0; // non-zero only on the call that compiled
    int sync_rounds = -1; // longest self-sync wavefront inside a segment
    int sync_chunks = 0;
    bool uploaded = false;

    bool extract(mango::ConstMemory jpeg, std::string& error);
    bool upload(mango::vulkan::Allocator& allocator, std::string& error);

    // Huffman into spectral coefficients, then IDCT and YCbCr into texture.
    // huffman_us and resolve_us are the two submits. No host readback.
    bool decodeBlocks(mango::vulkan::VulkanWindow& window,
        mango::vulkan::Allocator& allocator, DecodedTexture& texture, std::string& error);

    void release(mango::vulkan::Allocator& allocator);

    MangoGpuDecoder() = default;
    ~MangoGpuDecoder() = default;
    MangoGpuDecoder(const MangoGpuDecoder&) = delete;
    MangoGpuDecoder& operator=(const MangoGpuDecoder&) = delete;

private:
    mango::ConstMemory m_jpeg;
    bool m_pipeline = false;

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    VkShaderModule m_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipe_layout = VK_NULL_HANDLE;
    VkPipeline m_pipe = VK_NULL_HANDLE;
    VkDescriptorPool m_desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    VkShaderModule m_idct_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_idct_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_idct_pipe_layout = VK_NULL_HANDLE;
    VkPipeline m_idct_pipe = VK_NULL_HANDLE;
    VkDescriptorPool m_idct_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_idct_set = VK_NULL_HANDLE;

    mango::vulkan::BufferAllocation m_segs;
    mango::vulkan::BufferAllocation m_tables;
    mango::vulkan::BufferAllocation m_coeffs;
    mango::vulkan::BufferAllocation m_qt;
    mango::vulkan::BufferAllocation m_stats;
    mango::vulkan::BufferAllocation m_stats_rb;
};

} // namespace gpujpeg

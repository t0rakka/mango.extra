// Copyright 2026 Distance Technologies Oy. For internal use only.
//
#pragma once

#include "bench.hpp"

#include <cstdint>
#include <vector>

#include <mango/image/jpeg.hpp>
#include <mango/vulkan/vulkan.hpp>

namespace gpujpeg
{

struct DecodedTexture;

// Wide self-synchronising Huffman on Vulkan compute, then IDCT + YCbCr into
// the VkImage. The host unstuffs the scan into big-endian words; chunks are
// 2048 bits and never cross a restart segment, so serial and restart scans
// dispatch the same way.
class MangoGpu2Decoder
{
public:
    mango::image::jpeg::EntropyStream stream;
    mango::u64 extract_us = 0;
    mango::u64 upload_us = 0;
    mango::u64 compile_us = 0;
    mango::u64 compile_last_us = 0;
    mango::u64 decode_us = 0;   // GPU time (timestamps) when available, else submit wall time
    mango::u64 huffman_us = 0;  // bit copy + sync + prefix + write + DC scan
    mango::u64 pixel_us = 0;    // IDCT + colour into the VkImage
    mango::u64 wall_us = 0;     // submit + fence wait
    mango::u64 publish_us = 0;  // DecodedTexture publish copy after the IDCT
    // copy+clear, sync passes, prefix, write, DC scan, IDCT
    mango::u64 stage_us[6] {};
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
    bool createPipelines(mango::vulkan::VulkanWindow& window, DecodedTexture& texture, std::string& error);

    mango::ConstMemory m_jpeg;
    bool m_pipeline = false;
    bool m_static_copied = false;
    bool m_linear = false;  // IDCT writes the texture's publish buffer, then one copy into the image
    int m_nblk = 0;
    int m_nsegs = 0;
    std::uint32_t m_nwords = 0;
    std::uint32_t m_dc_pack = 0;
    std::uint32_t m_ac_pack = 0;
    std::uint32_t m_comp_pack = 0;
    std::uint32_t m_seg_start_words = 0;

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    VkQueryPool m_queries = VK_NULL_HANDLE;
    float m_tick_ns = 0.0f;

    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipe_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkShaderModule m_shaders[6] {};
    VkPipeline m_pipes[6] {};

    // host staging
    mango::vulkan::BufferAllocation m_bits;
    mango::vulkan::BufferAllocation m_chunks_host;
    mango::vulkan::BufferAllocation m_segs_host;
    mango::vulkan::BufferAllocation m_tables_host;
    mango::vulkan::BufferAllocation m_qt_host;
    mango::vulkan::BufferAllocation m_seg_start_host;

    // device
    mango::vulkan::BufferAllocation m_bits_dev;
    mango::vulkan::BufferAllocation m_chunks;
    mango::vulkan::BufferAllocation m_segs;
    mango::vulkan::BufferAllocation m_tables;
    mango::vulkan::BufferAllocation m_qt;
    mango::vulkan::BufferAllocation m_st_in;
    mango::vulkan::BufferAllocation m_st_out;
    mango::vulkan::BufferAllocation m_cnt;
    mango::vulkan::BufferAllocation m_cps;
    mango::vulkan::BufferAllocation m_cpc;
    mango::vulkan::BufferAllocation m_first;
    mango::vulkan::BufferAllocation m_coef;
    mango::vulkan::BufferAllocation m_dcd;
    mango::vulkan::BufferAllocation m_flags;
    mango::vulkan::BufferAllocation m_seg_start;
    mango::vulkan::BufferAllocation m_agg;
};

} // namespace gpujpeg

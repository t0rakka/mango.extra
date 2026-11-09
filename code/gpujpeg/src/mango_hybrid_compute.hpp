#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <mango/core/memory.hpp>
#include <mango/image/jpeg.hpp>
#include <mango/vulkan/allocator.hpp>
#include <vulkan/vulkan.h>

#include "decoded_texture.hpp"

namespace mango::vulkan { class VulkanWindow; }

namespace gpujpeg
{

struct MangoHybridResult
{
    bool ok = false;
    std::string error;

    int width = 0;
    int height = 0;
    int bands = 0;

    mango::u64 huffman_us = 0;
    mango::u64 ready_us = 0;
    mango::u64 gpu_wait_us = 0;
};

// Huffman workers write cached host memory only. A finished segment is private:
// the worker sets an atomic and returns. The graphics thread then streaming-copies
// that range into the upload buffer and submits its dispatch. No worker waits on
// the GPU, and no worker stores into host-visible / ReBAR memory.
class MangoHybridCompute
{
public:
    MangoHybridCompute() = default;
    ~MangoHybridCompute();

    MangoHybridCompute(const MangoHybridCompute&) = delete;
    MangoHybridCompute& operator=(const MangoHybridCompute&) = delete;

    bool create(mango::vulkan::VulkanWindow& window,
        mango::vulkan::Allocator& allocator,
        DecodedTexture& texture,
        const mango::image::jpeg::JpegInfo& info,
        std::string& error);

    void destroy();
    bool created() const { return m_device != VK_NULL_HANDLE; }

    bool startFrame(const mango::image::jpeg::JpegInfo& info, int hardware_threads, std::string& error);
    bool handoffBand(const mango::image::jpeg::CoeffView& view, std::string& error);
    mango::s16* coeffStorage();
    size_t coeffSamples() const { return m_frame_samples; }
    bool finishFrame(std::string& error);

    mango::u64 gpuWaitUs() const { return m_gpu_wait_us.load(); }
    void resetGpuWait() { m_gpu_wait_us = 0; }

private:
    struct SegmentPlan
    {
        int mcu_y0 = 0;
        int mcu_y1 = 0;
        size_t coeff_offset = 0; // s16 index into frame staging
        size_t samples = 0;
    };

    struct SegmentGpu
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
    };

    bool createPipeline(std::string& error);
    bool allocFrameResources(std::string& error);
    void freeFrameResources();
    void graphicsThreadMain();
    bool recordPrepare(std::string& error);
    bool recordSegmentCommands(std::string& error);
    bool recordLayoutToSampleReady(std::string& error);
    void addGpuWaitUs(mango::u64 us);
    int findSegment(int mcu_y0) const;

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_family = 0;
    mango::vulkan::Allocator* m_allocator = nullptr;
    DecodedTexture* m_texture = nullptr;

    mango::image::jpeg::JpegInfo m_info {};
    int m_natural_qt[4][64] {};

    VkShaderModule m_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_desc_pool = VK_NULL_HANDLE;
    VkCommandPool m_cmd_pool = VK_NULL_HANDLE;

    mango::vulkan::BufferAllocation m_qt_buffer {};
    mango::vulkan::BufferAllocation m_frame_staging {};
    // Cached DRAM. Huffman memset + scalar stores stay here. The upload mapping
    // is write-combined or device-local (ReBAR); writing it from the entropy
    // loop is several times slower than a full CPU decode.
    std::vector<mango::s16> m_host_coeffs;
    size_t m_frame_samples = 0;

    std::vector<SegmentPlan> m_plan;
    std::unique_ptr<std::atomic<bool>[]> m_segment_ready;
    size_t m_segment_ready_count = 0;
    std::vector<SegmentGpu> m_segment_gpu;
    std::atomic<int> m_handoffs { 0 };

    VkCommandBuffer m_prepare_cmd = VK_NULL_HANDLE;
    VkFence m_prepare_fence = VK_NULL_HANDLE;
    VkCommandBuffer m_layout_cmd = VK_NULL_HANDLE;
    VkFence m_layout_fence = VK_NULL_HANDLE;

    std::thread m_graphics_thread;
    std::mutex m_gpu_mutex;
    std::condition_variable m_gpu_cv;
    std::atomic<bool> m_huffman_done { false };
    std::atomic<bool> m_gpu_done { false };
    std::atomic<bool> m_gpu_error { false };
    std::string m_gpu_error_msg;
    bool m_frame_gpu_started = false;

    std::atomic<mango::u64> m_gpu_wait_us { 0 };
};

MangoHybridResult runMangoHybridCompute(mango::ConstMemory jpeg,
    mango::vulkan::VulkanWindow& window,
    mango::vulkan::Allocator& allocator,
    DecodedTexture& texture,
    MangoHybridCompute* reused = nullptr);

std::size_t hybridSegmentRowsPerTask(int ymcu, int hardware_threads);

} // namespace gpujpeg

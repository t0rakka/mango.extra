#include <mango/mango.hpp>
#include <mango/vulkan/vulkan.hpp>

#include "backends.hpp"
#include "bench.hpp"
#include "decoded_texture.hpp"
#include "mango_gpu_backend.hpp"
#include "mango_hybrid_backend.hpp"
#include "mango_hybrid_compute.hpp"
#include "nvjpeg_backend.hpp"
#include "roc_backend.hpp"
#if defined(GPUJPEG_HAS_APPLE_VT)
#include "apple_vt_backend.hpp"
#include "apple_imageio_backend.hpp"
#include "apple_vk_iosurface.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#if defined(GPUJPEG_HAS_NVJPEG)
#include <cuda_runtime.h>
#endif

using namespace mango;
using namespace mango::filesystem;
using namespace mango::image;
using namespace mango::math;
using namespace mango::vulkan;
using namespace gpujpeg;

namespace
{

enum class Backend
{
    Mango,
    Hybrid,  // mango Spectral Huffman + VK compute
    Gpu,     // mango-gpu: Mango1 entropy -> VkBuffer (Huffman compute later)
    NvJpeg,
    Roc,
    AppleVt,
    AppleImageIo,
    Both,    // mango + nvjpeg
    All,     // every built source
};

enum class NvJpegModeSelect
{
    Hybrid,
    Gpu,
    Both,
};

enum class RocModeSelect
{
    Hardware,
    Hybrid,
    Both,
};

struct Args
{
    std::string filename;
    Backend backend = Backend::All;
    NvJpegModeSelect nvjpeg_mode = NvJpegModeSelect::Both;
    RocModeSelect roc_mode = RocModeSelect::Both;
    int iterations = 1;
    int warmup = 1;
    int inflight = 1;
    int pipeline_frames = 0;
    bool validate = false;
    bool keep_open = false;
    bool hybrid_floor = false; // Huffman-only floor (skip VK compute)
    std::string error;
};

void configureParser(const CommandLine& commands, Args& args)
{
    commands.usage("[options] <file.jpg>");

    commands.option("--backend",
        "mango | hybrid | gpu | nvjpeg | roc | apple-vt | apple-imageio | both | all (default all)\n"
        "         hybrid = Huffman+VK compute; gpu = Mango1/RST/serial entropy -> VkImage;\n"
        "         roc = VCN JPEG; apple-vt = VideoToolbox;\n"
        "         apple-imageio = ImageIO still JPEG → BGRA IOSurface;\n"
        "         both = mango+nvjpeg; all = every built source",
        [&](std::string_view value)
        {
            if (value == "mango")
                args.backend = Backend::Mango;
            else if (value == "hybrid")
                args.backend = Backend::Hybrid;
            else if (value == "gpu" || value == "mango-gpu")
                args.backend = Backend::Gpu;
            else if (value == "nvjpeg")
                args.backend = Backend::NvJpeg;
            else if (value == "roc")
                args.backend = Backend::Roc;
            else if (value == "apple-vt" || value == "apple" || value == "vt")
                args.backend = Backend::AppleVt;
            else if (value == "apple-imageio" || value == "imageio")
                args.backend = Backend::AppleImageIo;
            else if (value == "both")
                args.backend = Backend::Both;
            else if (value == "all")
                args.backend = Backend::All;
            else
                args.error = fmt::format(
                    "Invalid --backend \"{}\" (mango|hybrid|gpu|nvjpeg|roc|apple-vt|apple-imageio|both|all)", value);
        });

    commands.option("--nvjpeg-mode", "hybrid | gpu | both (default both). "
        "hybrid=CPU Huffman, gpu=GPU-assisted Huffman",
        [&](std::string_view value)
        {
            if (value == "hybrid")
                args.nvjpeg_mode = NvJpegModeSelect::Hybrid;
            else if (value == "gpu")
                args.nvjpeg_mode = NvJpegModeSelect::Gpu;
            else if (value == "both")
                args.nvjpeg_mode = NvJpegModeSelect::Both;
            else
                args.error = fmt::format("Invalid --nvjpeg-mode \"{}\" (hybrid|gpu|both)", value);
        });

    commands.option("--roc-mode", "hardware | hybrid | both (default both). "
        "hardware = VCN JPEG ASIC, hybrid = ROCJPEG_BACKEND_HYBRID",
        [&](std::string_view value)
        {
            if (value == "hardware")
                args.roc_mode = RocModeSelect::Hardware;
            else if (value == "hybrid")
                args.roc_mode = RocModeSelect::Hybrid;
            else if (value == "both")
                args.roc_mode = RocModeSelect::Both;
            else
                args.error = fmt::format("Invalid --roc-mode \"{}\" (hardware|hybrid|both)", value);
        });

    commands.optionInt("--iterations", "timed present-path iterations after warmup (default 1)",
        [&](int value) { args.iterations = std::max(1, value); });

    commands.optionInt("--warmup", "discarded warm-up decodes before timing (default 1)",
        [&](int value) { args.warmup = std::max(0, value); });

    commands.optionInt("--inflight", "overlapping nvJPEG decodes / CUDA streams (default 1)",
        [&](int value) { args.inflight = std::max(1, value); });

    commands.optionInt("--pipeline", "overlapped READY throughput frames (0=off); depth=--inflight",
        [&](int value) { args.pipeline_frames = std::max(0, value); });

    commands.flag("--validate", "enable Vulkan validation layer",
        [&]() { args.validate = true; });

    commands.flag("--keep-open", "legacy no-op (window always stays open; ESC quits)",
        [&]() { args.keep_open = true; });

    commands.flag("--hybrid-floor", "hybrid: Huffman only (no VK compute) — measure floor",
        [&]() { args.hybrid_floor = true; });

    commands.positional([&](std::string_view token) { args.filename = token; });
}

// GPUJPEG_CHECK=1: read the hybrid VkImage back and compare to a CPU decode.
// Returns the max abs channel delta, or -1 on failure.
int checkHybridImage(DecodedTexture& texture, ConstMemory jpeg,
    VkQueue queue, uint32_t queue_family, std::string& error)
{
    const Format format(32, Format::UNORM, Format::RGBA, 8, 8, 8, 8);
    Bitmap cpu(jpeg, ".jpg", format);
    if (!cpu.image || cpu.width != texture.width || cpu.height != texture.height)
    {
        error = "CPU reference decode failed";
        return -1;
    }
    if (!texture.device || !texture.image || !texture.host_alloc)
    {
        error = "readback staging missing";
        return -1;
    }

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family,
    };
    if (vkCreateCommandPool(texture.device, &pool_info, nullptr, &pool) != VK_SUCCESS)
    {
        error = "readback pool failed";
        return -1;
    }
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc_info =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    vkAllocateCommandBuffers(texture.device, &alloc_info, &cmd);
    VkCommandBufferBeginInfo begin =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin);

    VkImageMemoryBarrier to_src =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = texture.layout,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &to_src);

    VkBufferImageCopy region =
    {
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = { uint32_t(texture.width), uint32_t(texture.height), 1 },
    };
    vkCmdCopyImageToBuffer(cmd, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        texture.host_alloc.buffer, 1, &region);

    VkMemoryBarrier to_host =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &to_host, 0, nullptr, 0, nullptr);

    VkImageMemoryBarrier to_read = to_src;
    to_read.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &to_read);
    vkEndCommandBuffer(cmd);

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(texture.device, &fence_info, nullptr, &fence);
    VkSubmitInfo submit =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    const VkResult sr = vkQueueSubmit(queue, 1, &submit, fence);
    if (sr == VK_SUCCESS)
        vkWaitForFences(texture.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(texture.device, fence, nullptr);
    vkDestroyCommandPool(texture.device, pool, nullptr);
    texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (sr != VK_SUCCESS)
    {
        error = "readback submit failed";
        return -1;
    }

    const auto* got = static_cast<const u8*>(texture.host_mapped);
    int max_delta = 0;
    int bad = 0;
    int first_x = -1;
    int first_y = -1;
    int last_y = -1;
    int bad_rows = 0;
    int row_bad = 0;
    for (int y = 0; y < texture.height; ++y)
    {
        const u8* row = got + size_t(y) * size_t(texture.width) * 4;
        const u8* ref = cpu.address<u8>(0, y);
        const size_t n = size_t(texture.width) * 4;
        row_bad = 0;
        for (size_t i = 0; i < n; ++i)
        {
            const int d = std::abs(int(row[i]) - int(ref[i]));
            if (d > max_delta)
                max_delta = d;
            if (d > 4)
            {
                ++bad;
                ++row_bad;
                if (first_x < 0)
                {
                    first_x = int(i / 4);
                    first_y = y;
                }
                last_y = y;
            }
        }
        if (row_bad)
            ++bad_rows;
    }
    if (bad > 0)
        printLine("  check mismatch: {} samples, {} rows, first {},{} last y {}",
            bad, bad_rows, first_x, first_y, last_y);
    return max_delta;
}

// SIMD + multithread decode into caller memory. Pass cached memory;
// the Vulkan staging buffer is write-combined and is a bad decode target.
bool decodeMangoTo(ConstMemory jpeg, void* dest, int width, int height,
    u64& decode_us, std::string& error)
{
    decode_us = 0;
    if (!dest || width <= 0 || height <= 0)
    {
        error = "staging missing";
        return false;
    }

    const Format format(32, Format::UNORM, Format::RGBA, 8, 8, 8, 8);
    Surface surface(width, height, format, size_t(width) * 4, dest);

    ImageDecoder decoder(jpeg, ".jpg");
    const ImageHeader header = decoder.header();
    if (header.width != width || header.height != height)
    {
        error = fmt::format("JPEG size {}x{} does not match texture {}x{}",
            header.width, header.height, width, height);
        return false;
    }

    ImageDecodeOptions options;
    options.simd = true;
    options.multithread = true;

    const u64 t0 = nowUs();
    const ImageDecodeStatus status = decoder.decode(surface, options);
    decode_us = nowUs() - t0;
    if (!status)
    {
        error = status.info.empty() ? "mango JPEG decode failed" : status.info;
        return false;
    }
    return true;
}

void fitCell(int width, int height, int& out_w, int& out_h)
{
    const int max_dim = 640;
    const int max_side = std::max(width, height);
    if (max_side <= max_dim)
    {
        out_w = width;
        out_h = height;
        return;
    }
    const float scale = float(max_dim) / float(max_side);
    out_w = std::max(1, int(float(width) * scale + 0.5f));
    out_h = std::max(1, int(float(height) * scale + 0.5f));
}

void gridDims(int count, int& cols, int& rows)
{
    // 1→1x1, 2→2x1, 3→2x2, 4→2x2, 5→3x2, ...
    count = std::max(1, count);
    cols = int(std::ceil(std::sqrt(double(count))));
    rows = (count + cols - 1) / cols;
}

struct Job
{
    std::string name;
    bool use_nvjpeg = false;
    bool use_hybrid = false;
    bool use_gpu = false;
    bool use_roc = false;
    bool use_apple_vt = false;
    bool use_apple_imageio = false;
    bool hybrid_floor = false; // Huffman-only (skip VK compute)
    NvJpegMode nv_mode = NvJpegMode::Hybrid;
    RocJpegMode roc_mode = RocJpegMode::Hardware;
    NvJpegDecoder* decoder = nullptr;
    RocJpegDecoder* roc = nullptr;
    MangoGpuDecoder* gpu = nullptr;
};

struct BackendSlot
{
    std::string name;
    DecodedTexture texture;
    std::unique_ptr<MangoHybridCompute> hybrid_compute;
#if defined(GPUJPEG_HAS_APPLE_VT)
    AppleImportedImage apple_image;
#endif
    bool ready = false;
};

class BenchWindow : public VulkanWindow
{
public:
    ConstMemory jpeg;
    std::vector<Job> jobs;
    int content_w = 0;
    int content_h = 0;
    int iterations = 1;
    int pipeline_frames = 0;
    int inflight = 1;
    int warmup = 1;
    bool want_cuda_texture = false;
    bool want_hip_texture = false;

    std::unique_ptr<Allocator> allocator;
    std::vector<BackendSlot> slots; // one retained VkImage per backend (grid tiles)
    std::vector<DecodedTexture> ready_ring; // inflight VkImages for READY pipeline
    std::vector<u8> host_rgba; // cached RGBA for the CPU decode; staging is write-combined
    std::vector<PipelineResult> results;
    bool first_frame_done = false;
    int grid_cols = 1;
    int grid_rows = 1;

    BenchWindow(VulkanContext& context, int win_w, int win_h,
        int image_w, int image_h, bool cuda_export, const VulkanDeviceConfig* config)
        : VulkanWindow(context, win_w, win_h, 0, config)
        , content_w(image_w)
        , content_h(image_h)
        , want_cuda_texture(cuda_export)
    {
    }

    ~BenchWindow() override
    {
        if (isDeviceReady())
            vkDeviceWaitIdle(device());
        for (DecodedTexture& t : ready_ring)
            t.destroy();
        ready_ring.clear();
        for (BackendSlot& slot : slots)
        {
            slot.hybrid_compute.reset();
#if defined(GPUJPEG_HAS_APPLE_VT)
            slot.apple_image.destroy(device());
            appleVtDiscardDecodeCache();
#endif
            slot.texture.destroy();
        }
        slots.clear();
        allocator.reset();
    }

    void onDeviceReady() override
    {
        if (!allocator)
            allocator = std::make_unique<Allocator>(
                instance(), physicalDevice(), device(), VK_API_VERSION_1_3);

        if (!slots.empty())
            return;

#if defined(GPUJPEG_HAS_SOURCE_ROC)
        // rocJPEG handle is bound to the HIP peer of this VkPhysicalDevice.
        for (Job& job : jobs)
        {
            if (!job.use_roc || !job.roc || job.roc->ok)
                continue;
            std::string error;
            if (!job.roc->init(job.roc_mode, content_w, content_h, inflight, physicalDevice(), error))
            {
                printLine("roc [{}] init FAILED: {}", rocJpegModeName(job.roc_mode), error);
                continue;
            }
            printLine("roc [{}] init: {:.2f} ms  {} ({}) inflight={}",
                rocJpegModeName(job.roc_mode), job.roc->init_us / 1000.0,
                job.roc->device_name, job.roc->arch_name, job.roc->inflight);
            if (!job.roc->warmup(jpeg, warmup, error))
                printLine("roc [{}] warmup FAILED: {}", rocJpegModeName(job.roc_mode), error);
            else if (warmup > 0)
            {
                printLine("roc [{}] warmup: {:.2f} ms ({} rounds)",
                    rocJpegModeName(job.roc_mode), job.roc->warmup_us / 1000.0, warmup);
            }
        }
#endif

        slots.resize(jobs.size());
        std::string error;
        for (size_t i = 0; i < jobs.size(); ++i)
        {
            slots[i].name = jobs[i].name;
            if (jobs[i].use_hybrid && jobs[i].hybrid_floor)
            {
                printLine("DecodedTexture [{}]: skipped (huffman-only floor)", jobs[i].name);
                continue;
            }
            const bool cuda_export = want_cuda_texture && jobs[i].use_nvjpeg;
            const bool hip_export = want_hip_texture && jobs[i].use_roc && jobs[i].roc && jobs[i].roc->ok;
            if (!slots[i].texture.create(*this, *allocator, content_w, content_h,
                    cuda_export, error, hip_export))
            {
                printLine("DecodedTexture [{}] create FAILED: {}", jobs[i].name, error);
                slots[i].texture.destroy();
            }
            else
            {
                printLine("DecodedTexture [{}]: {}x{} VkImage (cuda_export={} hip_export={})",
                    jobs[i].name, content_w, content_h,
                    slots[i].texture.cuda_ready ? "yes" : "no",
                    slots[i].texture.hip_ready ? "yes" : "no");
            }
        }

        gridDims(int(slots.size()), grid_cols, grid_rows);
    }

    void onKeyPress(Keycode key, u32 mask) override
    {
        MANGO_UNREFERENCED(mask);
        if (key == KEYCODE_ESC)
            requestQuit();
    }

    void onFrame(const FrameInfo& info) override
    {
        MANGO_UNREFERENCED(info);

        if (!first_frame_done)
        {
            first_frame_done = true;
            runJobs();
            setTitle(fmt::format("gpujpeg {}x{} grid (ESC quit)", grid_cols, grid_rows));
            presentGrid();
            return;
        }

        presentGrid();
    }

    static void recordSwapchainToTransferDst(VkCommandBuffer cmd, VkImage swapchain)
    {
        VkImageMemoryBarrier barrier =
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain,
            .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        const VkClearColorValue black = {{ 0.05f, 0.05f, 0.05f, 1.0f }};
        VkImageSubresourceRange range =
        {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        vkCmdClearColorImage(cmd, swapchain, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }

    static void recordSwapchainToPresent(VkCommandBuffer cmd, VkImage swapchain)
    {
        VkImageMemoryBarrier barrier =
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain,
            .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    bool presentGrid()
    {
        auto frame = beginDraw();
        if (!frame)
            return false;

        VkCommandBuffer cmd = commandBuffer(frame.imageIndex());
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo =
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &beginInfo);

        const VkExtent2D extent = swapchain().getExtent();
        recordSwapchainToTransferDst(cmd, frame.image());

        const int cell_w = std::max(1, int(extent.width) / std::max(1, grid_cols));
        const int cell_h = std::max(1, int(extent.height) / std::max(1, grid_rows));

        for (size_t i = 0; i < slots.size(); ++i)
        {
            if (!slots[i].ready)
                continue;

            const int col = int(i) % grid_cols; // western reading order: row-major
            const int row = int(i) / grid_cols;
            const int x0 = col * cell_w;
            const int y0 = row * cell_h;
            const int x1 = (col + 1 == grid_cols) ? int(extent.width) : x0 + cell_w;
            const int y1 = (row + 1 == grid_rows) ? int(extent.height) : y0 + cell_h;

#if defined(GPUJPEG_HAS_APPLE_VT)
            const bool apple_slot = i < jobs.size()
                && (jobs[i].use_apple_vt || jobs[i].use_apple_imageio);
            if (apple_slot && slots[i].apple_image.image)
            {
                appleRecordBlitTo(cmd, slots[i].apple_image, frame.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, x0, y0, x1, y1);
                continue;
            }
#endif
            if (!slots[i].texture.image)
                continue;
            slots[i].texture.recordBlitTo(cmd, frame.image(), x0, y0, x1, y1);
        }

        recordSwapchainToPresent(cmd, frame.image());
        vkEndCommandBuffer(cmd);

        frame.submitAndPresent(graphicsQueue(), cmd);
        vkQueueWaitIdle(graphicsQueue());
        return true;
    }

    // Timed present of a single backend tile (fullscreen) for present_us.
    bool presentSlotFullscreen(DecodedTexture& tex)
    {
        auto frame = beginDraw();
        if (!frame)
            return false;

        VkCommandBuffer cmd = commandBuffer(frame.imageIndex());
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo =
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &beginInfo);

        const VkExtent2D extent = swapchain().getExtent();
        recordSwapchainToTransferDst(cmd, frame.image());
        tex.recordBlitTo(cmd, frame.image(), 0, 0, int32_t(extent.width), int32_t(extent.height));
        recordSwapchainToPresent(cmd, frame.image());
        vkEndCommandBuffer(cmd);

        frame.submitAndPresent(graphicsQueue(), cmd);
        vkQueueWaitIdle(graphicsQueue());
        return true;
    }

#if defined(GPUJPEG_HAS_APPLE_VT)
    bool presentSlotApple(AppleImportedImage& src)
    {
        auto frame = beginDraw();
        if (!frame)
            return false;

        VkCommandBuffer cmd = commandBuffer(frame.imageIndex());
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo =
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &beginInfo);

        const VkExtent2D extent = swapchain().getExtent();
        recordSwapchainToTransferDst(cmd, frame.image());
        appleRecordBlitTo(cmd, src, frame.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, 0, int32_t(extent.width), int32_t(extent.height));
        recordSwapchainToPresent(cmd, frame.image());
        vkEndCommandBuffer(cmd);

        frame.submitAndPresent(graphicsQueue(), cmd);
        vkQueueWaitIdle(graphicsQueue());
        return true;
    }
#endif

    void runJobs()
    {
        if (slots.empty())
        {
            printLine("No DecodedTexture slots — aborting jobs.");
            return;
        }

        for (size_t ji = 0; ji < jobs.size(); ++ji)
        {
            const Job& job = jobs[ji];
            BackendSlot& slot = slots[ji];

#if defined(GPUJPEG_HAS_NVJPEG)
            if (job.use_nvjpeg && job.decoder && pipeline_frames > 0)
            {
                auto pipe = runReadyPipeline(*job.decoder, job.name);
                printPipeline(pipe);
                results.push_back(std::move(pipe));
            }
#endif

#if defined(GPUJPEG_HAS_SOURCE_ROC)
            if (job.use_roc && job.roc && job.roc->ok && pipeline_frames > 0)
            {
                auto pipe = runRocReadyPipeline(*job.roc, slot.texture, job.name);
                printPipeline(pipe);
                results.push_back(std::move(pipe));
            }
#endif

            if (!job.use_hybrid && !job.use_nvjpeg && !job.use_roc
                && !job.use_apple_vt && !job.use_apple_imageio && pipeline_frames > 0)
            {
                auto pipe = runMangoReadyPipeline(job.name);
                printPipeline(pipe);
                results.push_back(std::move(pipe));
            }

#if defined(GPUJPEG_HAS_SOURCE_MANGO_HYBRID)
            // Compile/create hybrid compute once before timed iterations.
            if (job.use_hybrid && !job.hybrid_floor && slot.texture.image)
            {
                if (!slot.hybrid_compute)
                    slot.hybrid_compute = std::make_unique<MangoHybridCompute>();
                if (!slot.hybrid_compute->created())
                {
                    image::jpeg::JpegDecoder warm(jpeg);
                    std::string werr;
                    if (!slot.hybrid_compute->create(*this, *allocator, slot.texture,
                            warm.info(), werr))
                        printLine("hybrid compute warm FAILED: {}", werr);
                    else
                        printLine("hybrid compute: pipeline ready");
                }
            }
#endif

#if defined(GPUJPEG_HAS_APPLE_VT)
            if (job.use_apple_vt)
            {
                std::string av_err;
                if (!appleVtValidateJpeg(jpeg, av_err))
                {
                    printLine("apple-vt validate FAILED: {}", av_err);
                    continue;
                }
                if (warmup > 0)
                {
                    u64 wus = 0;
                    if (!appleVtWarmup(jpeg, content_w, content_h, warmup, wus))
                        printLine("apple-vt warmup FAILED");
                    else
                        printLine("apple-vt warmup: {:.2f} ms ({} rounds, incl. session)",
                            wus / 1000.0, warmup);
                }
            }
            if (job.use_apple_imageio && warmup > 0)
            {
                u64 wus = 0;
                if (!appleImageIoWarmup(jpeg, warmup, wus))
                    printLine("apple-imageio warmup FAILED");
                else
                    printLine("apple-imageio warmup: {:.2f} ms ({} rounds)",
                        wus / 1000.0, warmup);
            }
#endif

            for (int iter = 0; iter < iterations; ++iter)
            {
                auto r = runOne(job, slot, iter);
                printPipeline(r);
                results.push_back(std::move(r));
            }
        }
    }

#if defined(GPUJPEG_HAS_NVJPEG)
    PipelineResult runReadyPipeline(NvJpegDecoder& dec, const std::string& name)
    {
        PipelineResult pipe;
        pipe.backend = name + " [ready-pipe]";
        pipe.device_only = true;
        pipe.pipeline_overlapped = true;

        std::string error;
        const int depth = std::max(1, std::min(inflight, dec.inflight));
        if (!ensureReadyRing(depth, error))
        {
            pipe.error = error;
            return pipe;
        }

        std::vector<bool> busy(size_t(depth), false);

        auto finishSlot = [&](int si) -> bool
        {
            if (!busy[size_t(si)])
                return true;
            // Stream already has decode + NPP queued; sync then promote to VkImage.
            if (!dec.syncSlot(si, error))
                return false;
            if (!ready_ring[size_t(si)].flushCopyCudaToImage(
                    graphicsQueue(), graphicsQueueFamilyIndex(), error))
                return false;
            busy[size_t(si)] = false;
            return true;
        };

        const u64 t0 = nowUs();
        for (int i = 0; i < pipeline_frames; ++i)
        {
            const int si = i % depth;
            if (!finishSlot(si))
            {
                pipe.error = error;
                return pipe;
            }
            if (!dec.decodeLaunch(jpeg, si, error))
            {
                pipe.error = error;
                return pipe;
            }
            auto* stream = static_cast<cudaStream_t>(dec.decodeStream(si));
            // NPP on same stream — overlaps with other slots' decode.
            if (!ready_ring[size_t(si)].writeDeviceRgbiToCudaRgba(
                    dec.deviceRgbi(si), dec.rgbiPitch(), stream, error))
            {
                pipe.error = error;
                return pipe;
            }
            busy[size_t(si)] = true;
        }
        for (int si = 0; si < depth; ++si)
        {
            if (!finishSlot(si))
            {
                pipe.error = error;
                return pipe;
            }
        }
        const u64 total_us = nowUs() - t0;

        pipe.ok = true;
        pipe.pipeline_count = pipeline_frames;
        pipe.pipeline_avg_us = total_us / u64(pipeline_frames);
        pipe.ready_us = pipe.pipeline_avg_us;
        pipe.decode_us = pipe.pipeline_avg_us; // approx; READY is the headline
        pipe.e2e_us = total_us;
        pipe.image.width = content_w;
        pipe.image.height = content_h;
        return pipe;
    }

    bool ensureReadyRing(int depth, std::string& error)
    {
        if (int(ready_ring.size()) == depth)
        {
            bool ok = true;
            for (const DecodedTexture& t : ready_ring)
                ok = ok && t.cuda_ready;
            if (ok)
                return true;
        }

        if (!allocator)
        {
            error = "Vulkan Allocator not ready";
            return false;
        }

        for (DecodedTexture& t : ready_ring)
            t.destroy();
        ready_ring.clear();
        ready_ring.resize(size_t(depth));
        for (int i = 0; i < depth; ++i)
        {
            if (!ready_ring[size_t(i)].create(*this, *allocator, content_w, content_h, true, error))
                return false;
        }
        printLine("READY ring: {} x {}x{} VkImages (inflight)",
            depth, content_w, content_h);
        return true;
    }
#endif

    bool ensureHostRing(int depth, std::string& error)
    {
        bool ready = int(ready_ring.size()) == depth;
        for (const DecodedTexture& t : ready_ring)
            ready = ready && t.host_mapped && !t.cuda_ready && !t.hip_ready;
        if (ready)
            return true;

        if (!allocator)
        {
            error = "Vulkan Allocator not ready";
            return false;
        }

        for (DecodedTexture& t : ready_ring)
            t.destroy();
        ready_ring.clear();
        ready_ring.resize(size_t(depth));
        for (int i = 0; i < depth; ++i)
        {
            if (!ready_ring[size_t(i)].create(*this, *allocator, content_w, content_h, false, error))
                return false;
        }
        printLine("READY ring: {} x {}x{} VkImages (inflight)",
            depth, content_w, content_h);
        return true;
    }

    // One CPU decode. Two slots are enough: DMA of frame N runs while the CPU decodes N+1.
    PipelineResult runMangoReadyPipeline(const std::string& name)
    {
        PipelineResult pipe;
        pipe.backend = name + " [ready-pipe]";

        std::string error;
        const int depth = std::max(1, std::min(inflight, 2));
        if (!ensureHostRing(depth, error))
        {
            pipe.error = error;
            return pipe;
        }

        std::vector<bool> busy(size_t(depth), false);
        const u64 t0 = nowUs();
        for (int i = 0; i < pipeline_frames; ++i)
        {
            const int si = i % depth;
            DecodedTexture& tex = ready_ring[size_t(si)];
            if (busy[size_t(si)] && !tex.waitCopy(error))
            {
                pipe.error = error;
                return pipe;
            }
            busy[size_t(si)] = false;

            mango::u64 decode_us = 0;
            host_rgba.resize(tex.rgba_bytes);
            if (!decodeMangoTo(jpeg, host_rgba.data(), tex.width, tex.height, decode_us, error))
            {
                pipe.error = error;
                return pipe;
            }
            std::memcpy(tex.host_mapped, host_rgba.data(), tex.rgba_bytes);
            if (!tex.submitCopyHostToImage(graphicsQueue(), graphicsQueueFamilyIndex(), error))
            {
                pipe.error = error;
                return pipe;
            }
            busy[size_t(si)] = true;
        }
        for (int si = 0; si < depth; ++si)
        {
            if (busy[size_t(si)] && !ready_ring[size_t(si)].waitCopy(error))
            {
                pipe.error = error;
                return pipe;
            }
        }
        const u64 total_us = nowUs() - t0;

        pipe.ok = true;
        pipe.pipeline_overlapped = depth > 1 && pipeline_frames > 1;
        pipe.pipeline_count = pipeline_frames;
        pipe.pipeline_avg_us = total_us / u64(pipeline_frames);
        pipe.ready_us = pipe.pipeline_avg_us;
        pipe.decode_us = pipe.pipeline_avg_us;
        pipe.e2e_us = total_us;
        pipe.image.width = content_w;
        pipe.image.height = content_h;
        return pipe;
    }

#if defined(GPUJPEG_HAS_SOURCE_ROC)
    // rocJPEG 1.4 decode blocks in vaSyncSurface. The following VkImage copy is
    // queued before the next decode so the DMA sits inside that wait.
    PipelineResult runRocReadyPipeline(RocJpegDecoder& dec, DecodedTexture& tex, const std::string& name)
    {
        PipelineResult pipe;
        pipe.backend = name + " [ready-pipe]";
        pipe.device_only = true;
        pipe.pipeline_overlapped = false;

        std::string error;
        if (!tex.hip_ready || !tex.hip_rgba)
        {
            pipe.error = "HIP-Vulkan export buffer not available";
            return pipe;
        }

        // VCN decode is one blocking submit. The VkImage copy of frame N is in flight
        // during decode of frame N+1 (same exported RGBA, so the next expand waits).
        const u64 t0 = nowUs();
        bool copy_pending = false;
        for (int i = 0; i < pipeline_frames; ++i)
        {
            mango::u64 decode_us = 0;
            if (!dec.decodeToDevice(jpeg, decode_us, error))
            {
                pipe.error = error;
                return pipe;
            }
            if (copy_pending && !tex.waitCopy(error))
            {
                pipe.error = error;
                return pipe;
            }
            copy_pending = false;
            if (!dec.copyRgbToRgba(tex.hip_rgba, 0, error) ||
                !tex.submitCopyHipToImage(graphicsQueue(), graphicsQueueFamilyIndex(), error))
            {
                pipe.error = error;
                return pipe;
            }
            copy_pending = true;
        }
        if (copy_pending && !tex.waitCopy(error))
        {
            pipe.error = error;
            return pipe;
        }
        const u64 total_us = nowUs() - t0;

        pipe.ok = true;
        pipe.pipeline_overlapped = pipeline_frames > 1;
        pipe.pipeline_count = pipeline_frames;
        pipe.pipeline_avg_us = total_us / u64(pipeline_frames);
        pipe.ready_us = pipe.pipeline_avg_us;
        pipe.decode_us = pipe.pipeline_avg_us;
        pipe.e2e_us = total_us;
        pipe.image.width = content_w;
        pipe.image.height = content_h;
        return pipe;
    }
#endif

    PipelineResult runOne(const Job& job, BackendSlot& slot, int iter)
    {
        PipelineResult r;
        r.backend = iterations > 1 ? fmt::format("{} [{}]", job.name, iter) : job.name;

        const u64 t0 = nowUs();
        addMilestone(r.milestones, "start", t0);

        DecodedTexture& texture = slot.texture;
        std::string error;

        if (job.use_gpu)
        {
            if (!job.gpu)
            {
                r.error = "mango-gpu decoder missing";
                return r;
            }
            if (!texture.image)
            {
                r.error = "DecodedTexture missing for mango-gpu";
                return r;
            }
            if (!job.gpu->upload(*allocator, error))
            {
                r.error = error;
                return r;
            }
            if (!job.gpu->decodeBlocks(*this, *allocator, texture, error))
            {
                r.error = error;
                return r;
            }

            r.device_only = true;
            r.image.width = job.gpu->stream.width;
            r.image.height = job.gpu->stream.height;
            r.decode_us = job.gpu->huffman_us;
            r.promote_us = job.gpu->resolve_us;
            mango::u64 wall = nowUs() - t0;
            if (job.gpu->compile_last_us > 0 && wall > job.gpu->compile_last_us)
                wall -= job.gpu->compile_last_us;
            r.ready_us = wall;
            addMilestone(r.milestones, "vk_ready", t0);
            printLine("  gpu: huffman {:.2f} ms, idct {:.2f} ms, sync rounds {} chunk {} B{}",
                job.gpu->huffman_us / 1000.0,
                job.gpu->resolve_us / 1000.0,
                job.gpu->sync_rounds,
                job.gpu->sync_chunks,
                job.gpu->compile_last_us
                    ? fmt::format("  (shader compile {:.2f} ms)", job.gpu->compile_last_us / 1000.0)
                    : "");

            if (std::getenv("GPUJPEG_CHECK"))
            {
                const int delta = checkHybridImage(texture, jpeg,
                    graphicsQueue(), graphicsQueueFamilyIndex(), error);
                if (delta < 0)
                    printLine("  gpu check FAILED: {}", error);
                else
                    printLine("  gpu check: max |gpu-cpu| = {}", delta);
            }

            const u64 t_pr0 = nowUs();
            if (!presentSlotFullscreen(texture))
            {
                r.error = "present blit failed";
                return r;
            }
            r.present_us = nowUs() - t_pr0;
            r.e2e_us = r.ready_us + r.present_us;
            r.ok = true;
            slot.ready = true;
            setTitle(fmt::format("gpujpeg [mango-gpu] READY {:.2f} ms", r.ready_us / 1000.0));
            return r;
        }

        if (job.use_hybrid)
        {
#if defined(GPUJPEG_HAS_SOURCE_MANGO_HYBRID)
            if (job.hybrid_floor)
            {
                // Huffman-only floor: READY == Spectral Huffman wall time.
                MangoHybridHuffmanResult hr = runMangoHybridHuffman(jpeg);
                if (!hr.ok)
                {
                    r.error = hr.error;
                    return r;
                }

                r.device_only = true;
                r.image.width = hr.width;
                r.image.height = hr.height;
                r.decode_us = hr.huffman_us;
                r.promote_us = 0;
                r.present_us = 0;
                r.ready_us = hr.huffman_us;
                r.e2e_us = hr.huffman_us;
                r.ok = true;
                addMilestone(r.milestones, "huffman_spectral", t0);

                printLine("  hybrid floor: {} bands, {} K coeffs, {}x{} {}, {}",
                    hr.bands,
                    hr.samples / 1000,
                    hr.width, hr.height,
                    hr.chroma,
                    hr.encoding);
                if (hr.progressive)
                    printLine("  note: progressive → one Natural band at EOI");
                setTitle(fmt::format("gpujpeg [{}] Huffman {:.2f} ms",
                    job.name, hr.huffman_us / 1000.0));
                return r;
            }

            if (!texture.image)
            {
                r.error = "DecodedTexture missing for hybrid compute";
                return r;
            }

            if (!slot.hybrid_compute)
                slot.hybrid_compute = std::make_unique<MangoHybridCompute>();

            MangoHybridResult hr = runMangoHybridCompute(
                jpeg, *this, *allocator, texture, slot.hybrid_compute.get());
            if (!hr.ok)
            {
                r.error = hr.error;
                return r;
            }

            r.device_only = true;
            r.image.width = hr.width;
            r.image.height = hr.height;
            r.decode_us = hr.huffman_us;
            r.gpu_wait_us = hr.gpu_wait_us;
            // Tail after Huffman: compute still running, then the sampleable-layout barrier.
            if (hr.ready_us > hr.huffman_us)
                r.promote_us = hr.ready_us - hr.huffman_us;
            else
                r.promote_us = 0;
            r.ready_us = hr.ready_us;
            addMilestone(r.milestones, "vk_ready", t0);

            printLine("  hybrid: {} bands, {}x{} huffman={:.2f}ms ready={:.2f}ms gpu_wait={:.2f}ms",
                hr.bands, hr.width, hr.height,
                hr.huffman_us / 1000.0, hr.ready_us / 1000.0, hr.gpu_wait_us / 1000.0);

            if (std::getenv("GPUJPEG_CHECK"))
            {
                const int delta = checkHybridImage(texture, jpeg,
                    graphicsQueue(), graphicsQueueFamilyIndex(), error);
                if (delta < 0)
                    printLine("  hybrid check FAILED: {}", error);
                else
                    printLine("  hybrid check: max |gpu-cpu| = {}", delta);
            }

            const u64 t_pr0 = nowUs();
            if (!presentSlotFullscreen(texture))
            {
                r.error = "present blit failed";
                return r;
            }
            r.present_us = nowUs() - t_pr0;
            addMilestone(r.milestones, "presented", t0);

            r.e2e_us = nowUs() - t0;
            r.ok = true;
            slot.ready = true;
            setTitle(fmt::format("gpujpeg [{}] READY {:.2f} ms",
                job.name, hr.ready_us / 1000.0));
            return r;
#else
            r.error = "mango-hybrid not built";
            return r;
#endif
        }

#if defined(GPUJPEG_HAS_APPLE_VT)
        if (job.use_apple_vt)
        {
            slot.apple_image.destroy(device());

            AppleVtDecodeResult vt = appleVtDecodeJpeg(jpeg, content_w, content_h);
            if (!vt.ok)
            {
                r.error = vt.error;
                return r;
            }
            r.decode_us = vt.decode_us;
            r.image.width = vt.width;
            r.image.height = vt.height;
            addMilestone(r.milestones, "vt_decode", t0);

            const u64 t_prom0 = nowUs();
            if (!appleCreateVkImageFromPixelBuffer(physicalDevice(), device(),
                    vt.pixel_buffer, slot.apple_image, error))
            {
                AppleVtReleasePixelBuffer(vt.pixel_buffer);
                r.error = error;
                return r;
            }
            if (!applePrepareImportedForPresent(device(), graphicsQueue(),
                    graphicsQueueFamilyIndex(), slot.apple_image, error))
            {
                AppleVtReleasePixelBuffer(vt.pixel_buffer);
                r.error = error;
                return r;
            }
            AppleVtReleasePixelBuffer(vt.pixel_buffer);
            r.promote_us = nowUs() - t_prom0;
            r.device_only = true;
            r.ready_us = nowUs() - t0;
            addMilestone(r.milestones, "vk_ready", t0);

            const u64 t_pr0 = nowUs();
            if (!presentSlotApple(slot.apple_image))
            {
                r.error = "present blit failed";
                return r;
            }
            r.present_us = nowUs() - t_pr0;

            printLine("  apple-vt: {}x{} path={} hw_cap={} hw_active={} hw_prop={} decode={:.2f}ms promote={:.2f}ms ready={:.2f}ms present={:.2f}ms",
                r.image.width, r.image.height, vt.path,
                vt.hardware_capable ? "yes" : "no", vt.hardware ? "yes" : "no",
                vt.hardware_property_read ? "read" : "require*",
                r.decode_us / 1000.0, r.promote_us / 1000.0, r.ready_us / 1000.0,
                r.present_us / 1000.0);
            printLine("    decode detail: probe={:.2f} session={:.2f} vt_wait={:.2f} resolve={:.2f} ms"
                "  (present = swapchain blit; vt_wait = VT decode+sync wall clock)",
                vt.detail.probe_us / 1000.0, vt.detail.session_us / 1000.0,
                vt.detail.vt_wait_us / 1000.0, vt.detail.resolve_us / 1000.0);
            addMilestone(r.milestones, "presented", t0);

            r.e2e_us = nowUs() - t0;
            r.ok = true;
            slot.ready = true;
            setTitle(fmt::format("gpujpeg [{}] READY {:.2f} ms", job.name, r.ready_us / 1000.0));
            return r;
        }

        if (job.use_apple_imageio)
        {
            slot.apple_image.destroy(device());

            AppleImageIoResult io = appleImageIoDecodeJpeg(jpeg);
            if (!io.ok)
            {
                r.error = io.error;
                return r;
            }
            r.decode_us = io.decode_us;
            r.image.width = io.width;
            r.image.height = io.height;
            addMilestone(r.milestones, "imageio_decode", t0);

            const u64 t_prom0 = nowUs();
            if (!appleCreateVkImageFromPixelBuffer(physicalDevice(), device(),
                    io.pixel_buffer, slot.apple_image, error))
            {
                appleImageIoRelease(io.pixel_buffer);
                r.error = error;
                return r;
            }
            if (!applePrepareImportedForPresent(device(), graphicsQueue(),
                    graphicsQueueFamilyIndex(), slot.apple_image, error))
            {
                appleImageIoRelease(io.pixel_buffer);
                r.error = error;
                return r;
            }
            appleImageIoRelease(io.pixel_buffer);
            r.promote_us = nowUs() - t_prom0;
            r.device_only = true;
            r.ready_us = nowUs() - t0;
            addMilestone(r.milestones, "vk_ready", t0);

            const u64 t_pr0 = nowUs();
            if (!presentSlotApple(slot.apple_image))
            {
                r.error = "present blit failed";
                return r;
            }
            r.present_us = nowUs() - t_pr0;

            printLine("  apple-imageio: {}x{} decode={:.2f}ms render={:.2f}ms promote={:.2f}ms ready={:.2f}ms present={:.2f}ms",
                r.image.width, r.image.height,
                io.detail.decode_us / 1000.0, io.detail.render_us / 1000.0,
                r.promote_us / 1000.0, r.ready_us / 1000.0, r.present_us / 1000.0);
            addMilestone(r.milestones, "presented", t0);

            r.e2e_us = nowUs() - t0;
            r.ok = true;
            slot.ready = true;
            setTitle(fmt::format("gpujpeg [{}] READY {:.2f} ms", job.name, r.ready_us / 1000.0));
            return r;
        }
#endif

        if (!texture.image)
        {
            r.error = "DecodedTexture missing for backend";
            return r;
        }

        if (job.use_nvjpeg)
        {
            r.device_only = true;
            if (!job.decoder)
            {
                r.error = "nvjpeg decoder missing";
                return r;
            }
            if (!texture.cuda_ready)
            {
                r.error = "CUDA-Vulkan export buffer not available";
                return r;
            }

            u64 decode_us = 0;
            if (!job.decoder->decodeToDevice(jpeg, decode_us, error))
            {
                r.error = error;
                return r;
            }
            r.decode_us = decode_us;
            r.image.width = job.decoder->width;
            r.image.height = job.decoder->height;
            addMilestone(r.milestones, "gpu_ready", t0);

            // RGBI -> exported RGBA buffer (still device), then promote to VkImage.
            const u64 t_tex0 = nowUs();
#if defined(GPUJPEG_HAS_NVJPEG)
            if (!texture.writeDeviceRgbiToCudaRgba(
                    job.decoder->deviceRgbi(),
                    job.decoder->rgbiPitch(),
                    static_cast<cudaStream_t>(job.decoder->decodeStream()),
                    error))
            {
                r.error = error;
                return r;
            }
            if (cudaStreamSynchronize(static_cast<cudaStream_t>(job.decoder->decodeStream())) != cudaSuccess)
            {
                r.error = "cudaStreamSynchronize after NPP failed";
                return r;
            }
#else
            r.error = "nvJPEG not built";
            return r;
#endif

            // Promote exported RGBA buffer → intermediate VkImage (no swapchain acquire).
            if (!texture.flushCopyCudaToImage(graphicsQueue(), graphicsQueueFamilyIndex(), error))
            {
                r.error = error;
                return r;
            }
            r.promote_us = nowUs() - t_tex0;
            r.ready_us = nowUs() - t0; // primary: texture ingest done
            addMilestone(r.milestones, "vk_ready", t0);

            const u64 t_pr0 = nowUs();
            if (!presentSlotFullscreen(texture))
            {
                r.error = "present blit failed";
                return r;
            }
            r.present_us = nowUs() - t_pr0;
            addMilestone(r.milestones, "presented", t0);

            r.e2e_us = nowUs() - t0;
            r.ok = true;
            slot.ready = true;
            setTitle(fmt::format("gpujpeg [{}] {}x{} (VkImage)", job.name,
                r.image.width, r.image.height));
            return r;
        }

        if (job.use_roc)
        {
            r.device_only = true;
            if (!job.roc || !job.roc->ok)
            {
                r.error = "rocJPEG decoder missing";
                return r;
            }
            if (!texture.hip_ready || !texture.hip_rgba)
            {
                r.error = "HIP-Vulkan export buffer not available";
                return r;
            }

            u64 decode_us = 0;
            if (!job.roc->decodeToDevice(jpeg, decode_us, error))
            {
                r.error = error;
                return r;
            }
            r.decode_us = decode_us;
            r.image.width = job.roc->width;
            r.image.height = job.roc->height;
            addMilestone(r.milestones, "gpu_ready", t0);

            const u64 t_tex0 = nowUs();
            if (!job.roc->copyRgbToRgba(texture.hip_rgba, 0, error))
            {
                r.error = error;
                return r;
            }
            if (!texture.flushCopyHipToImage(graphicsQueue(), graphicsQueueFamilyIndex(), error))
            {
                r.error = error;
                return r;
            }
            r.promote_us = nowUs() - t_tex0;
            r.ready_us = nowUs() - t0;
            addMilestone(r.milestones, "vk_ready", t0);

            const char* chroma = job.roc->subsampling();
            if (chroma && chroma[0])
                printLine("  roc: {}x{} {} ready={:.2f}ms",
                    r.image.width, r.image.height, chroma, r.ready_us / 1000.0);

            const u64 t_pr0 = nowUs();
            if (!presentSlotFullscreen(texture))
            {
                r.error = "present blit failed";
                return r;
            }
            r.present_us = nowUs() - t_pr0;
            addMilestone(r.milestones, "presented", t0);

            r.e2e_us = nowUs() - t0;
            r.ok = true;
            slot.ready = true;
            setTitle(fmt::format("gpujpeg [{}] {}x{} (VkImage)", job.name,
                r.image.width, r.image.height));
            return r;
        }

        // SIMD decode into cached memory. The staging buffer is write-combined;
        // decoding into it directly is several times slower than one streaming copy.
        u64 decode_us = 0;
        host_rgba.resize(texture.rgba_bytes);
        if (!decodeMangoTo(jpeg, host_rgba.data(), texture.width, texture.height, decode_us, error))
        {
            r.error = error;
            return r;
        }
        r.decode_us = decode_us;
        r.image.width = texture.width;
        r.image.height = texture.height;
        r.image.components = 4;
        addMilestone(r.milestones, "decoded", t0);

        const u64 t_up0 = nowUs();
        std::memcpy(texture.host_mapped, host_rgba.data(), texture.rgba_bytes);
        if (!texture.flushCopyHostToImage(graphicsQueue(), graphicsQueueFamilyIndex(), error))
        {
            r.error = error;
            return r;
        }
        r.promote_us = nowUs() - t_up0;
        r.ready_us = nowUs() - t0; // primary: texture ingest done
        addMilestone(r.milestones, "vk_ready", t0);

        const u64 t_pr0 = nowUs();
        if (!presentSlotFullscreen(texture))
        {
            r.error = "present blit failed";
            return r;
        }
        r.present_us = nowUs() - t_pr0;
        addMilestone(r.milestones, "presented", t0);

        r.e2e_us = nowUs() - t0;
        r.ok = true;
        slot.ready = true;
        setTitle(fmt::format("gpujpeg [{}] {}x{}", job.name, r.image.width, r.image.height));
        return r;
    }
};

bool probeSize(ConstMemory jpeg, int& width, int& height, std::string& error)
{
    ImageDecoder decoder(jpeg, ".jpg");
    ImageHeader header = decoder.header();
    if (!header.width || !header.height)
    {
        error = "cannot read JPEG header (mango)";
        return false;
    }
    width = header.width;
    height = header.height;
    return true;
}

} // namespace

int main(int argc, const char* argv[])
{
    CommandLine commands(argc, argv);
    Args args;
    configureParser(commands, args);

    if (argc < 2)
    {
        commands.printHelp();
        printLine("");
        printLine("nvJPEG at build: {}",
#if defined(GPUJPEG_HAS_NVJPEG)
            "yes"
#else
            "no"
#endif
        );
        return 1;
    }

    if (!commands.parse())
        return 1;
    if (args.validate)
    {
        // Validation messages arrive as warnings and info. Those channels are off
        // unless something asks for them.
        printEnable(Print::Warning, true);
        printEnable(Print::Info, true);
    }
    if (!args.error.empty())
    {
        printLine("{}", args.error);
        return 1;
    }
    if (args.filename.empty())
    {
        printLine("Missing JPEG filename.");
        return 1;
    }

    printLine(getSystemInfo());
    printLine("Workbench matrix (built):");
    printLine("  sources: mango={} hybrid={} nvjpeg={} roc={}",
        sourceBuilt(DecodeSource::Mango),
        sourceBuilt(DecodeSource::MangoHybrid),
        sourceBuilt(DecodeSource::NvJpeg),
        sourceBuilt(DecodeSource::Roc));
    printLine("  outputs: vulkan={} opengl={} d3d11={} d3d12={}",
        outputBuilt(OutputApi::Vulkan),
        outputBuilt(OutputApi::OpenGL),
        outputBuilt(OutputApi::D3D11),
        outputBuilt(OutputApi::D3D12));
    printLine("nvJPEG runtime: {}", nvjpegAvailable() ? "yes" : "no");
    {
        const RocRuntimeInfo roc_info = queryRocRuntime();
        const bool roc_yes = roc_info.hardware || roc_info.hybrid;
        if (!roc_info.error.empty() && roc_info.devices.empty())
            printLine("rocJPEG runtime: no ({})", roc_info.error);
        else
            printLine("rocJPEG runtime: {}", roc_yes ? "yes" : "no");
        for (const RocDeviceStatus& dev : roc_info.devices)
        {
            printLine("  hip[{}]: {} ({}) hardware={} hybrid={}",
                dev.index, dev.name, dev.arch,
                dev.hardware ? "yes" : dev.hardware_error,
                dev.hybrid ? "yes" : dev.hybrid_error);
        }
    }

    const u64 t_file0 = nowUs();
    File file(args.filename);
    ConstMemory jpeg = file;
    const u64 file_us = nowUs() - t_file0;
    if (!jpeg.size)
    {
        printLine("Failed to read: {}", args.filename);
        return 1;
    }
    printLine("file: {} ({} KB)  load={:.2f} ms",
        args.filename, jpeg.size / 1024, file_us / 1000.0);

    int width = 0;
    int height = 0;
    std::string probe_error;
    if (!probeSize(jpeg, width, height, probe_error))
    {
        printLine("{}", probe_error);
        return 1;
    }
    printLine("header: {} x {}", width, height);

    std::vector<std::unique_ptr<NvJpegDecoder>> nv_decoders;
    std::vector<Job> jobs;

    const bool want_mango =
        args.backend == Backend::Mango ||
        args.backend == Backend::Both ||
        args.backend == Backend::All;
    const bool want_hybrid =
        args.backend == Backend::Hybrid ||
        args.backend == Backend::All;
    const bool want_nv =
        args.backend == Backend::NvJpeg ||
        args.backend == Backend::Both ||
        args.backend == Backend::All;
    const bool want_roc =
        args.backend == Backend::Roc ||
        args.backend == Backend::All;
    const bool want_apple =
        args.backend == Backend::AppleVt ||
        args.backend == Backend::AppleImageIo ||
        args.backend == Backend::All;

    if (want_mango)
    {
        Job mj;
        mj.name = "mango";
        jobs.push_back(std::move(mj));
    }

    if (want_hybrid)
    {
#if defined(GPUJPEG_HAS_SOURCE_MANGO_HYBRID)
        Job hj;
        hj.name = args.hybrid_floor ? "mango-hybrid-floor" : "mango-hybrid";
        hj.use_hybrid = true;
        hj.hybrid_floor = args.hybrid_floor;
        jobs.push_back(std::move(hj));
#else
        printLine("mango-hybrid requested but not built (GPUJPEG_SOURCE_MANGO_HYBRID).");
#endif
    }

    const bool want_gpu =
        args.backend == Backend::Gpu ||
        args.backend == Backend::All;
    std::vector<std::unique_ptr<MangoGpuDecoder>> gpu_decoders;
    if (want_gpu)
    {
            auto gpu = std::make_unique<MangoGpuDecoder>();
            std::string error;
            if (!gpu->extract(jpeg, error))
            {
                printLine("mango-gpu extract FAILED: {}", error);
                if (args.backend == Backend::Gpu)
                    return 1;
            }
            else
            {
                const jpeg::EntropyStream& s = gpu->stream;
                const int mcus = s.segments.empty() ? 0 : s.segments.front().mcus;
                printLine("mango-gpu: {} segments x {} MCU rows, packed {:.1f} KB (file {:.1f} KB), {}x{} MCU, block {}x{}, tables={}",
                    s.segments.size(),
                    s.rows_per_segment,
                    s.packed_bytes / 1024.0,
                    jpeg.size / 1024.0,
                    s.xmcu, s.ymcu,
                    s.xblock, s.yblock,
                    s.huffman.size());
                printLine("mango-gpu: extract {:.2f} ms  ({} MCUs, {} blocks/MCU)",
                    gpu->extract_us / 1000.0, mcus, s.blocks_in_mcu);
                std::fflush(stdout);

                Job job;
                job.name = "mango-gpu";
                job.use_gpu = true;
                job.gpu = gpu.get();
                jobs.push_back(std::move(job));
                gpu_decoders.push_back(std::move(gpu));
            }
    }

    auto addNv = [&](NvJpegMode mode)
    {
        auto dec = std::make_unique<NvJpegDecoder>();
        std::string error;
        if (!dec->init(mode, width, height, args.inflight, error))
        {
            printLine("nvjpeg [{}] init FAILED: {}", nvJpegModeName(mode), error);
            return;
        }
        printLine("nvjpeg [{}] init: {:.2f} ms  (inflight={})",
            nvJpegModeName(mode), dec->init_us / 1000.0, args.inflight);

        if (!dec->warmup(jpeg, args.warmup, error))
        {
            printLine("nvjpeg [{}] warmup FAILED: {}", nvJpegModeName(mode), error);
            return;
        }
        if (args.warmup > 0)
        {
            printLine("nvjpeg [{}] warmup: {:.2f} ms ({} rounds)",
                nvJpegModeName(mode), dec->warmup_us / 1000.0, args.warmup);
        }

        Job job;
        job.name = fmt::format("nvjpeg-{}", nvJpegModeName(mode));
        job.use_nvjpeg = true;
        job.use_hybrid = false;
        job.nv_mode = mode;
        job.decoder = dec.get();
        jobs.push_back(std::move(job));
        nv_decoders.push_back(std::move(dec));
    };

    if (want_nv)
    {
        if (!nvjpegAvailable())
            printLine("nvJPEG unavailable (no CUDA device).");
        else
        {
            if (args.nvjpeg_mode == NvJpegModeSelect::Hybrid || args.nvjpeg_mode == NvJpegModeSelect::Both)
                addNv(NvJpegMode::Hybrid);
            if (args.nvjpeg_mode == NvJpegModeSelect::Gpu || args.nvjpeg_mode == NvJpegModeSelect::Both)
                addNv(NvJpegMode::Gpu);
        }
    }

    std::vector<std::unique_ptr<RocJpegDecoder>> roc_decoders;
    if (want_roc)
    {
#if defined(GPUJPEG_HAS_SOURCE_ROC)
        const RocRuntimeInfo roc_info = queryRocRuntime();
        auto addRoc = [&](RocJpegMode mode, bool probed_ok, const std::string& why_not)
        {
            if (!probed_ok)
            {
                printLine("roc [{}] unavailable: {}", rocJpegModeName(mode),
                    why_not.empty() ? "rocJpegCreate failed" : why_not);
                return;
            }
            // init + warmup run in onDeviceReady, once the Vulkan device is known.
            auto dec = std::make_unique<RocJpegDecoder>();
            Job job;
            job.name = fmt::format("roc-{}", rocJpegModeName(mode));
            job.use_roc = true;
            job.roc_mode = mode;
            job.roc = dec.get();
            jobs.push_back(std::move(job));
            roc_decoders.push_back(std::move(dec));
        };

        if (!rocAvailable())
        {
            printLine("rocJPEG unavailable ({}).",
                roc_info.error.empty() ? "no HIP device accepted rocJpegCreate" : roc_info.error);
        }
        else
        {
            // Probe is "any device". The Vulkan peer is checked again at init.
            std::string hw_why;
            std::string hy_why;
            if (!roc_info.hardware)
            {
                for (const RocDeviceStatus& dev : roc_info.devices)
                {
                    if (!hw_why.empty())
                        hw_why += "; ";
                    hw_why += dev.hardware_error;
                }
            }
            if (!roc_info.hybrid)
            {
                for (const RocDeviceStatus& dev : roc_info.devices)
                {
                    if (!hy_why.empty())
                        hy_why += "; ";
                    hy_why += dev.hybrid_error;
                }
            }
            if (args.roc_mode == RocModeSelect::Hardware || args.roc_mode == RocModeSelect::Both)
                addRoc(RocJpegMode::Hardware, roc_info.hardware, hw_why);
            if (args.roc_mode == RocModeSelect::Hybrid || args.roc_mode == RocModeSelect::Both)
                addRoc(RocJpegMode::Hybrid, roc_info.hybrid, hy_why);
        }
#else
        printLine("rocJPEG requested but not built (install rocjpeg and reconfigure).");
#endif
    }

#if defined(GPUJPEG_HAS_APPLE_VT)
    const bool want_apple_vt =
        args.backend == Backend::AppleVt ||
        args.backend == Backend::All;
    const bool want_apple_imageio =
        args.backend == Backend::AppleImageIo ||
        args.backend == Backend::All;
    if (want_apple_vt)
    {
        AppleVtAvailability av = probeAppleVt();
        if (!av.ok)
            printLine("apple-vt unavailable: {}", av.error);
        else
        {
            Job j;
            j.name = av.hardware_jpeg ? "apple-vt [hw]" : "apple-vt";
            j.use_apple_vt = true;
            jobs.push_back(std::move(j));
            printLine("apple-vt: hardware JPEG={}", av.hardware_jpeg ? "yes" : "no");
        }
    }
    if (want_apple_imageio)
    {
        Job j;
        j.name = "apple-imageio";
        j.use_apple_imageio = true;
        jobs.push_back(std::move(j));
        printLine("apple-imageio: CGImageSource → CI render → BGRA IOSurface");
    }
#else
    if (want_apple)
        printLine("apple-vt requested but not built (APPLE + GPUJPEG_SOURCE_APPLE_VT).");
#endif

    if (jobs.empty())
    {
        printLine("No backends to run.");
        return 1;
    }

    // Instance / surface setup mirrors examples/vulkan/vkbasic.cpp —
    // platform surface extensions come from mango::vulkan::requiredSurfaceExtensions().
    std::vector<const char*> layers;
    if (args.validate)
        layers.push_back("VK_LAYER_KHRONOS_validation");

    InstanceExtensionProperties instanceExtensionProperties;
    std::vector<const char*> extensions = requiredSurfaceExtensions();
    if (instanceExtensionProperties.contains(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME))
        extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);

    VkApplicationInfo appInfo =
    {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "gpujpeg",
        .applicationVersion = 1,
        .pEngineName = "gpujpeg",
        .engineVersion = 1,
        .apiVersion = VK_MAKE_VERSION(1, 3, 0),
    };

    Instance instance(appInfo, layers, extensions);
    VulkanContext ctx(instance);

    int cell_w = 0;
    int cell_h = 0;
    fitCell(width, height, cell_w, cell_h);
    int cols = 1;
    int rows = 1;
    gridDims(int(jobs.size()), cols, rows);
    const int win_w = cell_w * cols;
    const int win_h = cell_h * rows;

    const bool cuda_export = want_nv && nvjpegAvailable();
    const bool hip_export = want_roc && rocAvailable();
    VulkanDeviceConfig deviceConfig;
    const VulkanDeviceConfig* windowConfig = nullptr;
    if (cuda_export || hip_export)
    {
        // CUDA and HIP both import an opaque-fd/win32 VkBuffer. See decoded_texture.
        appendCudaVkInteropDeviceExtensions(deviceConfig);
        windowConfig = &deviceConfig;
    }
#if defined(GPUJPEG_HAS_APPLE_VT)
    if (want_apple)
    {
        appendAppleVkInteropDeviceExtensions(deviceConfig);
        windowConfig = &deviceConfig;
    }
#endif

    // A non-null deviceCreateInfoPNext replaces mango's Vulkan 1.3 chain, so this
    // duplicates dynamic rendering, timeline semaphores, and descriptor indexing.
    const VkPhysicalDevice selected_device = selectPhysicalDevice(instance);
    VkPhysicalDeviceDescriptorIndexingFeatures supported_indexing =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
    };
    VkPhysicalDeviceImageCompressionControlFeaturesEXT supported_compression =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT,
        .pNext = &supported_indexing,
    };
    VkPhysicalDeviceFeatures2 supported_features =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &supported_compression,
    };
    bool compression_ext = false;
    const auto device_extensions = getDeviceExtensionProperties(selected_device);
    for (const VkExtensionProperties& ext : device_extensions)
    {
        if (std::strcmp(ext.extensionName, VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME) == 0)
            compression_ext = true;
    }
    if (compression_ext)
        vkGetPhysicalDeviceFeatures2(selected_device, &supported_features);

    VkPhysicalDeviceImageCompressionControlFeaturesEXT compression_features =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT,
        .imageCompressionControl = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features features12 =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &compression_features,
        .descriptorBindingStorageBufferUpdateAfterBind =
            supported_indexing.descriptorBindingStorageBufferUpdateAfterBind,
        .timelineSemaphore = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features features13 =
    {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &features12,
        .dynamicRendering = VK_TRUE,
    };
    if (compression_ext && supported_compression.imageCompressionControl)
    {
        deviceConfig.physicalDevice = selected_device;
        deviceConfig.deviceExtensions.push_back(VK_EXT_IMAGE_COMPRESSION_CONTROL_EXTENSION_NAME);
        deviceConfig.deviceCreateInfoPNext = &features13;
        windowConfig = &deviceConfig;
        printLine("image compression control: storage images uncompressed");
    }

    BenchWindow window(ctx, win_w, win_h, width, height, cuda_export, windowConfig);
    window.jpeg = jpeg;
    window.iterations = args.iterations;
    window.pipeline_frames = args.pipeline_frames;
    window.inflight = args.inflight;
    window.warmup = args.warmup;
    window.want_hip_texture = hip_export;
    window.jobs = std::move(jobs);

    // Slots are created in onDeviceReady. If ctor already brought the device up
    // before jobs were assigned, ensure create once.
    if (window.isDeviceReady() && window.slots.empty())
        window.onDeviceReady();

    EventLoopConfig config;
    config.mode = FrameMode::Continuous;
    config.waitForFrame = true;

    EventLoop loop;
    loop.attach(window, config);
    loop.run();

    if (window.allocator)
    {
        for (auto& gpu : gpu_decoders)
            gpu->release(*window.allocator);
    }

    printLine("==================================================");
    printLine("Primary metric: READY = start -> VkImage usable");
    printLine("{:<22} {:>9} {:>9} {:>9} {:>9} {:>9}  {}",
        "backend", "READY", "decode", "promote", "present", "e2e", "path");
    printLine("");
    for (const auto& r : window.results)
    {
        if (!r.ok)
            continue;
        if (r.pipeline_count > 0)
        {
            printLine("{:<22} {:>7.2f}ms {:>7.2f}ms {:>7.2f}ms {:>7.2f}ms {:>7.2f}ms  {}",
                r.backend, r.ready_us / 1000.0, r.ready_us / 1000.0,
                0.0, 0.0, r.e2e_us / 1000.0, "ready-pipe");
            continue;
        }
        const char* path = r.bitstream_upload ? "huffman-blocks"
            : ((r.device_only && r.promote_us == 0 && r.present_us == 0)
                ? "huffman-floor"
                : (r.device_only ? "file->VkImage" : "host->VkImage"));
        printLine("{:<22} {:>7.2f}ms {:>7.2f}ms {:>7.2f}ms {:>7.2f}ms {:>7.2f}ms  {}",
            r.backend,
            r.ready_us / 1000.0,
            r.decode_us / 1000.0,
            r.promote_us / 1000.0,
            r.present_us / 1000.0,
            r.e2e_us / 1000.0,
            path);
    }

    return 0;
}

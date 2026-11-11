#include "mango_hybrid_compute.hpp"
#include "mango_hybrid_backend.hpp"
#include "bench.hpp"

#include <mango/core/core.hpp>
#include <mango/vulkan/compiler.hpp>
#include <mango/vulkan/vulkan.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace gpujpeg
{
namespace
{

using namespace mango;
using namespace mango::image;
using namespace mango::image::jpeg;
using namespace mango::vulkan;

// Spectral index i → natural 8x8 position (same table as mango jpeg.hpp).
constexpr int kZigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

struct PushConstants
{
    int32_t width;
    int32_t height;
    int32_t xmcu;
    int32_t mcu_y0;
    int32_t mcu_rows;
    int32_t xblock;
    int32_t yblock;
    int32_t blocks_in_mcu;
    int32_t components;
    int32_t y_hsf, y_vsf, y_qt, y_off;
    int32_t cb_hsf, cb_vsf, cb_qt, cb_off;
    int32_t cr_hsf, cr_vsf, cr_qt, cr_off;
};

// One workgroup = one MCU. Threads cover up to 16x16 MCU pixels.
// Coeffs stay Spectral s16 (tightly packed). The shader sign-extends them.
// QT binding is Natural order.
const char* kComputeSource = R"GLSL(
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba8) uniform writeonly image2D uOut;
layout(binding = 1) readonly buffer Coeffs { uint c[]; };
layout(binding = 2) readonly buffer Qt { int qt[4 * 64]; };

layout(push_constant) uniform PC {
    int width;
    int height;
    int xmcu;
    int mcu_y0;
    int mcu_rows;
    int xblock;
    int yblock;
    int blocks_in_mcu;
    int components;
    int y_hsf, y_vsf, y_qt, y_off;
    int cb_hsf, cb_vsf, cb_qt, cb_off;
    int cr_hsf, cr_vsf, cr_qt, cr_off;
} pc;

const int zigzag[64] = int[](
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
);

shared int sh_spatial[10 * 64]; // max blocks_in_mcu <= 10

void idct_pass(inout int x0, inout int x1, inout int x2, inout int x3,
               inout int y0, inout int y1, inout int y2, inout int y3,
               int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7)
{
    int n0 = (s2 + s6) * 2217;
    int t2 = n0 + s6 * (-7567);
    int t3 = n0 + s2 * 3135;
    int t0 = (s0 + s4) << 12;
    int t1 = (s0 - s4) << 12;
    x0 = t0 + t3;
    x3 = t0 - t3;
    x1 = t1 + t2;
    x2 = t1 - t2;

    int p1 = s7 + s1;
    int p2 = s5 + s3;
    int p3 = s7 + s3;
    int p4 = s5 + s1;
    int p5 = (p3 + p4) * 4816;
    p1 = p1 * (-3685) + p5;
    p2 = p2 * (-10497) + p5;
    p3 = p3 * (-8034);
    p4 = p4 * (-1597);
    y0 = p1 + p3 + s7 * 1223;
    y1 = p2 + p4 + s5 * 8410;
    y2 = p2 + p3 + s3 * 12586;
    y3 = p1 + p4 + s1 * 6149;
}

int clamp_u8(int v)
{
    return clamp(v, 0, 255);
}

// Little-endian packed s16. index 0 is the low half of c[0].
int coeff_at(int index)
{
    uint w = c[index >> 1];
    uint bits = ((index & 1) == 0) ? (w & 0xffffu) : (w >> 16);
    return int(bits) - int((bits & 0x8000u) << 1);
}

void decode_block(int block_index, int base_coeff, int qt_index)
{
    int natural[64];
    int qtn[64];
    int temp[64];
    for (int i = 0; i < 64; ++i)
        qtn[i] = qt[qt_index * 64 + i];
    for (int i = 0; i < 64; ++i)
        natural[i] = 0;
    for (int i = 0; i < 64; ++i)
        natural[zigzag[i]] = coeff_at(base_coeff + i);

    for (int i = 0; i < 8; ++i)
    {
        bool ac = false;
        for (int k = 1; k < 8; ++k)
            ac = ac || (natural[i + 8 * k] != 0);

        if (ac)
        {
            int s0 = natural[i +  0] * qtn[i +  0];
            int s1 = natural[i +  8] * qtn[i +  8];
            int s2 = natural[i + 16] * qtn[i + 16];
            int s3 = natural[i + 24] * qtn[i + 24];
            int s4 = natural[i + 32] * qtn[i + 32];
            int s5 = natural[i + 40] * qtn[i + 40];
            int s6 = natural[i + 48] * qtn[i + 48];
            int s7 = natural[i + 56] * qtn[i + 56];
            int x0, x1, x2, x3, y0, y1, y2, y3;
            idct_pass(x0, x1, x2, x3, y0, y1, y2, y3, s0, s1, s2, s3, s4, s5, s6, s7);
            const int bias = 0x200;
            x0 += bias; x1 += bias; x2 += bias; x3 += bias;
            temp[i * 8 + 0] = (x0 + y3) >> 10;
            temp[i * 8 + 1] = (x1 + y2) >> 10;
            temp[i * 8 + 2] = (x2 + y1) >> 10;
            temp[i * 8 + 3] = (x3 + y0) >> 10;
            temp[i * 8 + 4] = (x3 - y0) >> 10;
            temp[i * 8 + 5] = (x2 - y1) >> 10;
            temp[i * 8 + 6] = (x1 - y2) >> 10;
            temp[i * 8 + 7] = (x0 - y3) >> 10;
        }
        else
        {
            int dc = (natural[i] * qtn[i]) << 2;
            for (int k = 0; k < 8; ++k)
                temp[i * 8 + k] = dc;
        }
    }

    const int shift = 8 + 9;
    int base = block_index * 64;
    for (int i = 0; i < 8; ++i)
    {
        int x0, x1, x2, x3, y0, y1, y2, y3;
        idct_pass(x0, x1, x2, x3, y0, y1, y2, y3,
            temp[i + 0], temp[i + 8], temp[i + 16], temp[i + 24],
            temp[i + 32], temp[i + 40], temp[i + 48], temp[i + 56]);
        const int bias = 0x10000 + (128 << shift);
        x0 += bias; x1 += bias; x2 += bias; x3 += bias;
        sh_spatial[base + i * 8 + 0] = clamp_u8((x0 + y3) >> shift);
        sh_spatial[base + i * 8 + 1] = clamp_u8((x1 + y2) >> shift);
        sh_spatial[base + i * 8 + 2] = clamp_u8((x2 + y1) >> shift);
        sh_spatial[base + i * 8 + 3] = clamp_u8((x3 + y0) >> shift);
        sh_spatial[base + i * 8 + 4] = clamp_u8((x3 - y0) >> shift);
        sh_spatial[base + i * 8 + 5] = clamp_u8((x2 - y1) >> shift);
        sh_spatial[base + i * 8 + 6] = clamp_u8((x1 - y2) >> shift);
        sh_spatial[base + i * 8 + 7] = clamp_u8((x0 - y3) >> shift);
    }
}

int sample_comp(int first_block, int hsf, int vsf, int lx, int ly)
{
    int block_w = max(pc.xblock / max(hsf, 1), 1);
    int block_h = max(pc.yblock / max(vsf, 1), 1);
    int bx = min(lx / block_w, hsf - 1);
    int by = min(ly / block_h, vsf - 1);
    int px = ((lx % block_w) * 8) / block_w;
    int py = ((ly % block_h) * 8) / block_h;
    px = clamp(px, 0, 7);
    py = clamp(py, 0, 7);
    int block = first_block + by * hsf + bx;
    return sh_spatial[block * 64 + py * 8 + px];
}

void main()
{
    uint mcu_index = gl_WorkGroupID.x;
    int mcu_rows = pc.mcu_rows;
    int xmcu = max(pc.xmcu, 1);
    int local_mcu_y = int(mcu_index / uint(xmcu));
    int mcu_x = int(mcu_index % uint(xmcu));
    // Workgroup-uniform. A return above the barrier is undefined if any lane disagrees.
    bool live = local_mcu_y < mcu_rows;

    int mcu_y = pc.mcu_y0 + local_mcu_y;
    int lx = int(gl_LocalInvocationID.x);
    int ly = int(gl_LocalInvocationID.y);

    // One thread per block in the MCU (up to blocks_in_mcu ≤ 10).
    int b = int(gl_LocalInvocationIndex);
    if (live && b < pc.blocks_in_mcu)
    {
        int global_mcu = pc.mcu_y0 * pc.xmcu + int(mcu_index);
        int mcu_coeff = global_mcu * pc.blocks_in_mcu * 64;
        int qt_index = 0;
        if (b >= pc.y_off && b < pc.y_off + pc.y_hsf * pc.y_vsf)
            qt_index = pc.y_qt;
        else if (pc.components > 1 && b >= pc.cb_off && b < pc.cb_off + pc.cb_hsf * pc.cb_vsf)
            qt_index = pc.cb_qt;
        else if (pc.components > 2 && b >= pc.cr_off && b < pc.cr_off + pc.cr_hsf * pc.cr_vsf)
            qt_index = pc.cr_qt;
        decode_block(b, mcu_coeff + b * 64, qt_index);
    }
    barrier();

    if (!live || lx >= pc.xblock || ly >= pc.yblock)
        return;

    int px = mcu_x * pc.xblock + lx;
    int py = mcu_y * pc.yblock + ly;
    if (px >= pc.width || py >= pc.height)
        return;

    int Y = sample_comp(pc.y_off, pc.y_hsf, pc.y_vsf, lx, ly);
    int R, G, B;
    if (pc.components == 1)
    {
        R = G = B = Y;
    }
    else
    {
        int Cb = sample_comp(pc.cb_off, pc.cb_hsf, pc.cb_vsf, lx, ly) - 128;
        int Cr = sample_comp(pc.cr_off, pc.cr_hsf, pc.cr_vsf, lx, ly) - 128;
        // JFIF YCbCr → RGB (integer approximation)
        R = clamp_u8(Y + ((Cr * 91881) >> 16));
        G = clamp_u8(Y - ((Cb * 22554 + Cr * 46802) >> 16));
        B = clamp_u8(Y + ((Cb * 116130) >> 16));
    }

    imageStore(uOut, ivec2(px, py), vec4(float(R), float(G), float(B), 255.0) / 255.0);
}
)GLSL";

void fillNaturalQt(const JpegInfo& info, int natural_qt[4][64])
{
    std::memset(natural_qt, 0, sizeof(int) * 4 * 64);
    for (int t = 0; t < 4; ++t)
    {
        if (!info.quant[t])
            continue;
        if (info.coeff_order == CoeffOrder::Spectral)
        {
            for (int i = 0; i < 64; ++i)
                natural_qt[t][kZigzag[i]] = info.quant[t][i];
        }
        else
        {
            for (int i = 0; i < 64; ++i)
                natural_qt[t][i] = info.quant[t][i];
        }
    }
}

PushConstants makePush(const JpegInfo& info, int mcu_y0, int mcu_rows)
{
    PushConstants pc {};
    pc.width = info.width;
    pc.height = info.height;
    pc.xmcu = info.xmcu;
    pc.mcu_y0 = mcu_y0;
    pc.mcu_rows = mcu_rows;
    pc.xblock = info.xblock;
    pc.yblock = info.yblock;
    pc.blocks_in_mcu = info.blocks_in_mcu;
    pc.components = info.components;

    int off = 0;
    auto setComp = [&](int idx, int32_t& hsf, int32_t& vsf, int32_t& qt, int32_t& o)
    {
        if (idx >= info.components)
        {
            hsf = vsf = 1;
            qt = 0;
            o = 0;
            return;
        }
        hsf = info.component[idx].hsf;
        vsf = info.component[idx].vsf;
        qt = info.component[idx].qt;
        o = off;
        off += hsf * vsf;
    };
    setComp(0, pc.y_hsf, pc.y_vsf, pc.y_qt, pc.y_off);
    setComp(1, pc.cb_hsf, pc.cb_vsf, pc.cb_qt, pc.cb_off);
    setComp(2, pc.cr_hsf, pc.cr_vsf, pc.cr_qt, pc.cr_off);
    return pc;
}

} // namespace

MangoHybridCompute::~MangoHybridCompute()
{
    destroy();
}

std::size_t hybridSegmentRowsPerTask(int ymcu, int hardware_threads)
{
    constexpr int max_threads = 64;
    constexpr int min_rows_per_task = 8;
    if (hardware_threads <= 1)
        return size_t(ymcu);
    const int threads = std::min(hardware_threads, max_threads);
    return size_t(std::min(ymcu, std::max(ymcu / threads, min_rows_per_task)));
}

void MangoHybridCompute::destroy()
{
    m_huffman_done = true;
    m_gpu_cv.notify_all();
    if (m_graphics_thread.joinable())
        m_graphics_thread.join();
    freeFrameResources();
    m_pending.clear();
    m_row_covered.clear();
    m_rows_covered = 0;

    if (!m_device)
        return;

    vkDeviceWaitIdle(m_device);

    if (m_prepare_fence)
        vkDestroyFence(m_device, m_prepare_fence, nullptr);
    m_prepare_fence = VK_NULL_HANDLE;

    if (m_layout_fence)
        vkDestroyFence(m_device, m_layout_fence, nullptr);
    m_layout_fence = VK_NULL_HANDLE;

    if (m_allocator && m_qt_buffer)
        m_allocator->destroyBuffer(m_qt_buffer);
    m_qt_buffer = {};

    if (m_desc_pool)
        vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
    if (m_pipeline)
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
    if (m_pipeline_layout)
        vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
    if (m_set_layout)
        vkDestroyDescriptorSetLayout(m_device, m_set_layout, nullptr);
    if (m_shader)
        vkDestroyShaderModule(m_device, m_shader, nullptr);
    if (m_cmd_pool)
        vkDestroyCommandPool(m_device, m_cmd_pool, nullptr);

    m_desc_pool = VK_NULL_HANDLE;
    m_pipeline = VK_NULL_HANDLE;
    m_pipeline_layout = VK_NULL_HANDLE;
    m_set_layout = VK_NULL_HANDLE;
    m_shader = VK_NULL_HANDLE;
    m_cmd_pool = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_allocator = nullptr;
    m_texture = nullptr;
    m_frame_gpu_started = false;
    m_gpu_wait_us = 0;
    m_handoffs = 0;
    m_gpu_error = false;
    m_gpu_done = false;
    m_gpu_error_msg.clear();
}

void MangoHybridCompute::addGpuWaitUs(u64 us)
{
    m_gpu_wait_us.fetch_add(us);
}

void MangoHybridCompute::freeFrameResources()
{
    if (!m_device)
    {
        m_pending.clear();
        m_row_covered.clear();
        m_rows_covered = 0;
        m_segment_gpu.clear();
        return;
    }

    for (SegmentGpu& g : m_segment_gpu)
    {
        if (g.fence)
            vkDestroyFence(m_device, g.fence, nullptr);
        g.fence = VK_NULL_HANDLE;
        if (g.cmd)
            vkFreeCommandBuffers(m_device, m_cmd_pool, 1, &g.cmd);
        g.cmd = VK_NULL_HANDLE;
    }
    m_segment_gpu.clear();

    if (m_prepare_cmd)
        vkFreeCommandBuffers(m_device, m_cmd_pool, 1, &m_prepare_cmd);
    m_prepare_cmd = VK_NULL_HANDLE;

    if (m_layout_cmd)
        vkFreeCommandBuffers(m_device, m_cmd_pool, 1, &m_layout_cmd);
    m_layout_cmd = VK_NULL_HANDLE;

    if (m_desc_pool)
    {
        vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
        m_desc_pool = VK_NULL_HANDLE;
    }

    if (m_allocator && m_frame_staging)
        m_allocator->destroyBuffer(m_frame_staging);
    m_frame_staging = {};
    m_frame_samples = 0;
}

bool MangoHybridCompute::createPipeline(std::string& error)
{
    Compiler compiler;
    Shader shader = compiler.compile(kComputeSource, ShaderStage::Compute);
    if (!shader)
    {
        error = "hybrid compute shader compile failed:\n" + shader.log;
        return false;
    }
    m_shader = Compiler::createShaderModule(m_device, shader);

    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo set_info =
    {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(m_device, &set_info, nullptr, &m_set_layout) != VK_SUCCESS)
    {
        error = "descriptor set layout failed";
        return false;
    }

    VkPushConstantRange push =
    {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    VkPipelineLayoutCreateInfo pl_info =
    {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    };
    if (vkCreatePipelineLayout(m_device, &pl_info, nullptr, &m_pipeline_layout) != VK_SUCCESS)
    {
        error = "pipeline layout failed";
        return false;
    }

    VkComputePipelineCreateInfo pipe_info =
    {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = m_shader,
            .pName = "main",
        },
        .layout = m_pipeline_layout,
    };
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &m_pipeline) != VK_SUCCESS)
    {
        error = "compute pipeline failed";
        return false;
    }

    return true;
}

bool MangoHybridCompute::create(VulkanWindow& window, Allocator& allocator,
    DecodedTexture& texture, const JpegInfo& info, std::string& error)
{
    destroy();

    if (!texture.image || !texture.view)
    {
        error = "DecodedTexture missing for hybrid compute";
        return false;
    }
    if (info.xblock > 16 || info.yblock > 16)
    {
        error = fmt::format("MCU {}x{} exceeds compute workgroup (16x16)", info.xblock, info.yblock);
        return false;
    }

    m_device = window.device();
    m_queue = window.graphicsQueue();
    m_queue_family = window.graphicsQueueFamilyIndex();
    m_allocator = &allocator;
    m_texture = &texture;
    m_info = info;
    fillNaturalQt(info, m_natural_qt);

    VkCommandPoolCreateInfo pool_ci =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_queue_family,
    };
    if (vkCreateCommandPool(m_device, &pool_ci, nullptr, &m_cmd_pool) != VK_SUCCESS)
    {
        error = "command pool failed";
        return false;
    }

    if (!createPipeline(error))
        return false;

    // QT buffer (Natural order ints)
    m_qt_buffer = m_allocator->createBuffer(sizeof(m_natural_qt),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        MemoryUsage::Upload, true);
    if (!m_qt_buffer || !m_qt_buffer.mapped)
    {
        error = "QT buffer alloc failed";
        return false;
    }
    std::memcpy(m_qt_buffer.mapped, m_natural_qt, sizeof(m_natural_qt));
    m_allocator->flush(m_qt_buffer.allocation, 0, sizeof(m_natural_qt));

    VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(m_device, &fence_ci, nullptr, &m_layout_fence) != VK_SUCCESS)
    {
        error = "layout fence failed";
        return false;
    }
    if (vkCreateFence(m_device, &fence_ci, nullptr, &m_prepare_fence) != VK_SUCCESS)
    {
        error = "prepare fence failed";
        return false;
    }
    return true;
}

bool MangoHybridCompute::allocFrameResources(std::string& error)
{
    freeFrameResources();

    const size_t coeffs_per_row = size_t(m_info.xmcu) * size_t(m_info.blocks_in_mcu) * 64;
    m_frame_samples = coeffs_per_row * size_t(m_info.ymcu);
    const VkDeviceSize bytes = (VkDeviceSize(m_frame_samples) * sizeof(s16) + 3u) & ~VkDeviceSize(3);

    m_frame_staging = m_allocator->createBuffer(bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryUsage::Upload, true);
    if (!m_frame_staging || !m_frame_staging.mapped)
    {
        error = "frame coeff staging failed";
        return false;
    }

    // Grow only. resize() value-initializes, and the entropy loop already zeros
    // each MCU. Do not wipe the buffer on every frame.
    if (m_host_coeffs.size() != m_frame_samples)
        m_host_coeffs.resize(m_frame_samples);

    // One slot per MCU row: serial JPEGs emit short strips, Mango1 emits wider ones.
    const uint32_t n = uint32_t(std::max(m_info.ymcu, 1));
    VkDescriptorPoolSize sizes[2] =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, n },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n * 2 },
    };
    VkDescriptorPoolCreateInfo pool_info =
    {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = n + 1,
        .poolSizeCount = 2,
        .pPoolSizes = sizes,
    };
    if (vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_desc_pool) != VK_SUCCESS)
    {
        error = "frame descriptor pool failed";
        return false;
    }

    VkDescriptorBufferInfo coeff_info =
    {
        .buffer = m_frame_staging.buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo qt_info =
    {
        .buffer = m_qt_buffer.buffer,
        .offset = 0,
        .range = sizeof(m_natural_qt),
    };
    VkDescriptorImageInfo img_info =
    {
        .sampler = VK_NULL_HANDLE,
        .imageView = m_texture->view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    m_segment_gpu.resize(n);
    std::vector<VkDescriptorSetLayout> layouts(n, m_set_layout);
    VkDescriptorSetAllocateInfo set_alloc =
    {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_desc_pool,
        .descriptorSetCount = n,
        .pSetLayouts = layouts.data(),
    };
    std::vector<VkDescriptorSet> sets(n);
    if (vkAllocateDescriptorSets(m_device, &set_alloc, sets.data()) != VK_SUCCESS)
    {
        error = "frame descriptor sets failed";
        return false;
    }

    VkCommandBufferAllocateInfo cmd_alloc =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = n + 2,
    };
    std::vector<VkCommandBuffer> cmds(n + 2);
    if (vkAllocateCommandBuffers(m_device, &cmd_alloc, cmds.data()) != VK_SUCCESS)
    {
        error = "frame command buffers failed";
        return false;
    }
    m_prepare_cmd = cmds[n];
    m_layout_cmd = cmds[n + 1];

    for (uint32_t i = 0; i < n; ++i)
    {
        m_segment_gpu[i].cmd = cmds[i];
        m_segment_gpu[i].desc_set = sets[i];
        VkFenceCreateInfo fence_ci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(m_device, &fence_ci, nullptr, &m_segment_gpu[i].fence) != VK_SUCCESS)
        {
            error = "segment fence failed";
            return false;
        }
        VkWriteDescriptorSet writes[3] = {};
        writes[0] = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[i],
            .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &img_info };
        writes[1] = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[i],
            .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &coeff_info };
        writes[2] = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[i],
            .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &qt_info };
        vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
    }
    return true;
}

bool MangoHybridCompute::startFrame(const JpegInfo& info, int hardware_threads, std::string& error)
{
    if (m_graphics_thread.joinable())
        m_graphics_thread.join();

    resetGpuWait();
    m_info = info;
    m_frame_gpu_started = false;
    m_handoffs = 0;
    m_huffman_done = false;
    m_gpu_done = false;
    m_gpu_error = false;
    m_gpu_error_msg.clear();

    (void)hardware_threads;
    m_pending.clear();
    m_row_covered.assign(size_t(std::max(info.ymcu, 0)), 0);
    m_rows_covered = 0;

    if (!allocFrameResources(error))
        return false;
    if (!recordPrepare(error))
        return false;

    vkResetFences(m_device, 1, &m_prepare_fence);
    VkSubmitInfo prepare_submit =
    {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &m_prepare_cmd,
    };
    if (vkQueueSubmit(m_queue, 1, &prepare_submit, m_prepare_fence) != VK_SUCCESS)
    {
        error = "prepare submit failed";
        return false;
    }

    m_graphics_thread = std::thread([this] { graphicsThreadMain(); });
    return true;
}

mango::s16* MangoHybridCompute::coeffStorage()
{
    return m_host_coeffs.empty() ? nullptr : m_host_coeffs.data();
}

bool MangoHybridCompute::handoffBand(const CoeffView& view, std::string& error)
{
    if (!view.data || m_host_coeffs.empty() || !m_frame_staging.mapped)
    {
        error = "hybrid frame not started";
        return false;
    }
    if (m_gpu_error.load())
    {
        error = m_gpu_error_msg;
        return false;
    }

    const int y0 = view.mcu_y0;
    const int y1 = view.mcu_y1;
    const int rows = y1 - y0;
    const size_t stride = size_t(m_info.xmcu) * size_t(m_info.blocks_in_mcu) * 64;
    if (rows <= 0 || y0 < 0 || y1 > m_info.ymcu)
    {
        error = fmt::format("coefficient band rows {}..{} outside 0..{}", y0, y1, m_info.ymcu);
        return false;
    }
    if (view.samples != size_t(rows) * stride)
    {
        error = fmt::format("band y0={} sample mismatch", y0);
        return false;
    }

    const s16* expected = m_host_coeffs.data() + size_t(y0) * stride;
    if (view.data != expected)
    {
        error = fmt::format("band y0={} is outside host coefficient storage", y0);
        return false;
    }

    SegmentPlan plan;
    plan.mcu_y0 = y0;
    plan.mcu_y1 = y1;
    plan.coeff_offset = size_t(y0) * stride;
    plan.samples = view.samples;

    {
        std::lock_guard<std::mutex> lock(m_gpu_mutex);
        for (int y = y0; y < y1; ++y)
        {
            if (m_row_covered[size_t(y)])
            {
                error = fmt::format("MCU row {} published twice", y);
                return false;
            }
        }
        for (int y = y0; y < y1; ++y)
            m_row_covered[size_t(y)] = 1;
        m_rows_covered += rows;
        m_pending.push_back(plan);
    }

    m_handoffs.fetch_add(1, std::memory_order_relaxed);
    m_gpu_cv.notify_one();
    return true;
}

bool MangoHybridCompute::recordPrepare(std::string& /*error*/)
{
    vkResetCommandBuffer(m_prepare_cmd, 0);
    VkCommandBufferBeginInfo begin =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(m_prepare_cmd, &begin);

    const VkImageLayout layout = m_texture->layout;
    if (layout != VK_IMAGE_LAYOUT_GENERAL)
    {
        VkImageMemoryBarrier barrier =
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT : 0u,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_texture->image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(m_prepare_cmd,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
        m_texture->layout = VK_IMAGE_LAYOUT_GENERAL;
    }
    vkEndCommandBuffer(m_prepare_cmd);
    return true;
}

bool MangoHybridCompute::recordBandCommand(const SegmentPlan& plan, SegmentGpu& gpu, std::string& /*error*/)
{
    vkResetCommandBuffer(gpu.cmd, 0);
    VkCommandBufferBeginInfo begin =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(gpu.cmd, &begin);

    // Host coefficient writes are available after flush. They are not visible to
    // the shader until this dependency. The image barrier orders earlier compute
    // stores to the same VkImage (submission order alone is not a memory dependency).
    VkMemoryBarrier host_to_shader =
    {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    VkImageMemoryBarrier order_stores =
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = m_texture->image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(gpu.cmd,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &host_to_shader, 0, nullptr, 1, &order_stores);

    const int rows = plan.mcu_y1 - plan.mcu_y0;
    PushConstants pc = makePush(m_info, plan.mcu_y0, rows);
    vkCmdBindPipeline(gpu.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(gpu.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &gpu.desc_set, 0, nullptr);
    vkCmdPushConstants(gpu.cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(gpu.cmd, uint32_t(rows * m_info.xmcu), 1, 1);
    vkEndCommandBuffer(gpu.cmd);
    return true;
}

void MangoHybridCompute::graphicsThreadMain()
{
    {
        const u64 t0 = nowUs();
        if (vkWaitForFences(m_device, 1, &m_prepare_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        {
            m_gpu_error_msg = "prepare fence failed";
            m_gpu_error = true;
            m_gpu_done = true;
            m_gpu_cv.notify_all();
            return;
        }
        addGpuWaitUs(nowUs() - t0);
        vkResetFences(m_device, 1, &m_prepare_fence);
    }
    m_frame_gpu_started = true;

    const int total_rows = m_info.ymcu;
    int submitted_rows = 0;
    int slot = 0;
    std::vector<int> submitted_slots;

    while (submitted_rows < total_rows)
    {
        std::vector<SegmentPlan> batch;
        {
            std::unique_lock<std::mutex> lock(m_gpu_mutex);
            m_gpu_cv.wait(lock, [&]
            {
                if (m_gpu_error.load())
                    return true;
                if (!m_pending.empty())
                    return true;
                return m_huffman_done.load();
            });
            batch.swap(m_pending);
        }

        if (m_gpu_error.load())
            return;

        if (batch.empty())
            break;

        for (const SegmentPlan& plan : batch)
        {
            if (slot >= int(m_segment_gpu.size()))
            {
                m_gpu_error_msg = "more coefficient bands than MCU rows";
                m_gpu_error = true;
                m_gpu_cv.notify_all();
                return;
            }
            std::string rec_err;
            if (!recordBandCommand(plan, m_segment_gpu[size_t(slot)], rec_err))
            {
                m_gpu_error_msg = rec_err;
                m_gpu_error = true;
                m_gpu_cv.notify_all();
                return;
            }

            const VkDeviceSize off = VkDeviceSize(plan.coeff_offset) * sizeof(s16);
            const VkDeviceSize bytes = VkDeviceSize(plan.samples) * sizeof(s16);
            std::memcpy(reinterpret_cast<s16*>(m_frame_staging.mapped) + plan.coeff_offset,
                m_host_coeffs.data() + plan.coeff_offset, size_t(bytes));
            m_allocator->flush(m_frame_staging.allocation, off, bytes);

            VkSubmitInfo submit =
            {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &m_segment_gpu[size_t(slot)].cmd,
            };
            if (vkQueueSubmit(m_queue, 1, &submit, m_segment_gpu[size_t(slot)].fence) != VK_SUCCESS)
            {
                m_gpu_error_msg = "vkQueueSubmit failed";
                m_gpu_error = true;
                m_gpu_cv.notify_all();
                return;
            }
            submitted_slots.push_back(slot);
            submitted_rows += plan.mcu_y1 - plan.mcu_y0;
            ++slot;
        }
    }

    std::string err;
    if (submitted_rows != total_rows)
        err = "missing coefficient rows before GPU dispatch";
    else if (!submitted_slots.empty())
    {
        std::vector<VkFence> fences;
        fences.reserve(submitted_slots.size());
        for (int s : submitted_slots)
            fences.push_back(m_segment_gpu[size_t(s)].fence);
        const u64 t0 = nowUs();
        if (vkWaitForFences(m_device, uint32_t(fences.size()), fences.data(), VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            err = "compute fence failed";
        addGpuWaitUs(nowUs() - t0);
        vkResetFences(m_device, uint32_t(fences.size()), fences.data());

        if (err.empty() && m_frame_gpu_started && recordLayoutToSampleReady(err))
        {
            vkResetFences(m_device, 1, &m_layout_fence);
            VkSubmitInfo submit =
            {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &m_layout_cmd,
            };
            if (vkQueueSubmit(m_queue, 1, &submit, m_layout_fence) != VK_SUCCESS)
                err = "layout submit failed";
            else
            {
                const u64 t1 = nowUs();
                if (vkWaitForFences(m_device, 1, &m_layout_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
                    err = "layout fence failed";
                addGpuWaitUs(nowUs() - t1);
            }
        }
    }
    if (!err.empty())
    {
        m_gpu_error_msg = err;
        m_gpu_error = true;
    }
    m_gpu_done = true;
    m_gpu_cv.notify_all();
}

bool MangoHybridCompute::recordLayoutToSampleReady(std::string& /*error*/)
{
    if (m_texture->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        return true;

    vkResetCommandBuffer(m_layout_cmd, 0);
    VkCommandBufferBeginInfo begin =
    {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(m_layout_cmd, &begin);
    m_texture->recordPublishComputeStores(m_layout_cmd);
    vkEndCommandBuffer(m_layout_cmd);
    return true;
}

bool MangoHybridCompute::finishFrame(std::string& error)
{
    m_huffman_done = true;
    m_gpu_cv.notify_all();

    if (m_graphics_thread.joinable())
        m_graphics_thread.join();

    if (m_gpu_error.load())
    {
        error = m_gpu_error_msg;
        return false;
    }
    return m_gpu_done.load();
}

MangoHybridResult runMangoHybridCompute(ConstMemory jpeg,
    VulkanWindow& window, Allocator& allocator, DecodedTexture& texture,
    MangoHybridCompute* reused)
{
    MangoHybridResult r;

    JpegDecoder decoder(jpeg);
    const JpegInfo& info = decoder.info();

    std::string reason;
    if (!mangoHybridEligible(info, reason))
    {
        r.error = reason;
        return r;
    }
    if (info.progressive)
    {
        r.error = "progressive JPEG (hybrid compute v1 is baseline Spectral only)";
        return r;
    }

    r.width = info.width;
    r.height = info.height;

    decoder.setCoeffOrder(CoeffOrder::Spectral);

    MangoHybridCompute local;
    MangoHybridCompute& compute = reused ? *reused : local;
    std::string error;

    // First use (or non-reused): build pipeline. Subsequent iters skip compile.
    if (!reused || !reused->created())
    {
        if (!compute.create(window, allocator, texture, info, error))
        {
            r.error = error;
            return r;
        }
    }

    const int hw_threads = int(std::thread::hardware_concurrency());
    if (!compute.startFrame(info, hw_threads > 0 ? hw_threads : 8, error))
    {
        r.error = error;
        return r;
    }

    // Cached host buffer. The upload mapping is filled later, one finished segment
    // at a time, on the graphics thread.
    decoder.setCoeffStorage(compute.coeffStorage(), compute.coeffSamples());

    struct SinkState
    {
        MangoHybridCompute* compute = nullptr;
        std::atomic<int>* bands = nullptr;
        std::mutex* error_mu = nullptr;
        std::string* error = nullptr;
    };
    std::atomic<int> bands { 0 };
    std::mutex error_mu;
    SinkState sink { &compute, &bands, &error_mu, &error };
    decoder.setCoeffSink([](const CoeffView& view, void* user)
    {
        auto* sink = static_cast<SinkState*>(user);
        sink->bands->fetch_add(1, std::memory_order_relaxed);
        std::string local;
        if (!sink->compute->handoffBand(view, local))
        {
            std::lock_guard<std::mutex> lock(*sink->error_mu);
            if (sink->error->empty())
                *sink->error = std::move(local);
        }
    }, &sink);

    ImageDecodeOptions options;
    // Parallel Huffman when Mango1 offsets or row-aligned DRI (restartInterval == xmcu).
    options.multithread = true;

    const u64 t0 = nowUs();
    ImageDecodeStatus status = decoder.decodeCoefficients(options);
    const u64 t_huff = nowUs();

    if (!status)
    {
        r.error = status.info.empty() ? "decodeCoefficients failed" : status.info;
        return r;
    }
    if (!error.empty())
    {
        r.error = error;
        return r;
    }
    if (!compute.finishFrame(error))
    {
        r.error = error;
        return r;
    }

    const u64 t1 = nowUs();
    r.bands = bands.load();
    r.huffman_us = t_huff - t0;
    r.ready_us = t1 - t0;
    r.gpu_wait_us = compute.gpuWaitUs();
    r.ok = true;
    return r;
}

} // namespace gpujpeg

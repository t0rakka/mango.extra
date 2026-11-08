#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <mango/core/core.hpp>

namespace gpujpeg
{

struct Milestone
{
    const char* name = nullptr;
    mango::u64 us_from_start = 0;
};

struct DecodedImage
{
    int width = 0;
    int height = 0;
    int components = 0;
    std::vector<mango::u8> rgba; // tightly packed RGBA8 (host; mango / debug only)
};

struct PipelineResult
{
    std::string backend;
    DecodedImage image;
    std::vector<Milestone> milestones;

    // Primary: job start → intermediate VkImage usable as texture/compute input.
    mango::u64 ready_us = 0;

    mango::u64 decode_us = 0;   // bitstream → device RGBI or host RGBA
    mango::u64 gpu_wait_us = 0; // hybrid: single-frame compute + layout fence
    mango::u64 promote_us = 0;  // → intermediate VkImage (NPP+copy or host staging)
    mango::u64 present_us = 0;  // blit to swapchain (completion check; vsync-sensitive)
    mango::u64 e2e_us = 0;      // start → presented (guarantee work finished on screen)

    mango::u64 pipeline_avg_us = 0;
    int pipeline_count = 0;
    bool pipeline_overlapped = false;
    bool device_only = false;
    bool ok = false;
    std::string error;
};

inline mango::u64 nowUs()
{
    return mango::Time::us();
}

inline void addMilestone(std::vector<Milestone>& out, const char* name, mango::u64 t0)
{
    out.push_back({ name, nowUs() - t0 });
}

inline void printPipeline(const PipelineResult& r)
{
    mango::printLine("--------------------------------------------------");
    mango::printLine("backend: {}", r.backend);
    if (!r.ok)
    {
        mango::printLine("  FAILED: {}", r.error);
        return;
    }

    if (r.pipeline_count > 0)
    {
        const double ms = r.pipeline_avg_us / 1000.0;
        const double tex_s = ms > 0.0 ? 1000.0 / ms : 0.0;
        mango::printLine("  path:    {} READY (file -> VkImage)",
            r.pipeline_overlapped ? "overlapped" : "sequential");
        mango::printLine("  READY:   {:8.2f} ms/tex   ({:.1f} tex/s, {} frames)",
            ms, tex_s, r.pipeline_count);
        return;
    }

    const bool huffman_floor = r.device_only && r.promote_us == 0 && r.present_us == 0;

    if (huffman_floor)
    {
        mango::printLine("  path:    Spectral Huffman floor (sink drops bands)");
        mango::printLine("  image:   {} x {} (coeffs only — no VkImage yet)",
            r.image.width, r.image.height);
        mango::printLine("  READY:   {:8.2f} ms   <-- Huffman w/o zigzag (compute target)",
            r.ready_us / 1000.0);
        mango::printLine("    huffman: {:8.2f} ms", r.decode_us / 1000.0);
    }
    else if (r.device_only)
    {
        mango::printLine("  path:    file -> VkImage (no host bounce)");
        mango::printLine("  image:   {} x {} (intermediate VkImage)", r.image.width, r.image.height);
        mango::printLine("  READY:   {:8.2f} ms   <-- VkImage usable", r.ready_us / 1000.0);
        mango::printLine("    decode:  {:8.2f} ms", r.decode_us / 1000.0);
        if (r.gpu_wait_us)
        {
            mango::printLine("    gpu_wait: {:8.2f} ms  (single frame compute submit)",
                r.gpu_wait_us / 1000.0);
            mango::printLine("    note: decode = Huffman + coeff publish; promote = compute fences still open when Huffman returns, plus the sampleable-layout submit");
        }
        if (r.promote_us)
            mango::printLine("    promote: {:8.2f} ms", r.promote_us / 1000.0);
        if (r.present_us)
            mango::printLine("  present: {:8.2f} ms   (swapchain blit check)", r.present_us / 1000.0);
        mango::printLine("  e2e:     {:8.2f} ms   (through present)", r.e2e_us / 1000.0);
    }
    else
    {
        const size_t bytes = r.image.rgba.empty()
            ? size_t(r.image.width) * size_t(r.image.height) * 4
            : r.image.rgba.size();
        mango::printLine("  path:    host RGBA -> intermediate VkImage");
        mango::printLine("  image:   {} x {} (RGBA8, {} KB)",
            r.image.width, r.image.height, bytes / 1024);
        mango::printLine("  READY:   {:8.2f} ms   <-- VkImage usable", r.ready_us / 1000.0);
        mango::printLine("    decode:  {:8.2f} ms", r.decode_us / 1000.0);
        if (r.promote_us)
            mango::printLine("    promote: {:8.2f} ms", r.promote_us / 1000.0);
        if (r.present_us)
            mango::printLine("  present: {:8.2f} ms   (swapchain blit check)", r.present_us / 1000.0);
        mango::printLine("  e2e:     {:8.2f} ms   (through present)", r.e2e_us / 1000.0);
    }

    if (!r.milestones.empty())
    {
        mango::printLine("  milestones (us from job t0):");
        for (const Milestone& m : r.milestones)
        {
            mango::printLine("    {:>10}  {:8} us  ({:7.2f} ms)",
                m.name, m.us_from_start, m.us_from_start / 1000.0);
        }
    }
}

} // namespace gpujpeg

#include "mango_hybrid_backend.hpp"
#include "bench.hpp"

#include <atomic>
#include <vector>

#include <mango/core/core.hpp>

namespace gpujpeg
{
using namespace mango;
using namespace mango::image;
using namespace mango::image::jpeg;

bool mangoHybridEligible(const JpegInfo& info, std::string& reason)
{
    if (!info.valid)
    {
        reason = info.error.empty() ? "invalid JPEG header" : info.error;
        return false;
    }
    if (info.lossless)
    {
        reason = "lossless JPEG (no DCT coefficients)";
        return false;
    }
    if (info.ultrahdr)
    {
        reason = "UltraHDR / gain-map (out of scope for hybrid v1)";
        return false;
    }
    if (info.differential)
    {
        reason = "differential JPEG";
        return false;
    }
    if (info.arithmetic)
    {
        reason = "arithmetic coding (hybrid v1 is Huffman-focused)";
        return false;
    }
    if (info.components < 1 || info.components > 3)
    {
        reason = fmt::format("unsupported component count ({})", info.components);
        return false;
    }
    if (info.precision != 8)
    {
        reason = fmt::format("unsupported precision ({} bit)", info.precision);
        return false;
    }
    return true;
}

MangoHybridHuffmanResult runMangoHybridHuffman(ConstMemory jpeg)
{
    MangoHybridHuffmanResult r;

    JpegDecoder decoder(jpeg);
    const JpegInfo& info = decoder.info();

    std::string reason;
    if (!mangoHybridEligible(info, reason))
    {
        r.error = reason;
        return r;
    }

    r.width = info.width;
    r.height = info.height;
    r.components = info.components;
    r.ymcu = info.ymcu;
    r.progressive = info.progressive;
    r.mango1 = info.mango1;
    r.dri = info.restart_intervals;
    r.chroma = info.chroma;
    r.encoding = info.encoding;

    decoder.setCoeffOrder(CoeffOrder::Spectral);

    const size_t frame_samples = size_t(info.xmcu) * size_t(info.ymcu) * size_t(info.blocks_in_mcu) * 64;
    std::vector<s16> storage(frame_samples);
    decoder.setCoeffStorage(storage.data(), storage.size());

    struct FloorCount
    {
        std::atomic<int>* bands = nullptr;
        std::atomic<std::size_t>* samples = nullptr;
    };
    std::atomic<int> bands { 0 };
    std::atomic<std::size_t> samples { 0 };
    FloorCount floor { &bands, &samples };
    decoder.setCoeffSink([](const CoeffView& view, void* user)
    {
        auto* floor = static_cast<FloorCount*>(user);
        floor->bands->fetch_add(1, std::memory_order_relaxed);
        floor->samples->fetch_add(view.samples, std::memory_order_relaxed);
    }, &floor);

    ImageDecodeOptions options;
    options.multithread = true; // Mango1 offsets or row-aligned DRI → parallel bands

    const u64 t0 = nowUs();
    ImageDecodeStatus status = decoder.decodeCoefficients(options);
    r.huffman_us = nowUs() - t0;

    r.bands = bands.load();
    r.samples = samples.load();

    if (!status)
    {
        r.error = status.info.empty() ? "decodeCoefficients failed" : status.info;
        return r;
    }

    r.ok = true;
    return r;
}

} // namespace gpujpeg

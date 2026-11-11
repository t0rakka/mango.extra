# gpujpeg

Standalone JPEG **GPU ingest workbench** next to mango (`../mango`).  
**Functionality and numbers live here first.** Later: migrate to a mango plugin (most likely) or a separate library (less likely). Do not block features on mango packaging.

This README is the handoff context for continuing on other machines, and the roadmap for the GPU JPEG decoder. Read it before changing decode paths.

---

## Roadmap

**Why.** Almost nobody ships a JPEG path that is both fast and pleasant to call: file in, sampleable GPU texture out, on more than one graphics API. That is a thing people already want. This workbench is where it gets proven. Ease of use matters: a fast path that is painful to call will not be used.

It sits next to two neighbors that solve a different problem. Basis Universal supercompression (already in mango) is for authored GPU block textures. DirectStorage is for feeding the GPU from disk with minimal CPU. JPEG is the photographic and camera case those two do not cover. The long-term plan is to fuse GPU JPEG decode, Basis, and DirectStorage into one coherent loading system that is fun to use and fast. They stay separate until each one is good. Mango stays the CPU JPEG library. This workbench owns the GPU backends until the API settles, then it can move into a mango plugin.

**Now.** Latency of one file. File-at-a-time is enough. `READY` is the metric: from submit until the intermediate image is usable (fence signaled, layout sampleable). The swapchain blit is only a visual check.

**Graphics API.** Vulkan is the API we test with. It is flexible enough to stand in for the others while the decode paths settle. The product is not Vulkan-only. The same READY contract is meant for D3D11, D3D12, and OpenGL. Keep the public decode call free of Vulkan types so those backends can implement it. Shader and queue details stay inside the backend. Do not start the D3D or GL ports until the Vulkan path's one-file latency is in good shape.

**Later.** Throughput of a batch (10+ files) as fast as possible. That is the primary end goal. Pipelining is per backend and waits until the bench is stressing batches. A 128-thread CPU gains from more than one file in flight because one JPEG cannot fill the pool. nvJPEG and others may want their own depth for the same reason. Do not build that scheduler yet.

### Object lifetimes

- **GPU JPEG decoder** (the workbench backend): create once, keep it. It owns compiled shaders, pipelines, and sessions. `create()` compiles those pipelines so the first texture in a scene is not the one that pays for shader compile. Destroying a per-file object must not throw the pipelines away.
- **`mango::image::jpeg::JpegDecoder`**: one per file. Scan state is per image, and entropy views alias the caller's bytes. Reusing one parser across files does not work. The decoder creates parsers as it needs them.
- The caller can free the file bytes when `decode()` returns. The backend copies the bitstream or the coefficients into its own storage before that.

### Call shape

```text
decoder.decode(file, destination, on_ready)
```

Today the caller waits for `on_ready` before the next file. The same call can later be issued again while earlier files are still running. Each backend decides how many it overlaps. Completion may be out of order, so the callback identifies the task. The callback runs after the fence, on the graphics thread, and it stays short: it must not wait on another decode.

Submit stays on one graphics thread (the thread that owns the Vulkan queue today; the same rule for the D3D and GL device later). Caller threads only enqueue. If the application already owns that queue, the frame thread pumps the decoder. A second thread submitting on the same queue races the renderer.

One completion contract: sampleable RGBA. nvJPEG YUV resolve and Apple BGRA stay inside the backend. Errors use the same callback.

In-flight work stays bounded. Each task needs a destination image and, on the hybrid path, a full-frame coefficient buffer. The caller either passes an image that lives until the callback, or returns a decoder-owned slot when finished with it.

### Backend choice (later)

Not implemented. Still collecting timings per machine (`result.txt`). The decoder will own every backend the device can run, and each file picks among the ones that can handle it. Lossless, progressive, or arithmetic files simply drop the backends that cannot decode them. If none remain, the same callback still delivers a texture via the CPU decoder. A static rank such as NVIDIA, then Apple VT, then AMD is only a sketch. Per-file shape (restart vs one serial scan) changes who wins. Do not hard-wire a rank until the tables say so.

Size-specific state (VideoToolbox session, nvJPEG buffers) is recreated when dimensions change. Compute pipelines are not.

### What is already true in code

- `JpegDecoder` is per file: `decodePixels`, `decodeCoefficients` (`CoeffOrder::Spectral`, `CoeffSink`, `setCoeffStorage`), `extractEntropy`.
- `MangoGpuDecoder` and `MangoHybridCompute` still keep their pipelines on the per-image object. A new object recompiles. Lifting those pipelines onto a long-lived decoder is the next API step. Do not paper over that by telling callers to reuse a one-shot decoder: `MangoGpuDecoder::upload()` accepts one bitstream, and its descriptors stay bound to that image.
- Hybrid: Huffman workers never wait on the GPU, on a free buffer, or on each other. Segments may finish in any order. The graphics thread alone submits and waits on fences. Coefficient storage for the whole frame is allocated before the workers start.
- mango-gpu Huffman is one workgroup per entropy segment, 64 lanes, one submit. Shifts of 32 or more are rejected (AMD masks the count to 5 bits and would otherwise accept a bad speculative cursor). Writes stop at the segment coefficient count so restart pad bits do not spill into the next MCU row.

### Do not

- Put a ring, `waitSlot`, or a GPU fence wait on the Huffman path.
- D2H the image and upload it again on a GPU path.
- Treat present/e2e as the ingest number. `READY` is the number.
- Reintroduce timeline or binary semaphore chaining between bands (MoltenVK stall).

---

## Support matrix (roadmap)

```text
Sources                              Outputs
───────────────────────────────      ──────────────────
MANGO         CPU full (fallback)    Vulkan   (done in workbench)
MANGO_HYBRID  CPU Huffman → GPU      OpenGL   (roadmap)
              compute IDCT+rest      D3D11    (roadmap)
NVJPEG        NVIDIA / CUDA (done)   D3D12    (roadmap)
ROC           AMD rocJPEG (VK, SDK
              + runtime gate)
(+ Intel / … later)
```

**Available** = CMake option ON ∩ SDK found at configure ∩ runtime device present.  
`Auto` is not implemented. Ranking is a later task; see the roadmap. Do not treat any fixed order as the decision.

CMake axes (see `CMakeLists.txt`): `GPUJPEG_SOURCE_*`, `GPUJPEG_OUTPUT_*`.

---

## What already works (NVIDIA box)

- **NVJPEG → Vulkan:** decode to device RGBI, NPP → CUDA-exported RGBA buffer, `vkCmdCopyBufferToImage` into intermediate `VkImage`, optional blit to swapchain.
- Modes: `--nvjpeg-mode=hybrid` (CPU Huffman) | `gpu` (GPU-assisted Huffman).
- Warmup, `--inflight`, `--pipeline` overlapped **READY** ring (N VkImages).
- Mango CPU path: SIMD decode into cached RGBA, one streaming copy into staging, then DMA into the VkImage. `--pipeline` with `--inflight>=2` overlaps the next decode with the previous DMA.
- Helmet-class results (ballpark): mango READY ~8–10 ms multi-thread CPU; nvJPEG READY ~3 ms; ready-pipe with inflight=4 a bit faster still.

Hybrid nvJPEG moves **s16 DCT coeffs** CPU→GPU then IDCT on device (~2× u8 plane traffic vs final samples). GPU Huffman avoids that expansion on the host bus.

---

## ROC path (AMD)

**rocJPEG → Vulkan**, same READY contract as nvJPEG. Bitstream stays on the host; pixels do not.

| Mode | rocJPEG backend | What it is |
|------|-----------------|------------|
| `--roc-mode=hardware` | `ROCJPEG_BACKEND_HARDWARE` | VCN JPEG ASIC (the media engine) |
| `--roc-mode=hybrid` | `ROCJPEG_BACKEND_HYBRID` | Declared by the SDK. rocJPEG 1.4 returns not-implemented; the workbench skips it |

Two gates, both required:

1. **Configure.** `GPUJPEG_SOURCE_ROC` defaults ON. CMake looks for `rocjpeg/rocjpeg.h`, `librocjpeg`, `hip/hip_runtime.h`, `libamdhip64`, and `libhiprtc` under `$ROCM_PATH` or `/opt/rocm`. Any miss → ROC stays off, the rest of the workbench still builds.
2. **Runtime.** `rocJpegCreate` must succeed on the HIP device matched to the Vulkan device (UUID, else product name / `gfx` arch). A missing VCN JPEG engine, or an arch the library refuses, drops that mode.

Decode output is interleaved RGB in HIP memory. A small hiprtc kernel expands it to RGBA in a Vulkan-exported device buffer, then `vkCmdCopyBufferToImage`. No D2H.

rocJPEG 1.4 decode is one blocking VCN submit. A batch or a second thread does not add a second JPEG core on this GPU, so `--pipeline` still decodes one frame at a time. The RGBA copy of frame N is submitted before decode of frame N+1 so that DMA hides inside the next VCN job. `--inflight` sizes the RGB slots for that handoff.

```bash
# Arch
sudo pacman -S rocjpeg    # pulls hip-runtime-amd, rocminfo, libva, mesa

cmake -S . -B build -DGPUJPEG_SOURCE_ROC=ON
cmake --build build
build/gpujpeg --backend=roc --roc-mode=both --warmup=2 --iterations=5 photo.jpg
```

The published rocJPEG architecture table stops at gfx1102 (Navi3). RDNA4 (gfx1201, RX 9070 XT) still has a VCN JPEG block — Mesa VA-API exposes `VAProfileJPEGBaseline` on this GPU. Whether rocJPEG's userspace will submit to it is the runtime `rocJpegCreate` check, not the table.

---

## MANGO_HYBRID

Uses public `mango::image::jpeg::JpegDecoder` (`CoeffOrder::Spectral` + `CoeffSink`)
plus a Vulkan compute pass (IDCT + YCbCr→RGBA). Bands are whatever `decodeCoefficients` publishes; the graphics thread submits each one. There is no slot ring and no Huffman backpressure.
Spectral coefficients stay s16 on the way to the GPU; the shader sign-extends them.

| Step | Status |
|------|--------|
| 1. Spectral Huffman floor | **done** — `--hybrid-floor`; READY = Huffman time |
| 2. Compute IDCT+YCbCr→VkImage | **done** (VK) — `--backend=hybrid` / `all` |

```bat
build\gpujpeg.exe --backend=hybrid --warmup=2 --iterations=5 photo.jpg
build\gpujpeg.exe --backend=hybrid --hybrid-floor photo.jpg
build\gpujpeg.exe --backend=all photo.jpg
```

---

## Build / run (current NVIDIA workbench)

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake ^
  -DMANGO_ROOT=../mango -DMANGO_BUILD=../mango/build
cmake --build build

build\gpujpeg.exe --backend=nvjpeg --nvjpeg-mode=both --warmup=2 --iterations=3 photo.jpg
build\gpujpeg.exe --backend=nvjpeg --nvjpeg-mode=gpu --inflight=4 --pipeline=32 --iterations=1 photo.jpg
```

| Flag | Meaning |
|------|---------|
| `--backend` | `mango` \| `hybrid` \| `nvjpeg` \| `roc` \| `both` \| `all` |
| `--roc-mode` | `hardware` \| `hybrid` \| `both` (ROC only) |
| `--hybrid-floor` | Hybrid Huffman-only (no VK compute) |
| `--nvjpeg-mode` | `hybrid` \| `gpu` \| `both` |
| `--warmup=N` | Discard N decodes (boot CUDA) |
| `--inflight=N` | Overlapping streams / READY ring depth |
| `--pipeline=N` | Overlapped READY throughput frames (0=off) |

---

## Apple VideoToolbox → IOSurface → VkImage (`apple-vt`)

**macOS + `GPUJPEG_SOURCE_APPLE_VT=ON` (default on Apple).** Baseline JPEG via `VTDecompressionSession` into **IOSurface-backed BGRA**, then **`VK_EXT_metal_objects`** import — no host RGBA staging.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMANGO_ROOT=../mango
cmake --build build
build/gpujpeg --backend=apple-vt --warmup=2 --iterations=5 photo.jpg
build/gpujpeg --backend=all photo.jpg   # mango / hybrid / apple-vt on Mac
```

| Stage | Meaning |
|-------|---------|
| **decode** | VideoToolbox (HW when `hw=yes`) |
| **promote** | `vkCreateImage` import + layout → `SHADER_READ_ONLY` |
| **READY** | decode + promote |

v1 limits: **baseline 8-bit Huffman** only (same class as hybrid eligibility); progressive/arithmetic → use mango. First frame can be slow (session/window); use **`--warmup`**.

**M4 ballpark (3888×2592, after warmup):** READY ~**16 ms**, decode ~**14 ms**, promote ~**0.06 ms** — vs mango ~**15.5 ms** READY with ~**9.5 ms** in host **promote**. VT does not beat mango **decode** on this asset yet; **promote** is essentially free.

---

## Apple Silicon (MoltenVK) — probe notes

Early numbers on **M4** (one camera JPEG, sequential bench, not `--pipeline`). Treat as relative, not ceiling.

| backend | READY | decode | promote | path |
|---------|-------|--------|---------|------|
| mango | ~15.5 ms | ~3.7 ms | ~9.5 ms | host RGBA → VkImage |
| mango-hybrid | ~25 ms | ~23 ms | 0 | file → VkImage (compute) |

**How to read hybrid:** the `decode` column is `huffman_us` from `runMangoHybridCompute` — wall time of `decodeCoefficients` **including** the coeff sink (`memcpy` to upload buffers + record/submit compute per band), not Huffman alone. Only ~2 ms of the READY gap is `finish()` / final fence; the rest is “parse + ship coeffs” on CPU/MVK.

On this box **full mango NEON decode (~3.7 ms) beats stopping at coefficients (~23 ms)** before GPU IDCT even matters — the hybrid split assumes CPU IDCT is the expensive part (true on some desktops; **false here**). **`--hybrid-floor`** isolates Spectral Huffman without VkImage if you want that line item.

**promote (~9.5 ms)** on the mango path is host-visible staging + `vkCmdCopyBufferToImage` under MoltenVK — real cost to fold into any “Apple HW → texture” comparison.

**Next Mac ingest bet:** VideoToolbox HW JPEG → IOSurface → `VkImage` (`VK_EXT_metal_objects`), with mango / hybrid as fallback. nvJPEG/rocJPEG N/A.

---

## Agent checklist

1. Read the roadmap at the top of this file before changing a decode path. Then `src/backends.hpp`, the backend you are in, `decoded_texture.*`, and `main.cpp`.
2. Current stage is one-file latency (`READY`). Batch throughput and auto backend selection are later. The API call shape should still allow a second `decode()` before the previous callback.
3. Next structural step: a long-lived GPU JPEG decoder that keeps compiled pipelines, and a new `JpegDecoder` per file.
4. ROC on gfx1201 (RX 9070 XT) is outside AMD's published JPEG-core table. A successful `rocJpegCreate` on that GPU is not evidence of a JPEG engine. Confirm on device before trusting the number.
5. Stay stand-alone until the API settles. A mango plugin is a later migration. OpenGL / D3D outputs share the same READY contract when they land.

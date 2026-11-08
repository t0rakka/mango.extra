# gpujpeg

Standalone JPEG **GPU ingest workbench** next to mango (`../mango`).  
**Functionality and numbers live here first.** Later: migrate to a mango plugin (most likely) or a separate library (less likely). Do not block features on mango packaging.

This README is the handoff context for continuing on other machines (e.g. AMD / ROC).

---

## Design in one page

**Goal:** drip JPEG bitstreams → **intermediate GPU texture ready to sample/compute** (`READY`). Swapchain blit is only a visual check (vsync-sensitive); it is not the ingest metric.

**No host bounce on the GPU path.** Do not D2H then re-upload. File → device decode → promote into an intermediate image the client owns.

**API shape (roadmap, not fully coded):** fire-and-forget `submit` into an internal queue; fixed **inflight** GPU slots drain the queue; optional **completion callback** (capture input buffer → free when that job is done). Client controls priority by submit order. Sync one-at-a-time is already a clear win vs CPU; inflight squeezes more hardware.

**Primary metric: `READY`** = start → intermediate image usable. Present/e2e = “finished on screen” only.

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
`Auto` preference sketch: `NvJpeg → Roc → MangoHybrid → Mango`.

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
plus a Vulkan compute pass (IDCT + YCbCr→RGBA) with a 3-slot band ring.
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

## Agent checklist (AMD / next steps)

1. Read this README + `src/backends.hpp`, `nvjpeg_backend.*`, `decoded_texture.*`, `main.cpp`.
2. ROC source → Vulkan is in. Next: confirm `rocJpegCreate` on the Vulkan device (gfx1201 is not in AMD's published table).
3. Port hybrid compute to OpenGL / D3D when those outputs land (VK path is reference).
4. Keep output roadmap (GL/D3D) as plugins sharing the same READY/queue design.
5. Stay stand-alone until the API settles; mango plugin is a later migration.

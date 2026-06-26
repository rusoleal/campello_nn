# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.3.0] - 2026-06-26

### Fixed
- `GpuBackend::compileGraph()` no longer passes a null pointer to `campello_gpu`'s data-carrying
  `createBuffer()` overload for zero-byte constants (dead ONNX initializers). This caused a crash
  during import of real models such as YuNet on `DeviceType::GpuGeneric`.

### Added
- `tests/platform/test_gpu_generic_models.cpp`: end-to-end YuNet face-detection test running on the
  generic `campello_gpu` backend (`DeviceType::GpuGeneric`), mirroring the existing CPU/MPSGraph
  model tests.
- `benchmarks/benchmark_backends.cpp` and `BUILD_BENCHMARKS` CMake option: compares latency of
  `Cpu`, `GpuGeneric`, and native `Gpu` backends on the same transformer-block graph and on the
  real YuNet face-detection model, reporting min/median/mean/max and max absolute/score
  difference against the CPU reference.

### Changed
- Bumped `campello_gpu` dependency from `v0.13.2` to `v0.14.0`; removed the temporary
  `CAMPELLO_NN_CAMPELLO_GPU_LOCAL_DIR` local-checkout override from `CMakeLists.txt` now that the
  required Metal fence/download fixes are in the released tag.
- `GpuGeneric` matmul now uses 1D column tiling (8 columns per workgroup), improving the
  transformer-block benchmark latency on macOS/Metal by ~1.7×.
- `GpuGeneric` conv2d now uses 1D output tiling (8 flattened output elements per workgroup),
  improving the YuNet benchmark latency on macOS/Metal by ~3.9×.
- `GpuGeneric` conv2d now uses an `im2col + GEMM` path for `groups == 1` convolutions with
  small output width (`outW <= tileWidth / 4`), where the direct-convolution shader has poor
  thread utilization. On macOS/Metal (Intel UHD 630) this drops YuNet latency from ~617 ms to
  ~199 ms (~3.1×); ResNet-50 stays neutral at ~1.15 s vs. ~1.10 s before.
- `GpuGeneric` now fuses `Conv2d -> Add[bias] -> ReLU` into a single dispatch when the pattern is
  detected. This eliminates two dispatches and their bind-group builds per fused block. The pattern
  matches 18 Conv blocks in YuNet but does not match the exported ResNet-50 v1-7 (Conv→BatchNorm→
  ReLU), so YuNet stays at ~199 ms and ResNet-50 stays at ~1.15 s on macOS/Metal.
- `GpuGeneric` now caches `cgpu::BindGroup` objects across `dispatch()` calls in a per-graph cache
  keyed by bind-group layout and buffer bindings. This removes per-dispatch bind-group recreation
  overhead, but on macOS/Metal (Intel UHD 630) the latency change is within noise, confirming that
  the remaining gap is shader execution time rather than host-side dispatch overhead.
- `GpuGeneric` now folds inference-time BatchNorm into the preceding Conv2d and fuses the whole
  `Conv → BatchNorm → ReLU` block into a single dispatch. ResNet-50 v1-7 latency on macOS/Metal
  (Intel UHD 630) improved from ~1.156 s to ~1.098 s (~5%); YuNet and the transformer block are
  unchanged.
- `GpuGeneric` matmul and conv2d now size their tiles to the backend's actual workgroup width
  via the new `campello_gpu::ComputePipeline::getWorkgroupSize()` query, eliminating idle threads
  on Metal. On macOS/Metal (Intel UHD 630) this makes `GpuGeneric` faster than the CPU backend on
  YuNet for the first time (~675 ms CPU vs. ~654 ms `GpuGeneric`).
- `GpuGeneric` now detects `Conv2d -> Add[bias] -> Relu/Sigmoid` patterns and aliases the
  elementwise outputs to the Conv2d output buffer in-place, avoiding intermediate buffer
  allocations and memory round-trips. YuNet benchmark latency on macOS/Metal improved from
  ~697 ms to ~673 ms (~3.5%).
- `GpuGeneric` conv2d dispatch switched from a flattened 1D output grid to a 2D spatial-tile
  grid (`dispatchX = tileColsPerRow * N * O`, `dispatchY = outH`) and the shader was rewritten
  with a shared-memory tiled path behind a `USE_SHARED_MEMORY` toggle. The shared-memory path
  is currently disabled by default on Metal/Vulkan/DirectX because it regressed YuNet latency
  on Intel integrated graphics (likely barrier/smem overhead on unified memory). With the
  shared path off, the 2D dispatch still improved YuNet latency on macOS/Metal from ~673 ms
  to ~617 ms (~8%). The shared-memory implementation is left in place for re-enablement/tuning
  on discrete GPUs.
- **ONNX importer:** added support for `Flatten`, `GlobalAveragePool`, and `Gemm` with
  `transA`/`transB` (inserting explicit `transpose()` ops when needed). This enables importing
  standard ResNet-style image-classification models.
- **Benchmark:** added a third workload, ResNet-50 v1 image classification (224×224 ONNX,
  ImageNet preprocessing), alongside the existing transformer-block and YuNet workloads.
  On macOS/Metal (Intel UHD 630): CPU ~4.3 s, `GpuGeneric` ~1.1 s (~3.8× faster than CPU),
  MPSGraph ~185 ms.

---

## [0.2.0] - 2026-06-23

### Added
- `GraphInfo`/`NodeInfo`/`OpKind` (`inc/campello_nn/graph_info.hpp`): a public, display-safe
  mirror of the internal graph IR (`src/pi/ir.hpp`) for callers that need to inspect or visualize
  an imported graph's topology — op list, per-node shape/dtype/attributes, edges — without needing
  execution access to it. Built for `campello_editor`'s ONNX/TFLite graph viewer, but general-purpose.
- `OnnxImportResult::info` / `TfliteImportResult::info`: both importers now populate a `GraphInfo`
  alongside the compiled `Graph`, describing the graph exactly as built (not a re-derivation).
- `OpKind` is now declared in the public `graph_info.hpp` rather than only `ir.hpp` — a single
  definition shared by the internal IR and the public introspection API, instead of two enums
  that would need to stay in sync.

### Notes
- Purely additive: existing `OnnxImportResult`/`TfliteImportResult`/`Graph` usage is unaffected.
  `NodeInfo::constantByteSize` deliberately omits the actual constant bytes (a real model's
  weights can be hundreds of megabytes) — only the byte count is exposed for display purposes.

---

## [0.1.0] - 2026-06-22

### Added
- Initial release: WebNN-shaped compute-graph API (`Context`, `GraphBuilder`, `Operand`, `Graph`,
  `Tensor`, `Fence`).
- CPU reference backend (Float32, Float16, Int8 quantization).
- MPSGraph backend (macOS/iOS).
- ONNX import (`importOnnxFromFile`/`importOnnxFromMemory`), validated end-to-end against a real
  pretrained model (YuNet face detection) on real photos.
- TFLite import (`importTfliteFromFile`/`importTfliteFromMemory`), validated against MediaPipe
  BlazeFace.
- Graph caching: `GraphBuilder::serialize()`/`deserialize()` and `graph_cache.hpp`'s
  `saveGraphToFile`/`loadGraphFromFile`/`loadGraphFromMemory`.

[Unreleased]: https://github.com/rusoleal/campello_nn/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/rusoleal/campello_nn/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rusoleal/campello_nn/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rusoleal/campello_nn/releases/tag/v0.1.0

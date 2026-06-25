# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
- `GpuGeneric` matmul now uses 1D column tiling (8 columns per workgroup), improving the
  transformer-block benchmark latency on macOS/Metal by ~1.7×.
- `GpuGeneric` conv2d now uses 1D output tiling (8 flattened output elements per workgroup),
  improving the YuNet benchmark latency on macOS/Metal by ~3.9×.
- `GpuGeneric` matmul and conv2d now size their tiles to the backend's actual workgroup width
  via the new `campello_gpu::ComputePipeline::getWorkgroupSize()` query, eliminating idle threads
  on Metal. On macOS/Metal (Intel UHD 630) this makes `GpuGeneric` faster than the CPU backend on
  YuNet for the first time (~675 ms CPU vs. ~654 ms `GpuGeneric`).

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

[Unreleased]: https://github.com/rusoleal/campello_nn/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rusoleal/campello_nn/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rusoleal/campello_nn/releases/tag/v0.1.0

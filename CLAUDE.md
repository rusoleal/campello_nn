# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure

# macOS/iOS only — also exercises the real MPSGraph backend on GPU hardware:
cmake -B build -DBUILD_TESTS=ON -DBUILD_INTEGRATION_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Platform-specific CMake files are selected based on target: `macos.cmake`, `ios.cmake`,
`windows.cmake`, `linux.cmake`, `android.cmake`, `wasm.cmake`. All of them build the
cross-platform core sources (`CAMPELLO_NN_CORE_SOURCES`, set in the top-level
`CMakeLists.txt`). `macos.cmake`/`ios.cmake` additionally compile `src/metal/mps_backend.mm`
(Objective-C++, `enable_language(OBJCXX)`, `-fobjc-arc`) and link the Metal/MPSGraph
frameworks — the only accelerator backend implemented so far (see `TODO.md` Phase 3a).
Windows/Linux/Android/Web still build CPU-only.

Tests live under `tests/universal/` (CPU backend, no SDK required, run via GoogleTest/CTest)
and `tests/platform/` (accelerator-backend integration tests — currently `test_mps_ops.cpp`,
gated behind `BUILD_INTEGRATION_TESTS` and `if(APPLE)` in `tests/CMakeLists.txt`).

## Architecture Overview

`campello_nn` is a WebNN-shaped compute-graph library (C++20). It knows tensors and ops, and —
per a later scope decision recorded in `NN_ARCHITECTURE.md` §3/§5 — also imports graph-format
model files directly. ONNX import is implemented (`inc/campello_nn/onnx_importer.hpp`,
`importOnnxFromMemory(context, data, size)`, with `importOnnxFromFile(context, path)` as a
convenience wrapper — split the same way `campello_image::Image` splits `fromMemory`/`fromFile`,
so callers without a real filesystem path (Android `AAssetManager`, etc.) aren't forced through
one). TFLite is a planned follow-up. Graph caching is implemented (see "Graph Caching" below):
the IR `GraphBuilder::build()` would otherwise hand straight to `Backend::compileGraph()` can
instead be serialized to bytes and reloaded later, skipping `GraphBuilder` reconstruction. The
planned `campello_llm` layer (tokenization,
weight-only-format loading for known LLM architectures, KV-cache, sampling) is a separate,
not-yet-implemented project that sits on top for models that *don't* ship as ONNX/TFLite; a future
`campello_vision` would play the same role for weight-only-format vision models. All public types
in this repo live in the `systems::leal::campello_nn` namespace, matching the sibling
`campello_gpu` repo's convention.

### Handle-Based Abstraction Pattern

Public API classes (`Context`, `Tensor`, `Graph`, `Fence`) hold `void*` pointers to opaque
internal structs — never exposed through public headers. This mirrors `campello_gpu`'s
`Device`/`Buffer` pattern and keeps backend-specific types (and the internal graph IR) out of
`inc/campello_nn/`.

- Public headers: `inc/campello_nn/*.hpp`, with `constants/` and `descriptors/` subdirs.
- Internal graph IR: `src/pi/ir.hpp` (`OpKind`, `Node`, `GraphIR`) — built by `GraphBuilder`,
  consumed by `Backend::compileGraph()`. Never exposed publicly.
- Internal `Backend` interface: `src/pi/backend.hpp`. Every accelerator backend implements this;
  `Context`/`GraphBuilder` delegate to it through `void*` handles.
- `Context`/`Tensor`/`Graph`/`Fence` implementations: `src/pi/*.cpp`.
- CPU reference backend: `src/cpu/` — Float32 graph interpreter, no SDK dependency. This is what
  every platform builds against (selected for `DeviceType::Cpu`), and what `tests/universal/`
  exercises. "Float32 graph interpreter" because Float16 tensors are decoded to Float32 at the
  Input/Constant boundary and re-encoded only at the final output (`cpu_backend.cpp::dispatch`) —
  see "Dtype Support" below; every kernel in `ops.cpp` only ever sees Float32.
- MPSGraph backend (macOS/iOS): `src/metal/mps_backend.{hpp,mm}` — selected for
  `DeviceType::Gpu`/`Npu`/`Default` on Apple platforms. **MPSGraph has no Apple-provided C++
  binding** (unlike Metal/metal-cpp, which `campello_gpu` uses), so this is genuine
  Objective-C++: real `MPSGraph`/`MPSGraphTensor`/`MTLBuffer` calls, ARC-managed. `mps_backend.hpp`
  stays pure C++ (a pimpl'd `Impl`) so `context.cpp` (a plain `.cpp`) can include it without
  itself needing Objective-C++ compilation. Tensors use `MTLResourceStorageModeShared` so
  `write`/`read` are plain `memcpy` against `buffer.contents`. `compileGraph()` builds an
  `MPSGraph*` once by walking the IR (mapping each `OpKind` to real graph ops — `gelu` and `gemm`
  have no single native MPSGraph op, so they're composed from primitives); `dispatch()` calls
  `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` synchronously, so its `Fence` is
  always pre-signaled, same as the CPU backend.

### Dtype Support

All ops support both Float32 and Float16 (CPU: decode-at-boundary, see above; MPSGraph: native).
`gather`'s `indices` operand accepts Int32 or Uint32. Int8 is real quantization, not just
storage: `quantizeLinear`/`dequantizeLinear` (per-tensor scale/zero-point, matching ONNX's ops
and MPSGraph's `quantizeTensor:`/`dequantizeTensor:` scalar overloads) and `quantizedMatmul`
(weight-only quantization — dequantize then `matmul`, since neither backend has a dedicated
int8×int8 GEMM kernel). Per-channel scale (MPSGraph's `scaleTensor:` overloads) isn't wired up.

### ONNX Import

`src/onnx/` — internal, not exposed via any public header except the top-level
`inc/campello_nn/onnx_importer.hpp`. Three pieces:

- `proto_reader.hpp`: hand-rolled minimal protobuf wire-format reader (varint/tag/
  length-delimited primitives only — no `protoc`, no full `protobuf` dependency). Every ONNX
  message field number it's used to read was verified against a real `.onnx` file generated by
  Python's `onnx` package before being trusted.
- `onnx_model.hpp`/`onnx_parser.cpp`: parses an ONNX `ModelProto` into plain structs
  (`OnnxGraph`/`OnnxNode`/`OnnxTensor`/`OnnxValueInfo`). Dynamic/symbolic dimensions default to 1
  (single-item inference). `OnnxTensor::toInt64Vector()` reads int64 shape/scale constants (e.g.
  `Reshape`'s second input) without needing an `Int64` `campello_nn::DataType`, which doesn't
  exist; `onnx::onnxElemTypeHasDataType()` lets the importer skip binding initializers that are
  only ever import-time metadata (same reason).
- `onnx_importer.cpp`: the op-mapping table — `Conv`→`conv2d` (including a fused bias: reshaped to
  `[1,C,1,1]` and added via `add()`'s NumPy-style broadcasting — this turned out to be the
  *universal* case, not an edge case, once tested against a real model),
  `Add`→`add`, `Relu`→`relu`, `Transpose`→`transpose`, `MaxPool`/`AveragePool`→`maxPool2d`/
  `avgPool2d`, `Reshape`→`reshape` (resolves `0`/`-1` via `internal::operandShapeForImport`),
  `Resize`→`resize`, `BatchNormalization`→`batchNorm` (mind the input-order mismatch — ONNX is
  `X,scale,B,mean,var`; ours is `x,mean,variance,scale,bias`), etc. Remaining unsupported patterns
  (`Gemm` with transA/transB, multi-output nodes, a non-constant `Reshape`/`Resize` shape input,
  unmapped op types) throw rather than guess. `importOnnxFromMemory()` is the real implementation;
  `importOnnxFromFile()` just reads the file into a buffer and calls it — don't add file-specific
  logic to the memory path.
- `internal::operandShapeForImport(const Operand&)` (declared in the public `operand.hpp`, inside
  an `internal` namespace; implemented in `graph_builder.cpp`): the importer's only window into an
  intermediate tensor's actual inferred shape (needed by `Reshape`'s `-1`, `Resize`'s scales, and
  `Conv`'s bias broadcast). Returns a shape *copy* — never hold a reference, the IR node vector
  can reallocate on the next `GraphBuilder` call.

Test fixtures:
- `tests/fixtures/conv_add_relu.onnx` — synthetic, validated with `onnx.checker.check_model`,
  regenerable via `tests/fixtures/generate_conv_add_relu_onnx.py`.
- `tests/fixtures/yunet_n_320_320.onnx` + `tests/fixtures/images/{face,no_face}.jpg` — a real
  face-detection model (YuNet, BSD-3-Clause) and two real test images (provenance in
  `tests/fixtures/NOTICE.md` / `tests/fixtures/images/NOTICE.md`), exercised by
  `tests/universal/test_yunet_face_detection.cpp`. That test decodes images via `campello_image`
  (`FetchContent`, gated behind its own `BUILD_MODEL_TESTS` CMake option since it pulls in
  `basis_universal`+`libwebp` — don't fold it into the default `BUILD_TESTS` path) and resizes
  them via campello_nn's own `resize` op rather than separate image-resize code. The model expects
  BGR (not RGB), raw `[0,255]` pixel values, no normalization — see the test file's comments for
  exactly where that convention was confirmed (`libfacedetection.train`'s own training transform).
  Note: `campello_image` declares its own `option(BUILD_TESTS ...)`, which collides by name with
  this project's — building both options together also builds `campello_image`'s own test suite
  as a harmless side effect.

### Graph Caching

`GraphBuilder::build()` normally hands its IR straight to `Backend::compileGraph()` and discards
it. `GraphBuilder::serialize(outputs)` (instance method, same signature as `build()`) instead
serializes that IR to bytes via `src/pi/ir_serialization.{hpp,cpp}` (internal) — an explicit
field-by-field binary writer/reader (magic `"CNNG"` + version, not a struct memcpy, for
portability), covering every `Node` field including `Conv2dDescriptor`/`Pool2dDescriptor`/
`ResizeDescriptor`. `GraphBuilder::deserialize(context, data, size)` (static) compiles a graph
directly from those bytes, skipping `GraphBuilder` reconstruction (op-by-op calls, or re-parsing a
source model file like ONNX) — `deserializeGraphIR()` throws on a bad magic, unsupported version,
or truncated buffer (every read is bounds-checked).

**Decision:** this caches the backend-agnostic IR, not a backend-compiled native object (no cached
`MPSGraph*`, etc.) — `compileGraph()` still runs on every load. Simpler, and one cache file works
against any backend; only worth revisiting if `compileGraph()` itself becomes the bottleneck for
some backend.

Public API (`inc/campello_nn/graph_cache.hpp`) mirrors the ONNX importer's shape: `GraphCacheResult`
(`{graph, inputs, outputs}`, same as `OnnxImportResult`), `loadGraphFromMemory`/`loadGraphFromFile`
(same Memory/File split as `importOnnxFromMemory`/`importOnnxFromFile`), and `saveGraphToFile`.
`inputs`/`outputs` descriptors come from walking the deserialized IR's `Input` nodes and `outputs`
list — this means the IR gets deserialized twice on a cache load (once for descriptors, once inside
`GraphBuilder::deserialize()` for compilation); accepted as a one-time cache-load cost rather than
plumbing a `GraphIR` through a public-header-safe API. Not yet wired into the ONNX importer itself
(`importOnnxFromFile` doesn't transparently check for a cached sibling file) — left as future
integration work.

### Dispatch Model — Fence, not Future

`NN_ARCHITECTURE.md` originally sketched `Future<void> dispatch(...)`, but this repo follows
`campello_gpu`'s synchronous-submit-plus-fence convention instead (`Context::dispatch()` returns
a `Fence` you `wait()`/`isSignaled()` on) to keep the two libraries' async story consistent. The
CPU backend's `Fence` is always pre-signaled since CPU dispatch is synchronous.

### Graph Lifecycle

```
Context::create(desc) → Context
GraphBuilder(context) → builder.input()/constant()/add()/matmul()/... → Operand
builder.build({{"name", operand}, ...}) → Graph        // compiled once
context->createTensor(desc) → Tensor                    // bind by name
context->dispatch(graph, inputs, outputs) → Fence       // dispatched many times
fence->wait(); output->read(...)
```

### Key Types

| Category | Types |
|---|---|
| Initialization | `Context` |
| Graph construction | `GraphBuilder`, `Operand`, `Graph` |
| Resources | `Tensor`, `Fence` |
| Descriptors | `TensorDescriptor`, `ContextDescriptor` |
| Constants | `DeviceType`, `DataType` |

### Op Set

`GraphBuilder` started as exactly the ops needed for a transformer block: `add`, `mul`, `gelu`,
`softmax`, `layerNorm`, `matmul`, `gemm`, `reshape`, `transpose`, `concat`, `slice`, `gather`.
`relu`/`sigmoid` were added during ONNX-import work (real vision models use `Relu`, not `Gelu`).
Project scope has since grown to also cover vision/multimodal models (not in the original
`NN_ARCHITECTURE.md` — see that doc's "Scope update" note), adding `conv2d`, `maxPool2d`,
`avgPool2d` (NCHW input, OIHW conv weights, explicit padding — see `Conv2dDescriptor`/
`Pool2dDescriptor`), `resize` (nearest/bilinear — see `ResizeDescriptor`), and `batchNorm`/
`instanceNorm` (kept as two distinct ops matching ONNX/PyTorch's `BatchNorm2d`/`InstanceNorm2d`
exactly — `batchNorm` takes mean/variance as given inputs, `instanceNorm` computes them from `x`
over the spatial axes; both map to the same `normalizationWithTensor:` MPSGraph call `layerNorm`
uses), plus `quantizeLinear`/`dequantizeLinear`/`quantizedMatmul` (see "Dtype Support" below).
Shape/dtype inference and validation happen at build time in
`src/pi/graph_builder.cpp` (e.g. mismatched `matmul` dimensions throw `std::runtime_error` before
reaching a backend).

`add`/`mul` support NumPy/ONNX-style broadcasting (shapes aligned from the right; size-1 or
missing leading dims broadcast) — `computeBroadcastShape()` in `graph_builder.cpp` infers the
output shape and throws on genuinely incompatible shapes. CPU backend implements it generically
via `evalBroadcastBinaryOp<BinOp>()` in `src/cpu/ops.cpp` (fast exact-shape path plus a strided
fallback); MPSGraph's `additionWithPrimaryTensor:`/`multiplicationWithPrimaryTensor:` broadcast
natively, confirmed empirically (`MpsOps.AddBroadcastRowVector`/`MulBroadcastColumnVector`), not
just from the SDK docs. Other binary ops (`matmul`, etc.) still require exact shapes — broadcasting
was only added where a real need showed up (ONNX `Conv` bias).

## Roadmap

See `TODO.md` for the full phased plan. Phase 1 (core API), Phase 2 (CPU reference backend),
Phase 3a (MPSGraph backend, macOS/iOS), Phase 4a (ONNX import), and Phase 4c (graph caching) are
implemented and tested — Phase 4a is validated end-to-end against a real, standard pretrained
model (YuNet face detection) on real images, not just synthetic fixtures. Still open: the rest of
Phase 3 (DirectML, oneDNN/Vulkan, NNAPI/XNNPACK, WebNN passthrough), Phase 4b (TFLite import), and
Phase 5 (`campello_llm`/`campello_vision`).

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

Platform-specific CMake files live in `cmake/` and are selected based on target:
`cmake/macos.cmake`, `cmake/ios.cmake`, `cmake/windows.cmake`, `cmake/linux.cmake`,
`cmake/android.cmake`, `cmake/wasm.cmake`. All of them build the cross-platform core sources
(`CAMPELLO_NN_CORE_SOURCES`, set in the top-level `CMakeLists.txt`).
`cmake/macos.cmake`/`cmake/ios.cmake` additionally compile `src/metal/mps_backend.mm`
(Objective-C++, `enable_language(OBJCXX)`, `-fobjc-arc`) and link the Metal/MPSGraph
frameworks (see `TODO.md` Phase 3a). `cmake/windows.cmake` additionally compiles
`src/directml/directml_backend.cpp` (plain C++, no language extension needed — DirectML's
COM API is consumable via `<wrl/client.h>` ComPtr) and fetches the `Microsoft.AI.DirectML`
NuGet package via `FetchContent` (see `TODO.md` Phase 3b). Linux/Android/Web still build
CPU-only.

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
one). TFLite import is also implemented (`inc/campello_nn/tflite_importer.hpp`, same
`importTfliteFromMemory`/`importTfliteFromFile` split — see "TFLite Import" below). Graph
caching is implemented (see "Graph Caching" below):
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

### DirectML Backend (Windows)

`src/directml/directml_backend.{hpp,cpp}` — selected for every `DeviceType` other than
`Cpu` (`src/pi/context.cpp`'s `#elif defined(_WIN32)` branch). Plain C++, no language
extension needed (DirectML's COM API is consumable via `<wrl/client.h>` ComPtr, unlike
MPSGraph). **Decision: sequential per-node compiled operators, not a fused
`IDMLDevice1::CompileGraph(DML_GRAPH_DESC)`** — each non-`Input`/`Constant` IR node gets its
own `IDMLOperator`→`IDMLCompiledOperator` and a dedicated `DEFAULT`-heap `ID3D12Resource`
output buffer; `dispatch()` records one `RecordDispatch` + UAV barrier per node, in IR
order, on one shared command list, then a final GPU-to-GPU copy into each requested output
`Tensor`. This is a bigger simplicity cut than the MPSGraph backend's "skip
`MPSGraphExecutable`" decision (MPSGraph still fuses into one logical `MPSGraph*` object
graph; this skips fusion entirely) — costs perf, not correctness, since nothing in this op
set requires graph-level edge wiring to be expressible; revisit with the fused-graph API if
per-dispatch overhead matters later. `Reshape` is a zero-cost alias resolved dynamically at
dispatch time (not eagerly at compile time, since it may sit directly on a graph `Input`
whose buffer isn't known until the caller's `inputs` map is available) — same "pure shape
metadata, no new buffer" treatment as CPU/MPSGraph. Tensors are always `DEFAULT`-heap,
UAV-capable buffers (unlike MPSGraph's `MTLResourceStorageModeShared` unified memory, D3D12
`DEFAULT` buffers aren't CPU-mappable) — `write`/`read` stage through a throwaway
UPLOAD/READBACK buffer and a `CopyBufferRegion`, synchronously waited on one shared
`ID3D12Fence` used for every GPU submission in the backend. Requires `DML_FEATURE_LEVEL_5_1`
(driven by `ACTIVATION_SOFTMAX1`/`RESAMPLE2`), checked explicitly at construction rather
than discovered op-by-op. Adapter selection prefers hardware
(`IDXGIFactory6::EnumAdapterByGpuPreference`) and falls back to the WARP software adapter if
none is found, so CI on GPU-less `windows-latest` runners still exercises the real
op-mapping (`.github/workflows/ci.yml`'s `directml-integration` job) instead of skipping the
backend the way the macOS-only MPSGraph CI job has to. The DirectML SDK (headers + import
lib + redistributable DLL) is fetched via NuGet through CMake `FetchContent` in
`cmake/windows.cmake`, mirroring the GoogleTest/campello_image pattern already used in
`tests/CMakeLists.txt`. **Not yet built/run against real hardware or WARP** — written on a
machine with no local C++ toolchain available; DirectML enum/struct names were verified
against the actual fetched `DirectML.h` rather than trusted from memory, but the code is
otherwise unverified by compilation (see `TODO.md` Phase 3b's last item).

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
  as a harmless side effect. The same two images, plus a second real model
  (`blaze_face_short_range.tflite`), are reused by the TFLite importer's own real-model test — see
  "TFLite Import" below — rather than introducing a second pair of test images.

### TFLite Import

`src/tflite/` — same shape as `src/onnx/`, internal except for the top-level
`inc/campello_nn/tflite_importer.hpp` (`importTfliteFromMemory`/`importTfliteFromFile`). TFLite's
`.tflite` files are FlatBuffers, not protobuf — a vtable + byte-offset-indirection wire format
that's meaningfully easier to get subtly wrong by hand than protobuf's flat tag+value scheme, so
unlike `proto_reader.hpp`'s hand-rolled reader, this fetches Google's official `flatbuffers`
header-only runtime (`FlatBuffers::FlatBuffers` interface target, `FetchContent`'d in the
top-level `CMakeLists.txt` next to `CAMPELLO_NN_CORE_SOURCES`) and hand-writes field-ID accessors
directly on its `flatbuffers::Table`/`Vector` primitives — no `flatc` codegen step, no generated
schema header. Field IDs were verified against the real, current schema (now at
`google-ai-edge/LiteRT`'s `tflite/converter/schema/schema.fbs` — the old `tensorflow/tensorflow`
path 404s, the file moved during the TFLite→LiteRT migration), the same "verify against the real
thing, not memory" discipline the ONNX importer's protobuf field numbers followed.

- `tflite_model.hpp`/`tflite_parser.cpp`: parses a TFLite `Model` (subgraph 0 only, same
  single-graph assumption ONNX import makes) into plain structs (`TfliteGraph`/`TfliteOperator`/
  `TfliteTensor`/`TfliteBuffer`). Unlike ONNX's generic name-keyed `OnnxAttribute`, TFLite's
  builtin-options are per-op-type FlatBuffers tables known entirely from the op code, so the
  parser eagerly extracts every field a *supported* op might need into `TfliteOperator`'s flat
  named fields rather than keeping a generic options blob the importer would re-interpret itself.
- `tflite_importer.cpp`: the op-mapping table (`Conv2D`→`conv2d`, `DepthwiseConv2D`→`conv2d` with
  `groups`, `Add`→`add`, `Relu`→`relu`, `MaxPool2D`/`AveragePool2D`→`maxPool2d`/`avgPool2d`,
  `Reshape`→`reshape`, `Transpose`→`transpose`, `Softmax`→`softmax`, `Concatenation`→`concat`,
  `Gather`→`gather`, `FullyConnected`→`gemm`/`matmul`, `BatchMatMul`→`matmul`,
  `ResizeBilinear`/`ResizeNearestNeighbor`→`resize`, `Quantize`/`Dequantize`→`quantizeLinear`/
  `dequantizeLinear`, `Pad`→`concat`-with-a-zero-constant). `DepthwiseConv2D`'s weight layout
  (`[1,filter_height,filter_width, output_depth]`) and output-channel numbering
  (`oc = m + ic*depth_multiplier`) were verified against TFLite's own reference kernel source
  (`tflite/kernels/internal/reference/depthwiseconv_float.h`) rather than guessed — that numbering
  turns out to already match the standard "groups=input_channels" convention `conv2d()`'s `groups`
  parameter implements (the same one ONNX import already relies on for `Conv`'s `group` attribute),
  so the only new logic needed is a `[1,H,W,outC]`→`[outC,1,H,W]` byte permute, same trick as
  `Conv2D`'s own weight conversion just with a different axis mapping. `BatchMatMul`'s `adj_x`/
  `adj_y` were confirmed against `tflite/kernels/batch_matmul.cc` ("transpose the last two
  dimensions") and become an explicit `transpose()` of the operand's last two axes before the
  matmul, since `GraphBuilder::matmul()` has no transpose flag of its own. `Reshape`'s `-1`
  ("infer from total element count") is resolved by the importer itself before calling
  `builder.reshape()` (`resolveReshapeTarget()`, same logic as the ONNX importer's
  `resolveReshapeTarget()` minus ONNX's extra `0` sentinel, which TFLite's `Reshape` doesn't have) —
  `GraphBuilder::reshape()` requires an exact, already-resolved target shape.
  **`Pad`:** there's no dedicated core `pad` op (and none was added for this) — the only pattern
  actually seen in a real model (MediaPipe BlazeFace's residual channel-padding before an `add`,
  confirmed via its `Paddings` constant's actual bytes: zero everywhere except the channel axis,
  and zero *before*) is expressed as `concat(x, zeros)` along the channel axis, with every other
  axis/before-padding combination throwing rather than guessing. A real core `pad` op is worth
  adding if a future model needs a pattern this can't express.
  **Float16-weight `Dequantize`:** a "float16-weights" TFLite export (e.g. MediaPipe's `float16`
  variant) wraps every weight/bias constant in an explicit `Dequantize` purely as a precision cast
  (FLOAT16→FLOAT32), not real int8 quantization — distinguished from real quantization by checking
  whether the input tensor actually carries `QuantizationParameters` (`TfliteTensor::hasQuantization`).
  Since campello_nn's CPU backend already decodes any Float16-declared `Constant`/`Input` node to
  Float32 transparently at dispatch time (see "Dtype Support" below), this case is a no-op at the
  IR level — the importer just passes the operand through instead of emitting a real op. The
  trickier part: `Conv2D`/`DepthwiseConv2D`/`FullyConnected`'s import-time weight-byte permutation
  reads a weight/bias input's raw `Buffer` bytes *directly* (bypassing the lazy per-tensor `Operand`
  cache, since it needs to reorder bytes, not just reference a value) — when that weight is wrapped
  in a `Dequantize`, the permutation code needs the *original* constant tensor, not the
  `Dequantize`'s (bufferless, computed) output tensor. A `dequantSource` map (recorded when
  `Dequantize`'s no-op case runs, consulted via `applyOperator()`'s local `constIdx()` helper)
  redirects the permutation code back to the real underlying tensor index.
- **NHWC/OHWI ↔ NCHW/OIHW, the one piece with no ONNX precedent:** TFLite tensors are NHWC and
  conv weights are OHWI, unlike campello_nn's (and ONNX's) NCHW/OIHW throughout. Converting
  happens *only* at the graph boundary — a single `transpose()` right after each rank-4 graph
  input and right before each rank-4 graph output — not per-op; every interior op operates in
  NCHW exactly like an ONNX-imported graph. `Conv2D`/`FullyConnected` weight *constants* are
  pre-transposed at import time by reordering their raw bytes directly (`permuteBytes()` in
  `tflite_importer.cpp`) rather than inserting a runtime transpose op — the same "fold a
  resolvable shape op into the constant itself" trick as the ONNX importer's Conv-bias
  `[C]`→`[1,C,1,1]` reshape. **Known limitation:** axis-bearing ops (`Concatenation`/`Softmax`/
  `Gather`) remap TFLite's NHWC-numbered axis to NCHW assuming no intervening `Reshape`/
  `Transpose` has already broken the correspondence between the current operand and the original
  NHWC tensor it traces back to — documented in `tflite_importer.cpp`, not exercised by models
  that only reshape/transpose right before their final output.
- TFLite's quantization scale/zero-point lives on the *tensor* (`Tensor.quantization`), not the
  operator, unlike ONNX's `QuantizeLinear`/`DequantizeLinear` nodes carrying their own scale/
  zero-point inputs — `Quantize`/`Dequantize` read it straight off the relevant tensor.

Test fixtures (all synthetic, hand-written as JSON and compiled with `flatc --binary
tflite_schema.fbs <name>.json` — schema provenance in `tests/fixtures/NOTICE.md`):
`conv_add_relu.tflite` computes the identical Conv→Add→Relu graph as `conv_add_relu.onnx`, so
`tests/universal/test_tflite_importer.cpp` checks it against the exact same expected values as the
ONNX importer's test; `depthwise_conv2d.tflite` and `batch_matmul.tflite` each check against
expected values computed by hand independently (not just round-tripped through the importer
itself), specifically to verify the `DepthwiseConv2D`/`BatchMatMul` layout claims above are
actually correct, not just plausible-sounding.

Also validated end-to-end against a real, standard pretrained model — `tests/fixtures/
blaze_face_short_range.tflite` (MediaPipe's BlazeFace short-range face detector, Apache 2.0,
provenance in `tests/fixtures/NOTICE.md`), exercised by
`tests/universal/test_blazeface_face_detection.cpp`, gated behind `BUILD_MODEL_TESTS` alongside
the YuNet test. This is specifically what motivated adding `Pad` and the float16-weight
`Dequantize` handling above — the synthetic fixtures never needed either. Unlike YuNet's ~1800x
face-vs-no-face margin, this test's margin is modest (~2x: face.jpg scores ~0.47, no_face.jpg
scores ~0.23) — explained in the test file's comments: face.jpg is a conservatively-framed 1911
portrait, not a close-up selfie (this model's actual training distribution), and the test
deliberately skips MediaPipe's own letterbox preprocessing (measured to *hurt* this specific
fixture's score — shrinks the face further below the model's expected framing) and the real
product's anchor-decode/NMS/`min_score_thresh=0.5` pipeline, checking only the raw max per-anchor
sigmoid score.

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

`rmsNorm`/`rotaryEmbedding` (Phase 5 prep — LLaMA/GPT-style decoder blocks need both, neither
existed in the original transformer-block op set) were added with two different strategies:
- **`rmsNorm(x, scale, eps)`** is a new fused `OpKind::RmsNorm`, structurally `layerNorm` minus
  mean-centering and the bias/shift term (`out = x * rsqrt(mean(x^2) + eps) * scale`) — same
  precedent as `batchNorm`/`instanceNorm` each being their own op rather than composed from a
  generic reduce primitive (no such primitive is exposed). CPU kernel mirrors `evalLayerNorm`;
  MPSGraph composes from `squareWithTensor:`/`meanOfTensor:axes:`/`reciprocalSquareRootWithTensor:`
  (all verified against the real `MPSGraphArithmeticOps.h`/`MPSGraphReductionOps.h` headers before
  use, same standard as everywhere else in this codebase) — no single native MPSGraph op for this,
  same situation as `gelu`/`gemm`. **DirectML case intentionally left unimplemented** (falls
  through to the backend's existing default case, which throws `"unhandled OpKind"` rather than
  silently misbehaving) — `DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC` (already used for
  `LayerNorm` in this backend) looks like the right operator, but its `NormalizeVariance` flag's
  exact semantics need checking against the real `DirectML.h`/docs on Windows before trusting it;
  left for that platform's follow-up work.
- **`rotaryEmbedding(x, cos, sin)`** needed *no new op at all*. The standard "rotate-half" RoPE
  (`out = x*cos + rotateHalf(x)*sin`, `rotateHalf` = split the last dim in half and return
  `concat(-secondHalf, firstHalf)`) decomposes entirely into `slice`/`concat`/`mul`/`add` — all
  already implemented and verified on every backend. `GraphBuilder::rotaryEmbedding()` is a pure
  composition calling those same public methods, exactly like `quantizedMatmul()` already does
  (`dequantizeLinear()` then `matmul()`) — so it works on CPU/MPSGraph/DirectML automatically,
  with zero backend-specific code. The one bit of new logic is building a dtype-correct `-1.0`
  constant (`encodeFloat16()` for `Float16`, raw bytes for `Float32`) to negate the second half via
  `mul()`, since there's no dedicated negate/subtract op. Restricted to Float32/Float16 (rotary
  embeddings aren't meaningful for the other `DataType`s); `x`'s last dimension must be even.

## Roadmap

See `TODO.md` for the full phased plan. Phase 1 (core API), Phase 2 (CPU reference backend),
Phase 3a (MPSGraph backend, macOS/iOS), Phase 3b (DirectML backend, Windows), Phase 4a (ONNX
import), Phase 4b (TFLite import), and Phase 4c (graph caching) are implemented and tested — both
Phase 4a (YuNet) and Phase 4b (MediaPipe BlazeFace) are validated end-to-end against real, standard
pretrained models on real images, not just synthetic fixtures. Still open: the rest of Phase 3
(oneDNN/Vulkan, NNAPI/XNNPACK, WebNN passthrough) and Phase 5 (`campello_llm`/`campello_vision`).

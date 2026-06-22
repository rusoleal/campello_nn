# campello_nn / campello_llm — TODO

Roadmap derived from [NN_ARCHITECTURE.md](./NN_ARCHITECTURE.md). Phases are ordered so each one
is buildable and testable before the next depends on it: core API → one working backend (CPU) →
platform accelerator backends → model import (ONNX/TFLite) and graph caching, both inside
`campello_nn` → the weight-only-format layers (`campello_llm`, future `campello_vision`) on top.

---

## Phase 0 — Project Scaffolding ✅

- [x] CMake build system, matching `campello_gpu`'s conventions (`systems::leal::` namespace,
      handle-based `void*` pattern, per-platform `.cmake` includes, GTest via FetchContent)
- [x] Directory layout: `inc/campello_nn/{constants,descriptors}`, `src/{pi,cpu}`,
      `tests/{universal,platform}`, `examples/cpu` (`campello_llm/`, future `campello_vision/`,
      `third_party/` deferred to the phases that need them — no separate `campello_nn_convert/`
      project; model import and graph caching live inside `campello_nn` itself, see Phase 4)
- [ ] Set up dependency management for accelerator SDKs (DirectML headers, Metal-cpp/MPSGraph
      headers, NNAPI headers, XNNPACK, oneDNN, Vulkan SDK) — deferred to Phase 3, no SDK needed yet
- [ ] Add `.clang-format` / `.clang-tidy`
- [x] C++20, GoogleTest + `ctest` wired (`tests/CMakeLists.txt`)
- [x] CI workflow running universal tests on macOS/Linux/Windows (`.github/workflows/ci.yml`) —
      Android/Web CI deferred to Phase 3d/3e
- [x] Root `CLAUDE.md` documenting the architecture split and conventions

---

## Phase 1 — `campello_nn` Core API ✅

Goal: get the public API from §3 of the architecture doc compiling, with backend dispatch stubbed
behind a `Backend` interface.

- [x] `DeviceType`, `DataType` enums
- [x] `TensorDescriptor` struct
- [x] `Tensor` class (shape, dataType, `write`/`read`) as a thin handle over a backend-owned buffer
- [x] `ContextDescriptor` struct
- [x] `Context` class: `create()`, `createTensor()`, `dispatch()` — owns a `Backend` instance
      selected by `DeviceType`
- [x] Internal `Backend` interface (`src/pi/backend.hpp`) that concrete backends implement (tensor
      allocation, graph compilation, graph dispatch) — what Phase 2/3 backends plug into
- [x] `Operand` opaque handle + internal graph IR node representation (`src/pi/ir.hpp`: `OpKind`,
      `Node`, `GraphIR`) — built by `GraphBuilder`, compiled by a `Backend`
- [x] `GraphBuilder` class with all ops from the doc:
  - [x] `input`, `constant`
  - [x] elementwise/activation: `add`, `mul`, `gelu`, `softmax`, `layerNorm`, `relu`, `sigmoid`
        (`relu`/`sigmoid` added during Phase 4a — see "Float16/Int8/Uint32 Support" pattern; real
        ONNX models use `Relu`, not `Gelu`)
  - [x] linear algebra: `matmul`, `gemm`
  - [x] shape ops: `reshape`, `transpose`, `concat`, `slice`, `gather`
  - [x] vision (scope added after initial Phase 1 pass — see "Vision/Multimodal Op Set" below):
        `conv2d`, `maxPool2d`, `avgPool2d`, `resize`, `batchNorm`, `instanceNorm`
  - [x] quantization (see "Float16/Int8/Uint32 Support" below): `quantizeLinear`,
        `dequantizeLinear`, `quantizedMatmul`
  - [x] `build()` → `Graph`
- [x] Shape/dtype inference and validation at build time (`src/pi/graph_builder.cpp` — mismatched
      matmul dims, bad axes, out-of-bounds slices, etc. all throw `std::runtime_error` before
      reaching a backend; covered by `tests/universal/test_graph_builder_validation.cpp`)
- [x] `Graph` class: opaque compiled/optimized executable handle
- [x] **Decision:** `Context::dispatch()` returns a `Fence` (matching `campello_gpu`'s
      submit-plus-fence convention), not `Future<void>` as originally sketched in the doc
- [ ] Richer error handling/diagnostics (currently plain `std::runtime_error` with a message;
      no error codes/categories yet)
- [x] Public headers finalized under `inc/campello_nn/`

---

## Phase 2 — CPU Reference Backend ✅

Goal: first real, runnable backend. Exists so correctness can be validated everywhere before
chasing platform accelerators, and so Linux has a guaranteed fallback.

- [x] CPU tensor storage (`src/cpu/cpu_tensor.hpp`: `std::vector<uint8_t>` backed, `read`/`write`
      as raw memcpy)
- [x] CPU implementations of every `GraphBuilder` op (`src/cpu/ops.cpp`: matmul, gemm, softmax,
      layerNorm, gelu, add, mul, reshape, transpose, concat, slice, gather, conv2d, maxPool2d,
      avgPool2d, resize, batchNorm, instanceNorm, quantizeLinear, dequantizeLinear, relu, sigmoid)
- [x] Graph executor (`src/cpu/cpu_backend.cpp::dispatch`) — IR nodes are already topologically
      ordered by construction (an op's inputs are always earlier indices), so this is a single
      linear pass, no separate toposort needed. No dead-code elimination yet (every node in the
      built IR is evaluated, even ones not reachable from the requested outputs)
- [x] Float32 for all ops; `gather`'s `indices` operand supports Int32 and Uint32 (needed for
      embedding lookup with either signed or unsigned token ids)
- [x] Float16 compute support, all ops (see "Float16/Int8/Uint32 Support" below)
- [x] Unit tests per op, hand-computed expected values (`tests/universal/test_cpu_ops.cpp`)
- [x] End-to-end test: matmul → add (bias) → gelu → layerNorm graph, output checked against an
      independently-computed reference (`tests/universal/test_transformer_block.cpp`)
- [ ] Basic graph-level optimizations (constant folding, dead-node elimination) — informs what
      `campello_nn`'s own graph serialization (Phase 4) will later cache
- [x] Working example: `examples/cpu/main.cpp` (build with `-DBUILD_EXAMPLES=ON`)

---

## Vision/Multimodal Op Set ✅

The project's scope grew beyond text-only transformer LLMs to include vision and multimodal
models (user decision, see conversation — not in the original `NN_ARCHITECTURE.md`). This added
three ops, implemented across every layer the original op set touches:

- [x] `Conv2dDescriptor`/`Pool2dDescriptor` (`inc/campello_nn/descriptors/`) — explicit
      stride/dilation/padding/groups, no auto "same"/"valid" padding styles, matching the rest of
      the op set's explicit style. Input is NCHW, conv weights are OIHW (PyTorch/ONNX convention).
      Bias is not fused into `conv2d` — compose with `add()` after broadcasting, same pattern as
      `gemm`'s bias.
- [x] `GraphBuilder::conv2d/maxPool2d/avgPool2d` with shape inference + validation
      (`src/pi/graph_builder.cpp`); `resize` added separately below
- [x] CPU kernels (`src/cpu/ops.cpp`): naive direct-convolution loop with groups support; pooling
      shares one `evalPool2d` helper for max/avg, matching MPSGraph's default
      `includeZeroPadToAverage=NO` behavior (avgPool2d divides by valid-sample count, not the full
      padded kernel window)
- [x] MPSGraph mapping (`src/metal/mps_backend.mm`): `MPSGraphConvolution2DOpDescriptor` +
      `convolution2DWithSourceTensor:`, `MPSGraphPooling2DOpDescriptor` +
      `maxPooling2DWithSourceTensor:`/`avgPooling2DWithSourceTensor:` — direct 1:1 mapping, no
      composition needed (unlike `gelu`/`gemm`)
- [x] Tests on both backends: `tests/universal/test_cpu_ops.cpp` and `tests/platform/test_mps_ops.cpp`
      each gained `Conv2d`/`MaxPool2d`/`AvgPool2d`, hand-computed expected values
- [x] `resize` op (nearest/bilinear, NCHW, `ResizeDescriptor` with `centerResult`/`alignCorners`
      mirroring `MPSGraphResizeOps`'s documented semantics — defaults match OpenCV/TF2 behavior).
      CPU kernel implements the same coordinate-mapping formula MPSGraph documents; hand-derived
      expected values for both modes matched real MPSGraph/GPU output exactly, confirming the
      formulas agree. MPSGraph mapping is a direct 1:1 call to `resizeTensor:size:mode:
      centerResult:alignCorners:layout:name:`. Tests: `CpuOps`/`MpsOps`
      `ResizeBilinearAlignCorners`/`ResizeNearestAlignCorners`
- [x] `batchNorm`/`instanceNorm` ops, NCHW. **Decision:** kept as two distinct ops rather than one
      generic "normalize over given axes" primitive, matching ONNX/PyTorch's actual
      `BatchNorm2d`/`InstanceNorm2d` semantics 1:1 (this will matter once `campello_llm` loads
      weights from real vision models): `batchNorm(x, mean, variance, scale, bias, eps)` takes
      mean/variance as **given inputs** (inference-mode running stats — it does not compute them
      from `x`, mirroring how real inference engines treat BatchNorm); `instanceNorm(x, scale,
      bias, eps)` computes mean/variance itself, per-`(N,C)` over the spatial `(H,W)` axes — same
      idea as `layerNorm` but over spatial axes with channel-broadcast scale/bias instead of
      trailing-axis. Both map to the same MPSGraph `normalizationWithTensor:meanTensor:
      varianceTensor:gammaTensor:betaTensor:epsilon:name:` call `layerNorm` uses; `instanceNorm`
      additionally calls `meanOfTensor:axes:`/`varianceOfTensor:axes:` with axes `[2,3]` instead of
      the last axis. `batchNorm`/`instanceNorm`'s per-channel operands are reshaped to
      `[1,C,1,1]` (new `reshapeChannel` helper, alongside the existing `reshapeTrailing`) so
      MPSGraph broadcasts them against NCHW correctly. Tests: `CpuOps`/`MpsOps`
      `BatchNorm`/`InstanceNorm` — 47/47 tests passing across CPU + real MPSGraph/GPU hardware
- [ ] Cross-modal fusion needs no new ops — vision embeddings feed into the same `matmul`/`softmax`
      attention machinery already used for text; this is a `campello_llm`-layer wiring concern
      (Phase 5), not a `campello_nn` op-set gap

---

## Float16/Int8/Uint32 Support ✅

- [x] **Float16, all ops.** `inc/campello_nn/float16.hpp` — public `encodeFloat16`/`decodeFloat16`
      bit-manipulation functions (no platform deps; C++20 has no built-in half type, so
      `DataType::Float16` tensors store this 2-byte encoding directly — callers need this to
      produce/consume `Tensor::write()`/`read()` bytes).
  - **CPU backend decision:** boundary-conversion design, not per-kernel dtype branching.
    `cpu_backend.cpp::dispatch()` decodes Float16 Input/Constant bytes to Float32 right when
    populating a node, and re-encodes only when writing the final result into a
    Float16-declared output `Tensor`. Every kernel in `ops.cpp` is unmodified and unaware
    Float16 exists — it only ever sees genuinely-Float32 `CpuValue`s. `requireFloat32()` now
    accepts Float32 or Float16 (both are Float32 in memory by kernel time; only Int32/Uint32/
    Int8 are actually rejected).
  - **MPSGraph backend:** runs Float16 natively (no decode step — GPU/ANE prefer fp16). Found
    and fixed two latent bugs: `gelu` and `gemm`'s composed constants (`0.5`, `1.0`, `1/√2`,
    `alpha`, `beta`) were hardcoded to `MPSDataTypeFloat32`, which would have broken/mismatched
    against an actual Float16 tensor; both now use `mpsDataType(node.dataType)`.
  - Tests: `tests/universal/test_cpu_float16_ops.cpp` /
    `tests/platform/test_mps_float16_ops.cpp` — `Add`/`Mul`/`Gelu`/`MatMul`/`LayerNorm` (one
    representative op per category: elementwise, activation, linalg, normalization), passing on
    both CPU and real MPSGraph/GPU hardware.
- [x] **Int8, real quantization** (not storage-only — ONNX/WebNN-shaped). New ops
      `quantizeLinear(x, scale, zeroPoint)` (float → Int8) and `dequantizeLinear(x, scale,
      zeroPoint)` (Int8 → Float32), matching ONNX `QuantizeLinear`/`DequantizeLinear`'s formula
      exactly and mapped 1:1 to MPSGraph's `quantizeTensor:scale:zeroPoint:dataType:name:`/
      `dequantizeTensor:...` per-tensor scalar overloads.
  - `quantizedMatmul(activation, weightInt8, weightScale, weightZeroPoint)`: a `GraphBuilder`
    convenience that composes `dequantizeLinear(weightInt8)` then `matmul(activation,
    dequantized)` — **decision:** weight-only quantization (int8 storage, float32 compute), not
    a genuine int8×int8 GEMM kernel, since neither MPSGraph nor the CPU backend exposes one;
    this is the realistic deployment pattern (same approach GGUF/bitsandbytes int8 use) and
    needed zero new backend code beyond `quantizeLinear`/`dequantizeLinear`/`matmul` already
    being in place.
  - Per-tensor scale/zero-point only for now (MPSGraph also has per-axis `scaleTensor:`
    overloads for true per-channel weight quantization — not wired up; would matter for
    production-quality weight compression later).
  - Tests: `tests/universal/test_cpu_quantization_ops.cpp` /
    `tests/platform/test_mps_quantization_ops.cpp` — `QuantizeLinear`/`DequantizeLinear`/
    `QuantizedMatmul`, passing on both backends.
- [x] **Uint32 for `gather` indices**, alongside the existing Int32 support (some
      tokenizers/models use unsigned token ids). `CpuValue` gained a `u32()` accessor; CPU
      `evalGather` branches on `indices.dataType`. No MPSGraph change needed — its
      `gatherWithUpdatesTensor:` mapping was already dtype-agnostic. Tests:
      `CpuOps`/`MpsOps::GatherUint32Indices`.
- [x] All 65 tests passing across CPU + real MPSGraph/GPU hardware after this round (up from 47).

---

## Phase 3 — Platform Accelerator Backends

Each backend implements the same internal `Backend` interface from Phase 1. Suggested order:
macOS/iOS first (MPSGraph is architecturally closest to WebNN per the doc), then Windows
(DirectML), then Linux, then Android, then Web. Reorder based on what hardware/SDKs are actually
available to develop against.

### 3a. macOS / iOS — MPSGraph ✅
- [x] **Decision:** MPSGraph has no Apple-provided C++ binding (unlike Metal/metal-cpp), so
      `src/metal/mps_backend.mm` is Objective-C++. `mps_backend.hpp` stays pure C++ (pimpl'd
      `Impl`) so `context.cpp` can include it without needing Obj-C++ compilation itself.
- [x] `Context` backend selection wiring: `DeviceType::Cpu` → `CpuBackend`; `Gpu`/`Npu`/`Default`
      → `MpsBackend` on Apple platforms (`src/pi/context.cpp`, `#ifdef __APPLE__`)
- [x] IR-node → `MPSGraph` op mapping for every `GraphBuilder` op (`src/metal/mps_backend.mm`):
      `add`/`mul` direct; `gelu` composed from `erfWithTensor` + arithmetic (no native GELU op
      in MPSGraph); `softmax` via `softMaxWithTensor:axis:`; `layerNorm` composed from
      `meanOfTensor`/`varianceOfTensor`/`normalizationWithTensor` (gamma/beta reshaped to
      broadcast against the last axis); `matmul`/`gemm` via `matrixMultiplicationWithPrimaryTensor`
      (gemm's `alpha`/`beta`/bias-add composed manually); `reshape`/`transpose`/`concat`/`slice`
      via the corresponding `MPSGraphTensorShapeOps`; `gather` via `gatherWithUpdatesTensor:axis:`
- [x] Tensor ↔ `MPSGraphTensorData`/`MTLBuffer` bridging — tensors use
      `MTLResourceStorageModeShared` so `write`/`read` are plain `memcpy` against
      `buffer.contents`, same approach `campello_gpu`'s Metal backend uses
- [x] **Decision:** skipped the separate `MPSGraphExecutable`/`compileWithDevice:` step for v1 —
      `compileGraph()` builds the `MPSGraph*` once (IR → ops, "build once"), `dispatch()` calls
      `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` synchronously each time
      ("dispatch many times"), with `Fence` always pre-signaled. Revisit `MPSGraphExecutable` if
      per-dispatch specialization overhead matters later.
- [x] Dispatch/execution + `Fence` completion wiring (synchronous; see Phase 1's decision)
- [ ] Optional ANE delegate path reached through MPSGraph — not built explicitly: MPSGraph's own
      placement pass (`MPSGraphOptimizationLevel1`, the framework default) can already dispatch
      eligible ops to GPU or ANE on its own, so this may already be partially covered without
      extra work; revisit if ANE utilization needs to be verified/forced
- [x] Parity test suite: `tests/platform/test_mps_ops.cpp` mirrors every op test from
      `tests/universal/test_cpu_ops.cpp`, run against the real MPSGraph backend
      (`-DBUILD_INTEGRATION_TESTS=ON`) on Apple Silicon GPU hardware — 28/28 passing (12
      transformer ops + 7 vision ops + 9 Float16/Int8/Uint32 tests, see "Vision/Multimodal Op
      Set" and "Float16/Int8/Uint32 Support" below)
- [x] Float16/Int8/Uint32 tensor support — see "Float16/Int8/Uint32 Support" below

### 3b. Windows — DirectML ✅
- [x] **Decision:** sequential per-node compiled operators, not a single fused
      `IDMLDevice1::CompileGraph(DML_GRAPH_DESC)`. Each non-Input/Constant IR node
      gets its own `IDMLOperator`→`IDMLCompiledOperator`, with a dedicated
      DEFAULT-heap `ID3D12Resource` output buffer; `dispatch()` records one
      `RecordDispatch` per node plus a UAV barrier, in IR order, on one shared
      command list. Bigger simplicity cut than MPSGraph's "skip
      `MPSGraphExecutable`" decision (that one still fuses into one `MPSGraph*`
      object graph; this skips fusion entirely) — costs perf (extra DEFAULT-heap
      round trips between ops), not correctness, since no op here requires
      graph-level edge wiring to be expressible. Revisit with `DML_GRAPH_DESC` if
      per-dispatch overhead matters later, same framing as graph caching/
      MPSGraphExecutable's "revisit if it becomes the bottleneck" notes.
- [x] **Decision:** `Reshape` is a zero-cost alias — no `IDMLOperator`, no buffer.
      Resolved dynamically at `dispatch()` time (`resolveBuffer()` in
      `directml_backend.cpp`), not eagerly at compile time, since a Reshape may
      sit directly on top of a graph `Input` whose backing buffer isn't known
      until the caller's `inputs` map is available.
- [x] `Context` backend wiring: `DeviceType::Cpu` → `CpuBackend`; everything else
      → `DirectMlBackend` (`src/pi/context.cpp`, `#elif defined(_WIN32)`,
      parallel to the `__APPLE__` branch).
- [x] DirectML SDK fetched via NuGet through CMake `FetchContent` in
      `cmake/windows.cmake` (`Microsoft.AI.DirectML` 1.15.4 — mirrors the
      GoogleTest/campello_image `FetchContent` pattern already used in
      `tests/CMakeLists.txt`; the `.nupkg` is a plain zip, `DOWNLOAD_NAME
      directml.zip` forces CMake to recognize/auto-extract it despite the
      extensionless URL). `DirectML.dll` copied next to test binaries via a
      post-build step (`tests/CMakeLists.txt`).
- [x] **Decision:** require `DML_FEATURE_LEVEL_5_1` at backend construction
      (`IDMLDevice::CheckFeatureSupport`) — the floor driven by
      `ACTIVATION_SOFTMAX1`/`RESAMPLE2`, corresponding to DirectML redistributable
      1.9.0+. Throws a clear error at construction rather than discovering gaps
      op-by-op at dispatch time.
- [x] **Decision:** hardware adapter preferred (`IDXGIFactory6::
      EnumAdapterByGpuPreference`, `DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE`),
      falling back to the WARP software adapter
      (`IDXGIFactory4::EnumWarpAdapter`) if none found — lets CI on
      `windows-latest` (no GPU) still exercise the real op-mapping, not just
      skip the backend like the macOS-only MPSGraph CI job has to.
- [x] IR-node → `DML_OPERATOR_DESC` mapping for every op (`src/directml/
      directml_backend.cpp`): `Add`/`Mul`→`ELEMENT_WISE_ADD`/`MULTIPLY`
      (broadcast via stride-0 `DML_BUFFER_TENSOR_DESC.Strides`, computed by
      `TensorDescBuilder::setBroadcast`); `Relu`/`Sigmoid`→native activation ops;
      `Gelu`→composed from `ELEMENT_WISE_ERF` (with its `ScaleBias` field fusing
      the `x/√2` pre-scale) + two `ELEMENT_WISE_IDENTITY` ops (each using
      `ScaleBias` for `1+erf` and `0.5*x`) + one final `ELEMENT_WISE_MULTIPLY` —
      same `0.5*x*(1+erf(x/√2))` formula as MPSGraph, deliberately not the native
      `ACTIVATION_GELU` (feature-level gated, formula-fidelity risk);
      `Softmax`→`ACTIVATION_SOFTMAX1`; `LayerNorm`/`InstanceNorm`→both via
      `MEAN_VARIANCE_NORMALIZATION1` (LayerNorm: axis=last, scale/bias
      right-aligned NumPy-broadcast; InstanceNorm: axes={2,3}, scale/bias via a
      dedicated `setChannelBroadcast` helper — NCHW channel-axis broadcast is
      *not* NumPy right-alignment, it needs explicit `[0,1,0,0]`-style strides,
      unlike LayerNorm's case which right-alignment happens to get right for
      free); `BatchNorm`→native `BATCH_NORMALIZATION` (`Spatial=TRUE`, given
      mean/variance, not computed); `MatMul`/`Gemm`→both via `GEMM`
      (alpha=1/beta=0/no C for MatMul); `Reshape`→no operator, see above;
      `Transpose`→`ELEMENT_WISE_IDENTITY` with the *input* tensor desc's
      `Strides` set to the permutation of the original contiguous strides
      (materializes into a packed output buffer — verified the exact
      `inIdx[perm[i]] = outIdx[i]` convention against `src/cpu/ops.cpp::
      evalTranspose` before writing this, rather than assuming a direction);
      `Concat`→`JOIN`; `Slice`→`SLICE1`; `Gather`→`GATHER` (`IndexDimensions=0`,
      plain ONNX-style scalar-index gather); `Conv2d`→`CONVOLUTION`
      (`Mode=CROSS_CORRELATION`, no fused bias, matching the descriptor's
      "bias not fused" convention); `MaxPool2d`/`AvgPool2d`→native pooling ops
      (`AvgPool2d`: `IncludePadding=FALSE`, matching CPU's non-padded-count-only
      averaging); `Resize`→`RESAMPLE2` (not `RESAMPLE3`, which needs feature
      level 6.4 — too new/risky for a v1 floor). `QuantizeLinear`/
      `DequantizeLinear`→native quantize ops with scale/zero-point as tiny
      1-element constant buffers DML wants as tensor inputs (the IR stores them
      as scalars) — zero-point is cast from the IR's `float` to `int8_t` before
      upload, since a reinterpret would silently shift every dequantized value.
      `quantizedMatmul` needs no backend case — `GraphBuilder` already expands it
      into `DequantizeLinear`+`MatMul` nodes before the backend sees it.
- [x] **`Resize`/`RESAMPLE2` coordinate-mapping formula — verified against
      Microsoft's own docs rather than assumed**, after first writing down the
      wrong direction: `Scale = outSize/inSize` (not `inSize/outSize` — confirmed
      via the documented `OutputTensorX = (InputTensorX + InputPixelOffset) *
      Scale + OutputPixelOffset` formula and "scales > 1 scale up the image").
      `InputPixelOffsets=0.5`/`OutputPixelOffsets=-0.5` reproduces the standard
      half-pixel/OpenCV/TF2-style resize per Microsoft's own worked example —
      matches `centerResult=true,alignCorners=false`. `alignCorners=true` uses
      `Scale=(outSize-1)/(inSize-1)`, offsets 0. `RoundingDirection` (Nearest
      only): `nearestRoundsDown` → `DECREASING` (floor), else `INCREASING`.
- [x] Tensor ↔ `ID3D12Resource` bridging (`createTensor`/`writeTensor`/
      `readTensor` in `directml_backend.cpp`): every tensor is a DEFAULT-heap,
      UAV-capable buffer (DirectML requires `D3D12_RESOURCE_STATE_UNORDERED_
      ACCESS` for bound buffers) — unlike MPSGraph's `MTLResourceStorageModeShared`
      (Apple unified memory), D3D12 DEFAULT-heap resources aren't CPU-mappable,
      so `write()`/`read()` stage through a throwaway UPLOAD/READBACK-heap buffer
      and a `CopyBufferRegion`, synchronously waited on the same shared
      `ID3D12Fence` every other GPU submission uses (one fence/value-counter
      space for the whole backend, avoiding a copy-queue vs. compute-queue
      fence-value collision).
- [x] Graph compile → per-node `IDMLCompiledOperator` (`compileGraph()`) — every
      compiled operator is initialized once via `IDMLOperatorInitializer` at
      compile time (required even for stateless ops, not just ones with
      persistent state), with its persistent/temporary resources and descriptor
      heap/binding table allocated alongside it; all of compileGraph's constant
      uploads + operator initializations are batched onto one command list and
      flushed with a single `executeAndWait()` at the end.
- [x] Command-list dispatch + GPU fence wiring — `Context::dispatch()`'s
      `Fence` is always pre-signaled (`DmlFence{true}`), matching the CPU/
      MPSGraph backends' synchronous-dispatch convention: `dispatch()` records
      every node's `RecordDispatch` plus the final output-buffer copies, then
      calls `executeAndWait()` (submit + `ID3D12Fence::SetEventOnCompletion` +
      `WaitForSingleObject`) before returning — not a separate async fence the
      caller polls later.
- [x] Parity tests mirroring `tests/platform/test_mps_*.cpp` op-for-op (same
      hand-computed expected values/tolerances): `tests/platform/
      test_directml_ops.cpp` (25 core ops), `test_directml_float16_ops.cpp` (5),
      `test_directml_quantization_ops.cpp` (3), `test_directml_graph_cache.cpp`
      (1) — wired into `tests/CMakeLists.txt`'s `BUILD_INTEGRATION_TESTS` block
      (`elseif(WIN32)`, parallel to the `if(APPLE)` branch) and a new
      `directml-integration` CI job (`.github/workflows/ci.yml`, `windows-latest`,
      WARP-backed since that runner has no GPU).
- [ ] Not yet done: any actual run of these tests against real hardware or even
      WARP — written without a local C++ toolchain available (no `cmake`/`cl.exe`
      on the dev machine at the time), so exact DirectML enum/struct names were
      verified against the real fetched `DirectML.h` (downloaded and inspected
      directly) rather than trusted from memory, but the code itself is
      unverified by compilation. First real build/test run is follow-up work.

### 3c. Linux — oneDNN or Vulkan compute shaders
- [ ] Decide oneDNN vs. Vulkan-compute as the primary path (doc notes neither is an
      "official OS API" — pick one as primary, keep the other as a documented option)
- [ ] IR-node → primitive/shader mapping for every op (this is the most build-it-yourself backend
      since there's no ready-made graph API to target)
- [ ] Tensor buffer management (oneDNN memory objects or Vulkan buffers)
- [ ] Graph compile step (oneDNN primitive cache, or Vulkan pipeline + shader module cache)
- [ ] Dispatch + sync/fence wiring
- [ ] Parity tests against CPU reference

### 3d. Android — NNAPI successor / vendor delegate, fallback XNNPACK
- [ ] `Context` backend wiring, with capability detection (treat as best-effort per the doc —
      OEM coverage is inconsistent)
- [ ] IR-node → NNAPI-successor/vendor-delegate op mapping where available
- [ ] XNNPACK fallback path for ops/devices the vendor delegate doesn't cover
- [ ] Tensor buffer bridging for both paths
- [ ] Parity tests against CPU reference (may need per-device tolerance adjustments)

### 3e. Web — passthrough to native browser WebNN
- [ ] Thin shim mapping `GraphBuilder`/IR directly to the browser's `MLGraphBuilder` API
      (should be the simplest backend since the shapes already match 1:1)
- [ ] Tensor ↔ `MLTensor`/`ArrayBuffer` bridging
- [ ] Build target wiring (Emscripten or equivalent) to compile `campello_nn` for web
- [ ] Parity tests run in a browser test harness

---

## Phase 4 — `campello_nn` Model Import and Graph Caching

**Decision (see conversation, supersedes an earlier draft of this phase):** no separate
`campello_nn_convert` project. ONNX/TFLite import and `Graph` serialization are both pure
graph-level translation jobs needing zero architecture-specific knowledge — unlike weight-only
formats (safetensors/gguf), which genuinely need `campello_llm`/`campello_vision`'s per-architecture
wiring to make sense of. So both live directly inside `campello_nn`. Depends on Phase 1–3 being
far enough along that the op set and `Graph` representation are stable.

### 4a. ONNX import ✅ — validated end-to-end against a real model on real images
- [x] **Decision:** hand-rolled minimal protobuf reader (`src/onnx/proto_reader.hpp`) — varint/
      tag/length-delimited primitives only, no `protoc`, no full `protobuf` dependency. Field
      numbers for every ONNX message used (`ModelProto`, `GraphProto`, `NodeProto`,
      `AttributeProto`, `TensorProto`, `ValueInfoProto`, `TypeProto`, `TensorShapeProto`) were
      verified byte-for-byte against a real `.onnx` file generated by Python's official `onnx`
      package (1.19.1) before writing any parsing logic against them — every assumed field number
      matched on the first try.
- [x] `src/onnx/onnx_model.hpp` + `onnx_parser.cpp`: parses `ModelProto.graph` into nodes,
      initializers (with `TensorProto.raw_data`, falling back to `float_data`/`int32_data`/
      `int64_data` if `raw_data` is absent), and graph inputs/outputs (with shape inference for
      dynamic/symbolic dimensions — defaults to 1, the practical choice for single-item
      inference). Handles both packed and unpacked repeated-scalar encodings for `AttributeProto`
      (real ONNX files were observed using the unpacked form). `onnx::onnxElemTypeHasDataType()`
      lets the importer skip initializers that are only ever import-time metadata (e.g. an INT64
      `Reshape` shape constant) instead of failing to bind them as a `campello_nn::DataType`.
- [x] **Decision:** added `internal::operandShapeForImport(const Operand&)` (declared in the
      public `operand.hpp` inside an `internal` namespace, implemented in `graph_builder.cpp`,
      returns a shape *copy* — a reference would dangle once the IR node vector reallocates on a
      later `GraphBuilder` call). Needed once real ops required an intermediate tensor's actual
      shape during import (`Reshape`'s `-1` inference, `Resize`'s scale-relative sizing, and
      computing `Conv`'s broadcast-bias shape) rather than duplicating `GraphBuilder`'s own shape
      inference a second time in the importer.
- [x] `GraphBuilder::relu`/`sigmoid` added (CPU kernels + native MPSGraph `reLUWithTensor:`/
      `sigmoidWithTensor:` — no composition needed) — the gap flagged before was real; YuNet uses
      `Relu`, not `Gelu`. Tests on both backends.
- [x] `ResizeDescriptor` gained `nearestRoundsDown` (ONNX `nearest_mode="floor"` differs from this
      project's prior default of round-to-nearest). CPU: `std::floor` instead of `std::round`.
      MPSGraph: switches from the generic `resizeTensor:mode:` call (no rounding-mode parameter)
      to `resizeNearestWithTensor:sizeTensor:nearestRoundingMode:...` with
      `MPSGraphResizeNearestRoundingModeFloor` — that overload only accepts a `sizeTensor`, not a
      static `MPSShape`, so a small Int32 constant tensor is built for it. Verified on real
      MPSGraph/GPU hardware with a case specifically chosen where floor and round-to-nearest
      disagree (2×2→3×3, not the 2×2→4×4 case used elsewhere, which coincidentally gives the same
      answer either way).
- [x] Op-mapping table (`src/onnx/onnx_importer.cpp`): `Conv`→`conv2d`, `Add`→`add`, `Mul`→`mul`,
      `Relu`→`relu`, `Sigmoid`→`sigmoid`, `MatMul`→`matmul`, `Gemm`→`gemm` (no transA/transB),
      `BatchNormalization`→`batchNorm` (mind the input-order mismatch — ONNX is
      `X,scale,B,mean,var`; ours is `x,mean,variance,scale,bias`), `Transpose`→`transpose`,
      `MaxPool`/`AveragePool`→`maxPool2d`/`avgPool2d`, `Reshape`→`reshape` (resolves ONNX's `0`/`-1`
      sentinels via `operandShapeForImport`), `Resize`→`resize` (maps
      `coordinate_transformation_mode`→`centerResult`/`alignCorners`, `nearest_mode`→
      `nearestRoundsDown`; supports both `scales` and `sizes` inputs).
  - **`Conv` with a fused bias — real support, not a thrown error.** Turned out to be the
    *universal* case once tested against a real model (YuNet: 59/59 `Conv` nodes have a bias
    input). Initially worked around by replicating the per-channel bias across N/H/W into a
    full-shape constant (since `add()` had no broadcasting yet); once broadcasting landed (see
    below) this simplified to reshaping the bias to `[1,C,1,1]` and a plain broadcasting `add()` —
    re-ran `YuNetFaceDetection` after the simplification to confirm no regression.
  - **Remaining known gaps** (each throws a clear error rather than silently misbehaving): `Gemm`
    with transA/transB, multi-output nodes, a non-constant `Reshape`/`Resize` shape/scale input,
    any other unmapped op type.
- [x] Public API: `inc/campello_nn/onnx_importer.hpp` — `OnnxImportResult` (`{graph, inputs:
      name->TensorDescriptor, outputs: name->TensorDescriptor}`, since the caller needs the
      shapes/dtypes to create matching `Tensor`s). **Decision (caught in review):** split into
      `importOnnxFromMemory(context, data, size)` (the real implementation) and
      `importOnnxFromFile(context, path)` (a convenience wrapper around it) rather than a single
      path-only function — a path-only API breaks on Android, where APK assets aren't accessible
      via normal filesystem paths (need `AAssetManager`-read bytes instead). Mirrors
      `campello_image::Image::fromMemory`/`fromFile`'s existing split for the same reason. Output
      `TensorDescriptor`s now use `operandShapeForImport` on the actual built graph rather than
      trusting the file's declared output shape (more robust against a declared dynamic dim).
- [x] Test (synthetic): `tests/universal/test_onnx_importer.cpp` imports a real, valid `.onnx` file
      (Conv→Add→Relu, generated via `tests/fixtures/generate_conv_add_relu_onnx.py` + Python's
      `onnx` package, checked into the repo as `tests/fixtures/conv_add_relu.onnx`) and checks
      output against the same hand-computed values used by `CpuOps.Conv2d`.
- [x] **Test (real model, real images):** `tests/universal/test_yunet_face_detection.cpp` —
      imports YuNet (`yunet_n_320_320.onnx`, from
      [ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train), BSD
      3-Clause; see `tests/fixtures/NOTICE.md`), decodes two real images via `campello_image`
      (FetchContent, opt-in via the new `BUILD_MODEL_TESTS` CMake option since it pulls in
      `basis_universal`+`libwebp`), resizes them to 320×320 using campello_nn's *own* `resize` op
      (no separate image-resize code), converts to BGR (not RGB!) `[0,255]` NCHW float — matching
      libfacedetection.train's actual training preprocessing
      (`yunet_train/tasks/face/transforms.py`: `mean=(0,0,0), std=(1,1,1), to_rgb=False`) — runs
      the model, and combines `cls`/`obj` outputs into a confidence score exactly as
      `yunet_train/tasks/face/postprocess.py` does (`scores = max_scores * flatten_objectness`).
      **Measured result: 0.83 confidence on a real face photo vs. 0.0005 on an abstract image — a
      ~1800x margin.** Confirmed bit-identical on the real MPSGraph/GPU backend during
      development (test itself only runs CPU, since it's a "universal" test). Image fixtures
      (`tests/fixtures/images/`) sourced from Wikimedia Commons (public domain) and a macOS
      system wallpaper — see `tests/fixtures/images/NOTICE.md` for provenance.
  - Sourcing note: `opencv_zoo`'s own copy of YuNet and its demo images are Git-LFS-tracked, and
    that repo's LFS bandwidth quota was exhausted at the time — fetched the model from the
    upstream training repo instead (not LFS-tracked there), and the face photo from Wikimedia
    Commons via its real search API (not a guessed URL).
  - Minor build wrinkle, not a bug: `campello_image`'s own `option(BUILD_TESTS ...)` shares a name
    with this project's, so `-DBUILD_TESTS=ON -DBUILD_MODEL_TESTS=ON` together also builds
    `campello_image`'s ~35 own tests as a side effect. Harmless (everything still passes), just
    extra build time; not worth a CMake-scoping fix for a non-bug.

### 4a-followup. NumPy-style broadcasting for `add`/`mul`
- [x] `GraphBuilder::add`/`mul` now infer a broadcast output shape (`computeBroadcastShape()` in
      `src/pi/graph_builder.cpp` — shapes aligned from the right, size-1/missing leading dims
      broadcast, NumPy/ONNX rules) instead of requiring an exact shape match. Genuinely
      incompatible shapes still throw `std::runtime_error`.
- [x] CPU backend: `evalBroadcastBinaryOp<BinOp>()` in `src/cpu/ops.cpp` — fast path when both
      operand shapes already match the output, strided fallback (`broadcastInputIndex()`) otherwise.
      `evalAdd`/`evalMul` now share this via lambdas.
- [x] MPSGraph backend: no code change needed — `additionWithPrimaryTensor:`/
      `multiplicationWithPrimaryTensor:` broadcast natively. Confirmed empirically, not just
      from SDK docs (`MpsOps.AddBroadcastRowVector`/`MulBroadcastColumnVector` on real GPU hardware).
- [x] Tests: `CpuOps.AddBroadcastRowVector`/`MulBroadcastColumnVector` and the MPSGraph mirrors
      above; `GraphBuilderValidation.AddShapeMismatchThrows` confirms incompatible shapes still
      throw.
- [x] **Follow-up simplification:** `onnx_importer.cpp`'s `Conv` bias handling, previously a manual
      full-shape bias replication (see Phase 4a notes), simplified to reshape `[C]` → `[1,C,1,1]`
      plus a plain broadcasting `add()`. Re-ran `YuNetFaceDetection` afterward — same thresholds
      pass, confirming no regression.

### 4b. TFLite import ✅
- [x] **Decision:** `FetchContent` Google's official `flatbuffers` header-only runtime
      (`FlatBuffers::FlatBuffers` interface target, same `FetchContent` pattern as
      GoogleTest/DirectML) rather than hand-rolling a FlatBuffers reader the way
      `proto_reader.hpp` hand-rolls protobuf — FlatBuffers' vtable + byte-offset
      indirection is a meaningfully higher bug-risk format to reimplement blind than
      protobuf's flat tag+value scheme. No `flatc` codegen step, no generated schema
      header: `src/tflite/tflite_parser.cpp` hand-writes field-ID accessors directly on
      `flatbuffers::Table`/`Vector`, verified against the real schema (now at
      `google-ai-edge/LiteRT`'s `tflite/converter/schema/schema.fbs` — the old
      `tensorflow/tensorflow` path moved during the TFLite→LiteRT migration).
- [x] Parse a TFLite `Model` → `SubGraph` (operators, tensors, buffers) — `tflite_model.hpp`/
      `tflite_parser.cpp`, mirroring `onnx_model.hpp`/`onnx_parser.cpp`'s shape. Only
      subgraph 0 is imported, same single-graph assumption the ONNX importer makes.
- [x] Op-mapping table (`tflite_importer.cpp`'s `applyOperator()`, mirrors `onnx_importer.cpp`'s
      `applyNode()`): `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `MUL`, `RELU`, `LOGISTIC` (sigmoid),
      `MAX_POOL_2D`/`AVERAGE_POOL_2D`, `RESHAPE`, `TRANSPOSE`, `SOFTMAX`, `CONCATENATION`, `PAD`,
      `GATHER`, `FULLY_CONNECTED`, `BATCH_MATMUL`, `RESIZE_BILINEAR`/`RESIZE_NEAREST_NEIGHBOR`,
      `QUANTIZE`/`DEQUANTIZE`.
- [x] **`DEPTHWISE_CONV_2D`:** weight layout (`[1,filter_height,filter_width,output_depth]`) and
      output-channel numbering (`oc = m + ic*depth_multiplier`) verified against TFLite's own
      reference kernel source (`tflite/kernels/internal/reference/depthwiseconv_float.h`), not
      guessed — confirms the channel ordering already matches the standard "groups=input_channels"
      grouped-conv convention `conv2d()`'s `groups` parameter implements (and ONNX import already
      relies on for `Conv`'s `group` attribute), so no extra channel-reordering trick was needed
      beyond a `[1,H,W,outC]`→`[outC,1,H,W]` byte permute. Verified end-to-end with a hand-computed
      fixture (`tests/fixtures/depthwise_conv2d.tflite`), not just trusting the layout claim.
- [x] **`BATCH_MATMUL`:** `adj_x`/`adj_y` confirmed against `tflite/kernels/batch_matmul.cc`
      ("transpose the last two dimensions") — `GraphBuilder::matmul()` has no transpose flag, so
      these become an explicit `transpose()` of the operand's last two axes
      (`swapLastTwoAxes()` in `tflite_importer.cpp`) before the matmul. Verified with a
      hand-computed `adj_y=true` fixture (`tests/fixtures/batch_matmul.tflite`).
- [x] **Decision:** NHWC/OHWI ↔ NCHW/OIHW conversion happens *only* at the graph boundary —
      a single `transpose()` right after each graph input and right before each graph output
      (rank-4 tensors only), not per-op. `CONV_2D`/`FULLY_CONNECTED` weight constants are
      pre-transposed at import time (bytes reordered once, zero added runtime ops), same
      trick as the ONNX importer's Conv-bias `[C]`→`[1,C,1,1]` reshape. **Known limitation:**
      axis-bearing ops (`CONCATENATION`/`SOFTMAX`/`GATHER`) remap TFLite's NHWC-numbered axis
      to NCHW assuming no intervening `RESHAPE`/`TRANSPOSE` has already broken that
      correspondence — documented in `tflite_importer.cpp`, not exercised by the current test.
- [x] TFLite's quantization (scale/zero_point on the *tensor*, not the op) handled by reading
      `Tensor.quantization` directly for `QUANTIZE`/`DEQUANTIZE`'s campello_nn-side
      `scale`/`zeroPoint` arguments — no separate fused-attribute path needed.
- [x] Test: `tests/universal/test_tflite_importer.cpp` against synthetic fixtures (hand-written
      JSON + `flatc --binary`, see `fixtures/NOTICE.md`): `conv_add_relu.tflite` computes the
      identical graph as `conv_add_relu.onnx` (same expected values, checked against both
      importers); `depthwise_conv2d.tflite` and `batch_matmul.tflite` each check against
      independently hand-computed expected values (not just round-tripped through the importer
      itself) — see those tests' comments for the by-hand arithmetic.
- [x] **Test against a real TFLite vision model end-to-end**, the way Phase 4a's YuNet test
      validates the ONNX importer: `tests/universal/test_blazeface_face_detection.cpp` imports
      MediaPipe's BlazeFace short-range face detector (`blaze_face_short_range.tflite`, the
      official "float16" export, Apache 2.0; see `tests/fixtures/NOTICE.md`), reuses the same two
      real test images as the YuNet test (`tests/fixtures/images/`), resizes to 128×128 via
      campello_nn's own `resize` op, converts to RGB (not BGR — MediaPipe's
      `ImageToTensorCalculator` uses RGB, unlike YuNet's OpenCV-trained BGR) normalized to
      `[-1, 1]` (confirmed against MediaPipe's actual `face_detection.pbtxt`:
      `output_tensor_float_range { min: -1.0 max: 1.0 }`), runs the model, applies sigmoid to the
      raw `classificators` logits (this exported graph, unlike YuNet's, has no baked-in
      `Sigmoid`), and takes the max over all 896 anchors. **Measured: ~0.47 on the face photo vs.
      ~0.23 on the abstract no-face image** — a real but modest ~2x margin, not YuNet's ~1800x one
      (the face photo is a conservatively-framed 1911 portrait, not this model's close-up-selfie
      training distribution; the test also deliberately skips MediaPipe's own letterbox
      preprocessing — measured to *shrink* the face further and *lower* the score for this
      particular fixture — and the real product's anchor-decode/NMS/`min_score_thresh=0.5` stage).
  - Getting this model importing required two fixes neither synthetic fixture had exercised:
    - **`PAD`** (BlazeFace pads the smaller residual branch's channel count to match the main
      branch before an `ADD`): no dedicated core `pad` op was added — confirmed via the actual
      `Paddings` constant bytes that every instance in this model is zero-padding, after-only, on
      the channel axis alone, so it's expressed as `concat(x, zeros)` along that axis instead.
      Every other axis/before-padding combination throws rather than guessing.
    - **Float16-weight `DEQUANTIZE`:** this model's weights are FLOAT16, and the export wraps
      every weight/bias constant in a `DEQUANTIZE` that's a pure precision cast (no
      `QuantizationParameters` on the input), not real int8 dequantization — handled as a no-op
      pass-through (the CPU backend already decodes Float16 constants to Float32 transparently),
      with a `dequantSource` redirect map so `CONV_2D`/`DEPTHWISE_CONV_2D`/`FULLY_CONNECTED`'s
      raw-byte weight permutation can still find the real underlying constant tensor through the
      `DEQUANTIZE` wrapper.
    - Also needed `RESHAPE`'s `-1` inference (`resolveReshapeTarget()`), mirroring the ONNX
      importer's own — `GraphBuilder::reshape()` itself has no `-1` support.

### 4c. Graph caching (serialize/load a compiled `Graph`)
- [x] **Decision:** caches the backend-agnostic `GraphIR` (built by `GraphBuilder`, normally
      handed straight to `Backend::compileGraph()` and discarded), not a backend-compiled native
      object. This skips GraphBuilder reconstruction (op-by-op calls, or re-parsing a source model
      file like ONNX) on load, but `compileGraph()` still runs on every load — there is no
      backend-specific compiled-graph cache (e.g. no cached `MPSGraph*`). Simpler, and the IR is
      genuinely backend-agnostic so one cache file works against any backend; revisit only if
      `compileGraph()` itself becomes the bottleneck for some backend.
- [x] Serialization format (`src/pi/ir_serialization.{hpp,cpp}`, internal): explicit
      field-by-field binary writer/reader (magic `"CNNG"` + `uint32` version, not a struct
      memcpy — portable across compilers/alignment), covering every `Node` field including
      `Conv2dDescriptor`/`Pool2dDescriptor`/`ResizeDescriptor`. Versioned; `deserializeGraphIR()`
      throws `std::runtime_error` on a bad magic, unsupported version, or truncated buffer (every
      read is bounds-checked, never reads past the buffer).
- [x] `GraphBuilder::serialize(outputs)` (instance method, mirrors `build()`'s signature) —
      serializes the same IR `build()` would otherwise compile, without compiling it.
      `GraphBuilder::deserialize(context, data, size)` (static) — compiles a graph directly from
      those bytes, skipping `GraphBuilder` reconstruction.
- [x] Public API: `inc/campello_nn/graph_cache.hpp` — `GraphCacheResult` (mirrors
      `OnnxImportResult`'s shape: `{graph, inputs, outputs}`), `loadGraphFromMemory`/
      `loadGraphFromFile` (mirrors the ONNX importer's Memory/File split), `saveGraphToFile`.
      `inputs`/`outputs` descriptors are derived by walking the deserialized IR's `Input` nodes
      and `outputs` list — deserializes the IR a second time (once for descriptors, once inside
      `GraphBuilder::deserialize()` for compilation); accepted as a one-time cache-load cost rather
      than plumbing a `GraphIR` through a public-header-safe API.
- [x] Round-trip tests (`tests/universal/test_graph_cache.cpp`): `gelu`, `conv2d`+`resize` (with
      non-default descriptor fields, including `nearestRoundsDown`), a `constant` node's raw
      bytes, and a real file round trip via `saveGraphToFile`/`loadGraphFromFile` — each compares
      the cached graph's dispatch output against the same graph built directly (no cache). Plus
      `CorruptMagicThrows`/`TruncatedBufferThrows`/`LoadGraphFromFileMissingFileThrows`. Mirrored
      on the real MPSGraph/GPU backend (`tests/platform/test_mps_graph_cache.cpp`,
      `MpsOps.GraphCacheRoundTrip`) to confirm a cached graph compiles correctly there too, not
      just on the CPU reference backend.
- [ ] Not yet done: wiring this into the ONNX importer (e.g. `importOnnxFromFile` transparently
      checking for a sibling `.campellocache` file) — `onnx_importer.cpp`'s internal `GraphBuilder`
      instance isn't exposed today, so there's no way for a caller to get at its IR to serialize.
      Left as future integration work, not required for the generic caching mechanism itself.

---

## Phase 5 — `campello_llm` (and future `campello_vision`)

Goal: implement the layer described in §4 of the architecture doc — everything no graph format
(ONNX/TFLite) and no WebNN-shaped API models. **Scoped specifically to weight-only formats**
(safetensors, gguf — files with no embedded graph topology, where the architecture must be
supplied by code that already knows it). Models distributed as ONNX/TFLite skip this layer
entirely and go through `campello_nn`'s own importer (Phase 4) instead.

- [ ] `GenerationConfig` struct (maxTokens, temperature, topP, topK)
- [ ] Tokenizer support (start with one format, e.g. BPE/SentencePiece compatible with common
      LLaMA/GPT tokenizer files; chat template handling)
- [ ] Weights-file parsing for safetensors/gguf (architecture-agnostic byte/header format —
      could in principle be a tiny shared leaf dependency with a future `campello_vision` if it
      ever needs to read the same container format for raw vision weights; not urgent, nothing
      on the vision side needs it yet)
- [ ] Architecture registry: per-architecture (LLaMA-style, GPT-style) graph wiring that calls
      `GraphBuilder` ops per layer with weights bound as `constant()`s — this is the
      architecture-specific knowledge that justifies this layer existing separately from
      `campello_nn`
- [ ] `Model::load()`: ties tokenizer + weights + architecture wiring + `campello_nn::Context`
      together; should also support loading a pre-cached graph via `campello_nn`'s own graph
      caching (Phase 4c) instead of rebuilding one live
- [ ] Prefill: single graph dispatch over the full prompt
- [ ] KV-cache: explicit `Tensor`s fed back as inputs each decode step, read back out, growing
      cache management (pre-allocate vs. dynamic growth)
- [ ] Decode loop: one graph dispatch per token, feeding KV-cache deltas
- [ ] Sampling on CPU after reading back logits: temperature scaling, top-k, top-p, (consider
      greedy/argmax as a baseline mode too)
- [ ] Streaming: `generate()`'s `onToken` callback invoked per generated token
- [ ] Stop conditions: max tokens, EOS token, stop sequences
- [ ] Tests: deterministic generation test (temperature=0/greedy) against a known-good reference
      output for a small test model
- [ ] Public headers finalized under `campello_llm/include/`

### Future: `campello_vision`
Not started, not urgent given Phase 4's ONNX/TFLite import covers most "standard" vision models
already. Would only be needed for weight-only-format vision models (e.g. a raw PyTorch state
dict with no ONNX export) — same role as `campello_llm`, same split, no generation loop/KV-cache
to own. Revisit if/when a concrete weight-only vision model needs it.

---

## Phase 6 — Testing, Benchmarking, Examples

- [ ] Cross-backend conformance suite: same graph, same inputs, run on every available backend on
      a given platform, assert outputs agree within per-dtype tolerance
- [ ] Performance benchmarks: prefill throughput, decode tokens/sec, per-op latency, per backend
- [ ] Example: minimal `campello_nn` program building/running a hand-written graph (no LLM)
- [ ] Example: import and run a real ONNX vision model end-to-end (the face-detection test from
      Phase 4a is a natural candidate to promote here)
- [ ] Example: minimal `campello_llm` CLI chat/completion demo using a small open-weights model
- [ ] Example: graph caching end-to-end (build/import a graph, serialize it, then load the cached
      graph in a second run) to demonstrate the startup-cost savings claim from §5

---

## Phase 7 — Documentation & Packaging

- [ ] API reference docs for `campello_nn` (Context/Tensor/GraphBuilder/Graph) and `campello_llm`
      (Model/GenerationConfig)
- [ ] Backend support matrix doc (mirrors §2 table, kept current with actual implementation state
      instead of the aspirational one in the architecture doc)
- [ ] Build/integration instructions per platform (toolchains, SDK versions required per backend)
- [ ] Versioning and release process for `campello_nn`'s serialized/cached `Graph` format
- [ ] CONTRIBUTING notes on adding a new backend (what `Backend` interface methods must be
      implemented, how to add parity tests)

---

## Open Questions

- [x] **Relationship to `campello_gpu`:** separate repo/build (`~/Documents/GitHub/campello_gpu`),
      but `campello_nn` mirrors its conventions — `systems::leal::` namespace, the handle-based
      `void*` pattern, per-platform `.cmake` files, GTest-via-FetchContent test setup. No shared
      build system or shared types yet. Dispatch uses a `Fence` (matching `campello_gpu`'s
      submit-plus-fence model), not a new `Future<void>` primitive.
- [x] Whether the DirectML backend (Phase 3b) should share a device/command-queue with
      `campello_gpu` on Windows, or stay fully independent — **resolved (user decision):**
      fully independent, owns its own `ID3D12Device`/command queue, same as the MPSGraph
      backend doesn't share Metal state with `campello_gpu` either.
- [ ] Minimum viable platform set for v1 — doc flags Linux/Android as inherently best-effort; decide
      if v1 ships without them or with CPU-only fallback on those platforms
- [x] Float16 and quantized (Int8) op support — done, see "Float16/Int8/Uint32 Support" above.
      Per-tensor scale/zero-point only; per-channel weight quantization deferred.
- [x] **Where does model-loading capability live?** Resolved (see conversation): `campello_nn`
      owns importing graph-format files (ONNX/TFLite) and caching its own compiled `Graph`s,
      since both need zero architecture-specific knowledge. `campello_llm`/future
      `campello_vision` own weight-only formats (safetensors/gguf) specifically because those
      need externally-supplied architecture knowledge to wire up. No separate
      `campello_nn_convert` project. Checked: none of our backends (MPSGraph, DirectML,
      oneDNN/Vulkan, NNAPI, WebNN) have native ONNX/TFLite loading to rely on instead — by
      design, they're all WebNN-shaped graph-*building* primitives with zero file-format
      opinions, same as WebNN itself.
- [ ] protobuf dependency for ONNX import (Phase 4a) — vendor the full `protobuf` library via
      FetchContent, or hand-roll a minimal parser scoped to just ONNX's wire format/schema?
- [ ] FlatBuffers dependency for TFLite import (Phase 4b) — same question, different library
- [ ] Threading/concurrency model for `Context::dispatch` — single dispatch queue per `Context` or
      concurrent dispatches allowed? (Currently moot: CPU backend dispatch is synchronous and
      single-threaded.)
- [ ] Library type: currently `STATIC` in all platform `.cmake` files, unlike `campello_gpu`'s
      `SHARED` — revisit when Windows symbol-export conventions matter (Phase 3b)

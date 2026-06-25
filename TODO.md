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

### Future: CPU backend performance
**Motivation:** `src/cpu/ops.cpp` is currently plain scalar loops — no SIMD, no multithreading.
With Phase 3c/3d's GPU-acceleration story now resting on a from-scratch `campello_gpu` compute
backend (see Phase 3 below) instead of borrowing a vendor inference runtime, the CPU backend stays
load-bearing for longer in practice (the only backend with zero external dependency, and what
every device falls back to before/without that work landing) — worth investing in directly rather
than treating it as a placeholder.
- [x] **Multithread the per-op evaluators.** New `src/cpu/thread_pool.hpp`/`thread_pool.cpp`: a
      process-wide `ThreadPool` (`std::thread::hardware_concurrency()` workers, lazily constructed
      singleton) plus a `parallelFor(begin, end, grainSize, body)` fork-join helper (`std::latch`
      for the join). Wrapped the 10 compute-bound kernels in `ops.cpp` — `evalBroadcastBinaryOp`
      (`add`/`mul`), `evalGelu`/`evalRelu`/`evalSigmoid`, `evalSoftmax`, `evalLayerNorm`/
      `evalRmsNorm`, `evalBatchNorm`/`evalInstanceNorm`, `evalMatMul`/`evalGemm`, `evalConv2d`,
      `evalPool2d`, `evalResize` — leaving the pure data-movement ops (`reshape`/`transpose`/
      `concat`/`slice`/`gather`/`quantizeLinear`/`dequantizeLinear`) untouched as a smaller-expected-
      gain follow-up. **Decision:** where the existing code nested `for(n) for(c)`/`for(n) for(o)`,
      flattened to one `N*C`/`N*O` range first — wrapping just the outer `n` loop would give zero
      parallelism whenever `N == 1` (the common inference case). Grain sizes per op category are
      initial heuristics (chosen so today's small unit-test shapes stay on `parallelFor`'s serial
      fast-path — confirmed via a full `ctest` run, all 137 tests pass unchanged); real tuning is
      Phase 6 benchmark work. **Platform guard:** on Emscripten without pthreads enabled (the
      default — `cmake/wasm.cmake` doesn't pass `-pthread`/`-sUSE_PTHREADS=1`), `parallelFor`
      degrades to calling `body` inline and `ThreadPool` is never linked in at all
      (`CAMPELLO_NN_CPU_THREADING_ENABLED` in `thread_pool.hpp`) — every other platform gets the
      real pool. New `tests/universal/test_cpu_threading.cpp`: existing tests all use small shapes
      that never leave `parallelFor`'s serial path, so these add shapes sized above each grain
      threshold (300k-element `add`, batched 64×64×64 `matmul`, 256-row `layerNorm`, `N=1` `conv2d`/
      `avgPool2d` to specifically prove the N\*C/N\*O flattening) with independently-computed
      expected values, to actually exercise and verify the multi-threaded path. Informal benchmark
      (512×512×512 matmul, scratch program, not checked in): 63ms threaded vs. 222ms naive-serial on
      a 6-core dev machine (~3.5x), checksums matching — real speedup confirmed, not just "doesn't
      crash."
- [x] **SIMD the hot inner loops.** Resolved the earlier-deferred AVX2-vs-NEON verification gap with
      a portable SIMD library (**xsimd**, FetchContent'd, `GIT_TAG 14.2.0`) instead of hand-rolled
      per-ISA intrinsics — one implementation, not two. New `src/cpu/simd.hpp`: `FloatBatch =
      xsimd::batch<float>` and `kSimdWidth`. **ISA baseline decision:** no `-mavx2`/`-march=native` —
      xsimd auto-detects from the compiler's default predefined macros, so with zero special flags it
      safely targets SSE2 (x86_64) / NEON (AArch64), both mandatory baselines, everywhere; real AVX2
      needs runtime CPU-feature dispatch to stay safe on non-AVX2 hardware, left as future work.
      Vectorized (each as a vector loop + scalar remainder inside the existing `parallelFor` chunk
      body from the threading round — the two compose with no interaction, since loads are
      `load_unaligned`/`store_unaligned`, never `load_aligned`, so chunk boundaries don't need
      SIMD-width alignment): `evalAdd`/`evalMul` fast path (made the `BinOp` lambdas generic —
      `[](auto x, auto y){ return x + y; }` — so the same callable works on `float` and `FloatBatch`),
      `evalGelu`/`evalSigmoid` (`xsimd::erf`/`xsimd::exp`), `evalRelu` (`xsimd::max`),
      `evalLayerNorm`/`evalRmsNorm`/`evalInstanceNorm` (vectorized mean/variance reduction via an
      accumulator batch + `xsimd::reduce_add`, then a vectorized normalize pass), `evalBatchNorm`
      (no reduction needed — pure vectorized affine), and **`evalMatMul`/`evalGemm`** — a real loop
      *reorder*, not a mechanical wrap: B is row-major `[K,N]`, so the old k-loop strided through
      memory per (m,n) and wasn't vectorizable directly; instead, for each `k`, broadcast `a[m,k]`
      and `xsimd::fma` it against a contiguous slice of B's row `k`, accumulating straight into the
      output row (bit-identical summation order over `k`, just N-wide groups at a time — confirmed
      by the existing `EXPECT_FLOAT_EQ`-based matmul/gemm tests still passing unchanged). `evalGemm`
      additionally vectorizes the `alpha*sum+beta*c` epilogue across all three of `c`'s broadcast
      shapes (scalar/row-vector/full-matrix), since `c`'s layout matches `out`'s row layout in every
      case. **Not bit-identical** (tolerated by existing `EXPECT_NEAR`-based tests, confirmed by
      reading them first): `gelu`/`sigmoid` (different transcendental approximation than libm) and
      the three reduction-based norm ops (lane-grouped partial sums before a horizontal reduce,
      reordering float summation). **Deliberately not vectorized this round** (separate, riskier
      follow-ups): `evalConv2d` (padding/dilation/stride bounds-checking breaks the simple
      contiguous-load pattern) and `evalSoftmax` (axis can be non-last, i.e. non-contiguous stride).
      New `tests/universal/test_cpu_simd.cpp`: shapes deliberately *not* a multiple of `kSimdWidth`
      for every vectorized op (to exercise the scalar-remainder tail) plus `gemm`'s three `c`-shape
      variants — all passing, plus the full existing suite (73 tests total) unchanged.
  - **Real discovery while benchmarking:** this project's documented build command (`CLAUDE.md`'s
    `cmake -B build -DBUILD_TESTS=ON`, no `-DCMAKE_BUILD_TYPE`) leaves `CMAKE_BUILD_TYPE` empty —
    unoptimized. Measured the vectorized matmul at **447ms/iter unoptimized vs. 222ms/iter for the
    plain scalar version it replaced** (512×512×512, scratch benchmark) — xsimd's abstraction layers
    don't get inlined away without optimization, making this round's change a real regression for
    anyone following the documented build steps literally. Fixed in `CMakeLists.txt`: default
    `CMAKE_BUILD_TYPE` to `Release` when the user/CI doesn't pick one explicitly (guarded by
    `NOT CMAKE_CONFIGURATION_TYPES`, so multi-config generators like Xcode/Visual Studio are
    untouched). With that fix, the same benchmark: **6.3ms/iter threaded+SIMD vs. 224ms/iter
    naive-serial-scalar (~35x) on this 6-core dev machine**, checksums matching.
- [ ] Benchmark before/after (ties into Phase 6's "Performance benchmarks" item) to confirm gains
      before adding complexity — the informal checks above (threading round and this one) are smoke
      tests, not a real harness; also worth re-measuring once a real harness exists, since both
      informal numbers were single-machine, single-run readings, not averaged/statistically robust
- [x] **AVX2+FMA3 runtime CPU-feature dispatch.** Unlike NEON, this dev machine's CPU (Intel Core
      i5-8500B, Coffee Lake) genuinely supports AVX2+FMA3, so this round's new code path was built
      *and run* end-to-end here, not compiled blind. Researched xsimd's actual documented dispatch
      pattern from the real source/docs in `build/_deps/xsimd-src` (not assumed): write each kernel
      as `template<class Arch> void evalXxxImpl(...)`, put the **definition** of the AVX2
      instantiation in a separate translation unit compiled with `-mavx2 -mfma`, reference it from
      everywhere else via `extern template` (compiling `xsimd::fma3<xsimd::avx2>` codegen in a TU
      *not* built with those flags would be invalid). The runtime cpuid check (confirmed in
      `xsimd/config/xsimd_cpuid.hpp`) is `xsimd::available_architectures().has(xsimd::fma3<xsimd::avx2>{})`
      — `fma3<avx2>` is xsimd's combined "AVX2 + FMA3" arch tag, since they're technically separate
      cpuid bits even though virtually every real AVX2 chip since Haswell has both. New
      `src/cpu/simd_kernels.hpp`: the 9 vectorized kernel bodies from the prior SIMD round, moved
      here and templated on `Arch` (mechanically the same logic, `Batch`/`width` instead of the
      fixed `FloatBatch`/`kSimdWidth`) — plus named `AddOp`/`MulOp` functor structs replacing the
      generic lambdas `evalAdd`/`evalMul` used to pass in (lambda closure types are anonymous and
      can't appear in an `extern template` declaration across translation units, hence needing a
      nameable type), `extern template` declarations for `xsimd::fma3<xsimd::avx2>` of all 9, and a
      cached (function-local static, checked once not per-call) `cpuSupportsAvx2Fma()`. New
      `src/cpu/simd_kernels_avx2.cpp`: explicit instantiation definitions for all 9, x86_64-only
      (`CMakeLists.txt`: `-mavx2;-mfma` via `set_source_files_properties` on GCC/Clang, `/arch:AVX2`
      on MSVC — **MSVC path unverified**, can't test that compiler from this machine, same
      documented-but-unverified treatment already accepted for the DirectML backend). `ops.cpp`'s 9
      wrapper functions (`evalAdd`, `evalGelu`, `evalMatMul`, etc.) became a 4-line dispatch each:
      call the AVX2 instantiation if `cpuSupportsAvx2Fma()`, else `xsimd::default_arch` (mechanically
      unchanged from the already-tested prior round). The detection macro
      (`CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED`) is derived directly from compiler-predefined
      architecture macros in `simd_kernels.hpp` itself (`__x86_64__`/`_M_X64`/etc.), same pattern as
      `thread_pool.hpp`'s `CAMPELLO_NN_CPU_THREADING_ENABLED` — no CMake `target_compile_definitions`
      needed. **Verified, not just compiled:** full `ctest` suite (73 tests) passes on this AVX2-
      capable machine, which is a genuine end-to-end run of the new path (`cpuSupportsAvx2Fma()`
      confirmed `true` here via a standalone scratch check, AVX2 batch width confirmed 8 vs. the
      default/SSE2 path's 4 when each is compiled with its real matching flags); a separate scratch
      check called `evalMatMulImpl<xsimd::default_arch>` and `evalMatMulImpl<xsimd::fma3<xsimd::avx2>>`
      directly on identical odd-sized (37×23×59) input and diffed every element — 0 mismatches across
      2183 elements, confirming the two paths agree exactly, not just that neither crashes. Informal
      benchmark (512×512×512 matmul, same scratch-program pattern as prior rounds): **4.26ms/iter
      AVX2-dispatched vs. 6.3ms/iter from the SSE2-only round (~33% further improvement) vs.
      217ms/iter naive-serial-scalar (~51x total)**, checksums matching.
- [ ] AVX-512 runtime dispatch — same infrastructure, one more arch tier; not done, no AVX-512
      hardware available to verify on
- [ ] Vectorize `evalConv2d` and `evalSoftmax` — deferred above, see those notes for why each is
      harder than the ops already done

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

### 3c/3d. Generic GPU backend on `campello_gpu` (`DeviceType::GpuGeneric`)
**Decision (supersedes the earlier separate Linux/Android plans below, and widens scope beyond
them):** instead of oneDNN (Linux) and NNAPI/LiteRT delegates (Android) — both vendor- or
third-party-runtime-dependent, with inconsistent OEM/SDK coverage — build one shared GPU backend
on top of the sibling `campello_gpu` library. Originally scoped to just Linux/Android (where
MPSGraph/DirectML have no equivalent), but widened on request: `campello_gpu`'s
`createComputePipeline()`/`dispatchWorkgroups()` are genuinely implemented across **all three** of
its native backends (confirmed by reading its DirectX12 source directly, not just its slightly
stale `shader_module.hpp` doc comment) — Metal (macOS/iOS), Vulkan (Linux/Android), DirectX12
(Windows) — so one integration reaches every platform. Added a new, additive
`DeviceType::GpuGeneric` (`inc/campello_nn/constants/device_type.hpp`) rather than overloading
`DeviceType::Gpu`: `Gpu` keeps routing to the platform-native backend (MPSGraph/DirectML) as the
default; `GpuGeneric` is an explicitly-selected addition, for benchmarking the two against each
other later (the actual goal: build this, then benchmark CPU vs. `GpuGeneric` vs. native `Gpu`) —
not a replacement. This also keeps the project's existing philosophy (own backend per accelerator,
verified against the real API, no third-party inference-runtime dependency) consistent with the
MPSGraph/DirectML backends, rather than introducing a different kind of dependency (NNAPI driver,
LiteRT delegate, oneDNN) for the platforms that lacked a native backend.

**How real GPU NN runtimes actually execute (research, informs the shape of this backend):**
production runtimes are a *sequence of kernel dispatches*, never one giant fused shader for a
whole model — fusion only ever spans a handful of ops at a time (data dependencies, dynamic
shapes, and register/shared-memory limits make a single mega-shader impractical regardless of
vendor). Two real strategies exist:
  - **Fixed precompiled-kernel library, dispatched per op** — DirectML (HLSL compute shaders
    shipped inside the DLL, selected/parameterized by shape at compile time) and most NNAPI vendor
    drivers / mobile NPU runtimes work this way. No runtime shader generation at all.
  - **Compile-time fusion / codegen** — MPSGraph (and CoreML, TensorRT, XLA/TVM) pattern-match
    subgraphs against known fusion templates and JIT-compile specialized kernels once at graph-
    build time (still many fused *chunks*, not one shader for the network).
  `campello_gpu`'s `ShaderModule` only accepts **precompiled native binaries** (SPIR-V for Vulkan,
  `.metallib` for Metal — no runtime shader-source compilation API), which makes the
  fixed-precompiled-kernel-per-op strategy both the realistic option and consistent with what
  DirectML/NNAPI drivers already do — not a step down from how real runtimes work. This also
  mirrors this repo's own DirectML backend's already-made "sequential per-node compiled operators,
  not a fused graph" decision (see 3b above). Compile-time fusion (MPSGraph-style) would need a
  GLSL/HLSL→SPIR-V compiler (e.g. `shaderc`) embedded as a new dependency to generate specialized
  SPIR-V at `compileGraph()` time — a real option later, out of scope for v1.

- [x] **Vertical slice implemented and verified on Metal: `relu`, exact-shape `add`, rank-2
      unbatched `matmul`; later expanded to also cover exact-shape `mul`, `sigmoid`, `gelu`,
      `layerNorm`, `rmsNorm`** (9 `OpKind`s total — see the dedicated entry below for the
      expansion). Everything else throws rather than guessing (matches this codebase's
      "throw on unsupported variant" precedent, e.g. ONNX's Gemm-with-transpose). New
      `src/gpu/gpu_backend.{hpp,cpp}` implements `Backend`: `createTensor`/`writeTensor`/
      `readTensor` via `Buffer::upload()`/`download()`; `compileGraph()` builds one
      `BindGroupLayout`/`PipelineLayout`/`ComputePipeline` per `OpKind` (cached, reused across
      nodes of the same kind — same precedent as MPSGraph/DirectML's "build once") plus per-node
      output/params buffers; `dispatch()` rebuilds each real op's `BindGroup` fresh every call
      (campello_gpu's `BindGroup` is immutable once created, WebGPU-style, unlike D3D12's
      rebindable descriptor tables DirectML's `resolveBuffer()` exploits — deliberately not
      optimized away for provably-static subgraphs in this round). New
      `src/gpu/shaders/{relu,add,matmul}.{comp,metal,hlsl}` — hand-written GLSL/MSL/HLSL per op
      (not a single cross-compiled source: `campello_gpu`'s `ShaderModule` only accepts
      **precompiled** native binaries, confirmed in the 3c/3d research below), with the compiled
      `.metallib`/`.spv` bytes embedded as generated C++ byte-array headers
      (`src/gpu/shaders/*_metallib.hpp`/`*_spv.hpp`) rather than loaded from a file path at
      runtime — avoids the "where do I find this file" problem on Android (no real filesystem for
      assets) or any other deployment target, same reasoning `campello_gpu`'s own precompiled-shader
      design already follows. **Binding convention** (all 3 ops): storage buffers at binding
      `0..N-1` (inputs then output), one uniform params buffer at binding `N` (element count for
      `relu`/`add`; `M`/`K`/`N` for `matmul`) — kept uniform across ops rather than special-casing
      which ones "need" params. **Dispatch model:** one workgroup per output element (not one
      thread per element) — discovered while verifying that `campello_gpu`'s Metal
      `ComputePassEncoder::dispatchWorkgroups()` always uses the pipeline's
      `threadExecutionWidth()` as the per-group thread count, ignoring whatever the shader source
      declares, with no public accessor to query that value before dispatch — so every shader gates
      to thread-0-within-group (`relu.metal`/`add.metal`/`matmul.metal`) to stay correct regardless
      of that unqueryable value, trading GPU utilization for correctness; the Vulkan/HLSL shaders
      use `local_size_x=1`/`[numthreads(1,1,1)]` so they genuinely have one thread per group and
      don't need the gate. New `DeviceType::GpuGeneric` wired into `Context::create()`
      (`src/pi/context.cpp`) ahead of the existing per-platform branches, on every platform.
  - **Two real bugs found in `campello_gpu` itself while verifying end-to-end on Metal, both now
    fixed upstream** (not assumed — found by writing a minimal standalone reproduction against
    `campello_gpu` directly, bypassing this backend entirely, after every `GpuGenericOps` test
    initially read back all-zero output despite no errors anywhere, including with
    `MTL_DEBUG_LAYER=1`/`MTL_SHADER_VALIDATION=1` enabled). Filed in `campello_gpu/TODO.md`'s own
    "Bugs" section; **fixed locally as `campello_gpu` `v0.13.3`** (not yet tagged/pushed upstream
    as of this writing — see the temporary `CAMPELLO_NN_CAMPELLO_GPU_LOCAL_DIR` CMake override
    below):
    1. **The actual cause:** `Device::createFence()`'s Metal `MetalFenceData::signaled` defaulted
       to `true` ("start signaled so first frame doesn't block" — a comment describing a
       ring-buffer-of-fences *rendering* pattern, not this backend's one-shot
       create-fence→submit→wait usage, which is also `campello_gpu`'s own documented "typical
       usage" example in `fence.hpp`). So `Fence::wait()` on a freshly created fence returned
       `true` immediately, without ever waiting for the submission it was passed to — every read
       happened before the GPU had necessarily even started, let alone finished. **Fixed in
       `campello_gpu` 0.13.3:** `Device::submit(cmdBuffer, fence)` now resets the fence to
       unsignaled right before `commit()`, mirroring the Vulkan backend's `vkResetFences()` call —
       safe for both a fresh fence and a reused ring-buffer one. `GpuBackend::dispatch()` now uses
       a real `campello_gpu::Fence` again (`createFence()` + `submit(cmdBuffer, fence)` +
       `fence->wait()`, blocking before returning — still hands back an always-pre-signaled
       `GpuFence`, matching CPU/MPSGraph/DirectML's synchronous-dispatch convention).
    2. **A separate, latent issue, not the proximate cause but real:** `Buffer::download()`'s
       Metal implementation did a raw `memcpy` from `buffer->contents()` with no
       `synchronizeResource:` call first — invalid/stale for a buffer in
       `MTLResourceStorageModeManaged` (Apple's docs require an explicit GPU-side sync blit before
       a CPU read can see a prior GPU write in that mode). `Device::createBuffer()` only picks the
       always-coherent `MTLResourceStorageModeShared` instead when `BufferUsage::mapRead` or
       `mapWrite` is set. **Fixed in `campello_gpu` 0.13.3:** `download()` now encodes and
       synchronously waits on a blit-encoder `synchronizeResource:` before the `memcpy`, gated on
       the buffer's storage mode being `Managed` (no-op for `Shared`/iOS). With the real fix in
       place, `tensorBufferUsage()` (`src/gpu/gpu_backend.cpp`) no longer forces `mapRead|mapWrite`
       — back to plain `storage|copySrc|copyDst`, letting buffers use the (more efficient on
       discrete-GPU hardware) `Managed` mode again.
  - **Temporary build wiring:** since the 0.13.3 fixes are only committed in the local sibling
    checkout (not yet tagged/pushed), `CMakeLists.txt` adds an opt-in
    `CAMPELLO_NN_CAMPELLO_GPU_LOCAL_DIR` cache variable — when set, `FetchContent_Declare` uses
    `SOURCE_DIR` pointing at it instead of fetching `GIT_TAG v0.13.2`; defaults to empty, so CI and
    every other machine still get the real fetch. **Revert once a release containing both fixes is
    tagged:** drop the override entirely, bump `GIT_TAG` past `v0.13.2`.
  - **Verified, not just compiled:** full `ctest` suite (77 tests: 73 prior + 4 new) passes on this
    Metal-capable machine, built against the real 0.13.3 fixes with no workarounds —
    `Relu`/`AddExactShape`/`MatMul` individually, plus a **chained** `Relu`→`Add` graph
    specifically added to test whether `campello_gpu` auto-tracks resource hazards between two
    dispatches in the same compute pass (no explicit barrier call exists in `ComputePassEncoder`'s
    API to confirm this otherwise) — it does, on Metal.
  - **Vulkan:** GLSL→SPIR-V compilation verified (`glslangValidator`, installed via
    `brew install glslang` for this), but **not execution** — no Vulkan ICD/MoltenVK on this
    machine. **DirectX12:** `.hlsl` sources written against real D3D12/HLSL semantics but entirely
    unverified — no Windows toolchain available — same documented-but-unverified treatment already
    accepted for the DirectML backend.
- [x] **Coverage expanded from the 3-op vertical slice to 9 `OpKind`s: `mul` (exact-shape),
      `sigmoid`, `gelu`, `layerNorm`, `rmsNorm`** (plus the original `Relu`/`Add`/`MatMul`/
      `Input`/`Constant`). New `src/gpu/shaders/{mul,sigmoid,gelu,layernorm,rmsnorm}.{comp,metal,
      hlsl}`, embedded the same `*_metallib.hpp`/`*_spv.hpp` generated-header way as the original
      three. **`mul`/`sigmoid`/`gelu`** reuse the existing one-workgroup-per-element dispatch model
      exactly (`ParamsElementwise{count}` at the last binding) — no new dispatch shape needed.
      **`layerNorm`/`rmsNorm`** needed a new dispatch model instead, since both reduce over the
      whole last dimension before producing any output: one workgroup per output *row*
      (`dispatchX = outerTotal = numElements(shape) / lastDim`), with the single active thread
      (still `local_size_x=1`/thread-0-gated, same correctness-over-utilization tradeoff as the
      original three shaders) looping over `lastDim` internally — mean/variance first pass, then a
      second pass writing the normalized row (`layernorm.comp`/`rmsnorm.comp`); formulas verified to
      match `src/cpu/ops.cpp`'s `evalLayerNorm`/`evalRmsNorm` exactly. New shared `ParamsNorm{lastDim,
      eps, pad1, pad2}` uniform struct (`gpu_backend.cpp`) — mixed `uint32_t`/`float` is fine at this
      size since a flat sequence of 4-byte scalars packs identically under GLSL std140, HLSL
      cbuffer, and MSL constant-struct rules (no vec2/vec3 alignment surprises to worry about).
      `eps`/`lastDim` come from `Node::floatAttr0`/`shape.back()` already on the IR node — no new
      `Node` fields needed. **`gelu`'s `erf`:** neither GLSL nor Metal's `metal_stdlib` has a
      built-in `erf()` (checked the real installed Metal toolchain headers directly, not assumed) —
      `gelu.comp`/`gelu.metal`/`gelu.hlsl` all use the identical Abramowitz & Stegun 7.1.26
      polynomial approximation (max absolute error ~1.5e-7) for cross-backend consistency; not
      bit-exact with libm's `erff()`/xsimd's `erf()` (the CPU backend's own two, already-disagreeing
      approximations), hence the new tests' `EXPECT_NEAR` rather than `EXPECT_FLOAT_EQ`. Five new
      `GpuGenericOps` tests (`MulExactShape`/`Sigmoid`/`Gelu`/`LayerNorm`/`RmsNorm`) plus a sixth,
      `ChainedGeluThenAdd`, added specifically to re-confirm the hazard-tracking question (see the
      bugs writeup above) holds for a row-reduction op feeding an elementwise one, not just the
      original `Relu`→`Add` elementwise-into-elementwise case — **verified, full suite now 83 tests
      (73 prior + 10 new — `ChainedReluThenAdd` already existed) passing on this Metal-capable
      machine.** Vulkan/DirectX12 status unchanged from the original three ops: GLSL compiles via
      `glslangValidator`, `.hlsl` is written but unverified, neither executed on real hardware here.
- [x] **Shape/data-movement ops added: `Reshape`, `Transpose`, `Slice`, `Concat`, `Gather`** (14
      `OpKind`s total now). Chosen as the next batch over vision/transformer ops specifically
      because they're prerequisites for any real *imported* (ONNX/TFLite) graph to run end-to-end
      on `GpuGeneric` at all — those import paths emit `Reshape`/`Transpose` routinely (NCHW/NHWC
      boundary conversion, `Conv` bias reshaping, etc.) even for graphs that are otherwise pure
      elementwise/matmul. **`Reshape`** is a zero-cost alias, no shader at all — same precedent as
      the DirectML backend's `resolveBuffer()`: `compileGraph()` allocates nothing for it, and
      `dispatch()` resolves it by pointing `resolved[i]` straight at its source node's already-
      resolved buffer (no recursion needed here, unlike DirectML's lazy `resolveBuffer()`, since
      this backend's `resolved[]` array is filled in IR dependency order — a `Reshape`'s source is
      always already resolved by the time the loop reaches it). **`Transpose`/`Slice`** needed a
      genuinely new, generic per-dim remap (arbitrary permutation/multi-axis slicing don't collapse
      to a simple loop shape the way the previous round's ops did): new `ParamsTranspose`/
      `ParamsSlice` structs in `gpu_backend.cpp` carry a precomputed row-major `divisor` (to decode
      a flat index into a per-dim index via repeated div/mod) and a `gatherStride`/`multiplier`
      (the corresponding offset into the source buffer) for each of up to `kMaxRank=8` dimensions —
      capped and throwing past that rather than guessing, same precedent as everywhere else in this
      codebase. **Deliberately flat named scalar fields (`divisor0`..`divisor7`, etc.), not real
      GLSL/HLSL arrays:** GLSL std140 and HLSL cbuffers both pad every array element inside a
      uniform block up to 16 bytes, but Metal's `constant` address-space structs use natural 4-byte
      packing for arrays instead — a real array would need a different byte layout per backend to
      stay byte-compatible with one shared C++ host struct. Flat scalar fields side-step the
      question entirely (confirmed already safe by `ParamsNorm`/`ParamsElementwise`'s existing
      scalar-only fields; this just has more of them) — each shader copies the named fields into a
      local array once at the top of `main()`/`computeMain()` for normal indexed-loop access.
      **`Concat`** is the one op this round that didn't fit the existing "one `OpResources` +
      one dispatch per node" model at all — unlike every other `OpKind`, its real input count
      varies per call site, but `OpResources`/`BindGroupLayout` are built once per `OpKind` and
      reused, assuming fixed arity. Solved by *not* trying to give it a variable-arity bind group:
      `compileGraph()` instead emits one `ConcatPiece` per input (own `axisOffset`-bearing params
      buffer, dispatch count = that input's own element count) that all dispatch against the
      *same* 1-input `OpResources` (`Concat`'s `numInputsFor() == 1`) and the same shared output
      buffer — `dispatch()` special-cases `OpKind::Concat` to loop over those pieces with N
      `setBindGroup`+`dispatchWorkgroups` calls instead of one. The per-piece math itself collapses
      to a 3-scalar outer/axis/inner split (`ParamsConcat`), not a full per-dim remap, since within
      one input's copy no other dimension changes size or order — same decomposition
      `evalConcat`/`evalGather` in `src/cpu/ops.cpp` are already logically equivalent to.
      **`Gather`** reuses that same outer/axisSize/innerSize split (`ParamsGather`) and fits the
      existing fixed-2-input dispatch model untouched; `indices` (Int32 or Uint32 per "Dtype
      Support" above) is bound as a raw `uint` storage buffer in the shader regardless of which —
      bit-identical for the non-negative indices this op (and the CPU backend, which never
      range-checks for negative values either) actually supports. `Concat`/`Gather`'s axis is read
      directly from `node.axis` with no negative-axis normalization needed in this backend, since
      `GraphBuilder::concat()`/`GraphBuilder::gather()` (`graph_builder.cpp`) already resolve it to
      a non-negative index before the node ever reaches a backend. New
      `src/gpu/shaders/{transpose,slice,concat,gather}.{comp,metal,hlsl}`, compiled/embedded the
      same way as every previous round (`glslangValidator`/`xcrun metal`+`metallib`, byte arrays in
      generated `*_spv.hpp`/`*_metallib.hpp` headers). **Verified:** 6 new `GpuGenericOps` tests
      (`ReshapeThenAdd` — chained specifically to confirm alias resolution, not just compiling;
      `TransposeRank3` — a true 3-axis permutation, not just a rank-2 swap; `Slice2D` — slices two
      dimensions at once, not just one axis on a rank-1 tensor; `ConcatThreeInputs` — exercises the
      `concatPieces` loop past the trivial 2-piece case; `Gather`/`GatherUint32Indices`), full suite
      now 89 tests (83 prior + 6 new) passing on this Metal-capable machine. Vulkan/DirectX12
      status unchanged: GLSL compiles via `glslangValidator`, `.hlsl` is written but unverified,
      neither executed on real hardware here.
- [x] **`Softmax` and `Gemm` added** (16 `OpKind`s total now) — completes the original
      transformer-block op set on `GpuGeneric` (everything else it needs — `Add`/`Mul`/`Gelu`/
      `LayerNorm`/`RmsNorm`/`MatMul`/`Reshape`/`Transpose`/`Slice`/`Concat`/`Gather` — was already
      implemented by the two previous rounds). **`Gemm`** is a direct extension of the existing
      `matmul.comp`'s one-workgroup-per-(m,n)-output-element model: same naive K-loop, plus
      `alpha`/`beta`/bias-`C` folded into the same dispatch (`out = alpha*(A@B) + beta*C`, `C`
      broadcasting by its own element count — `cElems==1`/`==n`/`==m*n`, exactly mirroring the CPU
      backend's `evalGemmImpl` in `src/cpu/simd_kernels.hpp`) rather than composing three separate
      dispatches (matmul, then scale, then broadcast-add) the way the MPSGraph backend composes it
      from primitive graph ops — one fused kernel was simpler here since this backend has no
      graph-level op fusion to lean on anyway. `GraphBuilder::gemm()` (`graph_builder.cpp`) already
      rejects anything but rank-2 `a`/`b` with no transA/transB, so nothing further to validate in
      `compileGraph()`. **`Softmax`** needed a genuinely new dispatch shape: one workgroup per
      output *row* like `layernorm.comp`/`rmsnorm.comp`, but unlike those two — whose reduction
      axis is always the tensor's last dimension — softmax's axis can be *any* dimension
      (`GraphBuilder::softmax()` accepts and resolves a negative axis to any non-negative index, no
      "must be last" restriction). Reused `transpose.comp`/`slice.comp`'s generic per-dim-decode
      machinery (flat named `divisor`/`origStride` scalar fields, capped at `kMaxRank=8`, same
      reasoning as those two for why not real arrays) to unravel a row index into the per-dim
      offset of every dimension *except* the axis being reduced over, then runs the same
      numerically-stable three-pass body (max, exp+sum, normalize) as the CPU backend's
      `evalSoftmax`. **Verified:** 4 new `GpuGenericOps` tests (`Gemm`/`GemmBroadcastCRow` —
      the latter specifically exercising the `cElems==n` branch the former's full-`[M,N]` `C`
      doesn't touch; `SoftmaxLastAxis`/`SoftmaxNonLastAxis` — the latter softmaxing over axis 0 of
      a rank-3 tensor specifically to exercise the generic outer-decode, since a last-axis-only
      test can't catch a bug in skipping a non-last dimension), full suite now 93 tests (89 prior +
      4 new) passing on this Metal-capable machine. Vulkan/DirectX12 status unchanged: GLSL
      compiles via `glslangValidator`, `.hlsl` is written but unverified, neither executed on real
      hardware here.
- [x] Vision/normalization ops on `GpuGeneric`: `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Resize`,
      `BatchNorm`, `InstanceNorm` — shader sources already existed; generated embedded SPIR-V/
      Metal byte arrays and wired them into `gpu_backend.cpp`; parity tests pass on Metal.
- [x] Quantization ops on `GpuGeneric`: `QuantizeLinear`, `DequantizeLinear` — added GLSL/Metal
      byte-addressed shaders (HLSL written but unverified, same status as the rest of the DirectX
      path), wired into `gpu_backend.cpp`, and verified with round-trip and `QuantizedMatmul` tests
      on Metal.
- [x] **`add`/`mul` broadcasting and batched `matmul` on `GpuGeneric`.** New
      `src/gpu/shaders/broadcast_binary.{comp,metal,hlsl}` implements NumPy/ONNX-style broadcasting
      for `add` and `mul` (shapes aligned from the right, size-1 dims broadcast) via a single shared
      kernel parameterized by rank/output shape and the two operands' strides; `matmul` now accepts
      matching batch dimensions and dispatches `batchCount` as `dispatchZ`. Tests:
      `GpuGenericOps.AddBroadcast`/`MulBroadcast`/`MatMulBatched` pass on Metal.
- [x] **End-to-end real-model test on `GpuGeneric`.** `tests/platform/test_gpu_generic_models.cpp`
      imports YuNet (`yunet_n_320_320.onnx`) and runs the same face-vs-no-face confidence check as
      the CPU/MPSGraph model tests, using `campello_image` to decode/resize the real fixtures. Passes
      on Metal. Required a fix for zero-byte ONNX initializers: `GpuBackend::compileGraph()`'s
      `Constant` node path now uses the no-initial-data `createBuffer(size, usage)` overload when
      `size == 0` instead of the data-carrying overload, avoiding a `memcpy` from `nullptr` inside
      `campello_gpu::Buffer::upload()`.
- [x] **Backend latency benchmark.** Added `benchmarks/benchmark_backends.cpp` and a
      `BUILD_BENCHMARKS` CMake option. It runs a synthetic transformer-block graph
      (`matmul → add → gelu → layerNorm`) on every available backend and reports
      min/median/mean/max latency plus max absolute difference vs. the CPU reference.
      Verified on macOS/Metal: CPU and MPSGraph are both faster than the current naive
      `GpuGeneric` implementation across the tested 1×512/1×1024/1×2048/1×4096 and batched
      16–64×1024 shapes, confirming the benchmark is useful for tracking future
      `GpuGeneric` performance work.
- [ ] Real Vulkan execution verification (Linux/Android hardware or `llvmpipe`/Mesa software
      Vulkan) and any real DirectX12 verification (Windows toolchain) — both currently
      compile-only-or-less, as noted above
- [ ] `Context` backend wiring replaces nothing for Android — `android_backend.{hpp,cpp}`'s NNAPI
      path stays as documented history (no `.cpp` was ever written for it), separate from this
      backend's own `DeviceType::GpuGeneric` selection
- [ ] CI coverage for the Vulkan/DirectX12 paths — GitHub-hosted runners have no GPU; DirectML's
      WARP-software-adapter trick has a Vulkan equivalent in Mesa's `llvmpipe`, worth investigating,
      not solved now
- [ ] `dispatch()`'s "rebuild every real op's `BindGroup` every call" is correctness-first, not
      perf-first (see above) — worth optimizing once there's a benchmark to justify it

<details>
<summary>Superseded plans (kept for history)</summary>

#### 3c (superseded). Linux — oneDNN or Vulkan compute shaders
- [ ] Decide oneDNN vs. Vulkan-compute as the primary path (doc notes neither is an
      "official OS API" — pick one as primary, keep the other as a documented option)
- [ ] IR-node → primitive/shader mapping for every op (this is the most build-it-yourself backend
      since there's no ready-made graph API to target)
- [ ] Tensor buffer management (oneDNN memory objects or Vulkan buffers)
- [ ] Graph compile step (oneDNN primitive cache, or Vulkan pipeline + shader module cache)
- [ ] Dispatch + sync/fence wiring
- [ ] Parity tests against CPU reference

#### 3d (superseded). Android — NNAPI now, LiteRT delegates later
**Decision (confirmed against the real docs, not memory):** NNAPI (`<android/NeuralNetworks.h>`)
is deprecated as of Android 15, but still present and functional — the NDK page's own warning is
"you can continue to use NNAPI... we expect the majority of devices in the future to use the CPU
backend" (i.e. deprecation risk is *silent accel loss on future devices*, not a build break).
`android_backend.{hpp,cpp}` targets NNAPI for v1 anyway, because NNAPI's
`ANeuralNetworksModel`/`Compilation`/`Execution` C API is an IR-walk-and-compile API matching this
repo's other backends' shape (`compileGraph(GraphIR)` → native graph object → `dispatch()`) almost
exactly. The actually-current replacement, **LiteRT's GPU/NPU delegates**
(`TfLiteGpuDelegateV2Create`, and a now-unified NPU delegate covering Qualcomm/MediaTek/Google
Tensor), does *not* fit that shape — delegates attach to a `TfLiteInterpreter` loaded from a TFLite
FlatBuffer model, so adopting them would mean writing a `GraphIR`→TFLite-flatbuffer exporter (the
inverse of the existing TFLite *importer*, see `src/tflite/`) and dispatching through an interpreter
instead of a hand-built native graph object — a real architecture change, not a drop-in swap.
**Superseded:** rather than wait on that revisit, decided to skip NNAPI/LiteRT entirely in favor of
the self-contained `campello_gpu` Vulkan backend above, which also happens to cover Linux for free.
- [ ] `Context` backend wiring, with capability detection (treat as best-effort per the doc —
      OEM coverage is inconsistent)
- [ ] IR-node → NNAPI op mapping where available
- [ ] XNNPACK fallback path for ops/devices NNAPI doesn't cover
- [ ] Tensor buffer bridging for both paths
- [ ] Parity tests against CPU reference (may need per-device tolerance adjustments)

</details>

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

### Op-set prep (`campello_nn`) ✅
LLaMA/GPT-style decoder blocks need two pieces the original transformer-block op set didn't have:
- [x] `GraphBuilder::rmsNorm(x, scale, eps)` — new `OpKind::RmsNorm` (`layerNorm` minus
      mean-centering/bias: `out = x * rsqrt(mean(x^2)+eps) * scale`). CPU + MPSGraph implemented
      and tested on real GPU hardware (`CpuOps.RmsNorm`/`MpsOps.RmsNorm`). **DirectML
      intentionally not implemented yet** — falls through to that backend's existing
      `"unhandled OpKind"` default-throw rather than guessing; `DML_MEAN_VARIANCE_NORMALIZATION1_
      OPERATOR_DESC` (already used for `LayerNorm` there) is the likely operator, but its
      `NormalizeVariance` flag needs checking against real `DirectML.h`/docs on Windows before
      trusting it — follow-up for that platform.
- [x] `GraphBuilder::rotaryEmbedding(x, cos, sin)` — **no new IR op**. Standard "rotate-half" RoPE
      decomposes entirely into existing `slice`/`concat`/`mul`/`add`, so this is pure composition
      (same pattern as `quantizedMatmul`) — works on CPU/MPSGraph/DirectML automatically, verified
      on CPU + real MPSGraph/GPU hardware (`CpuOps.RotaryEmbedding`/`MpsOps.RotaryEmbedding`).
      Restricted to Float32/Float16; throws if `x`'s last dimension is odd.
- [x] Validation tests: `RmsNormScaleSizeMismatchThrows`, `RotaryEmbeddingOddLastDimThrows`.

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
- [x] Performance benchmarks: per-op/per-graph latency per backend — initial harness in
      `benchmarks/benchmark_backends.cpp` (transformer block on `Cpu`/`GpuGeneric`/`Gpu`).
      Prefill/decode token/sec and LLM-specific benchmarks deferred to `campello_llm` (Phase 5).
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

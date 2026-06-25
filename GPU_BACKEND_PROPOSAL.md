# Generic GPU Backend — Work Status

> Scope: `DeviceType::GpuGeneric` backed by `campello_gpu` (`src/gpu/`). Goal: finish the missing op coverage so real imported models (ONNX/TFLite) can run end-to-end, then broaden shape/dtype support.

## 1. Current State (as of 2026-06-25)

Implemented and verified on Metal (**30/30 `GpuGenericOps` tests passing**, full suite **103/103**):

- `Input`, `Constant`, `Reshape` (zero-cost alias)
- Elementwise/activation: `Relu`, `Add` (exact shape), `Mul` (exact shape), `Sigmoid`, `Gelu`
- Normalization: `LayerNorm`, `RmsNorm`, `BatchNorm`, `InstanceNorm`
- Linear: `MatMul` (rank-2 unbatched), `Gemm`, `QuantizedMatmul`
- Shape: `Transpose`, `Slice`, `Concat`, `Gather`
- Reduction/softmax: `Softmax`
- Vision: `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Resize`
- Quantization: `QuantizeLinear`, `DequantizeLinear`

Still restricted / throws today:

- `Add`/`Mul` broadcasting
- Batched `MatMul`
- Float16 support (backend currently rejects anything except `Float32`)

## 2. Op Breakdown

| Op | Status | Notes |
|----|--------|-------|
| `Conv2d` | ✅ | Wired; parity test passes on Metal. |
| `MaxPool2d` | ✅ | Wired; parity test passes on Metal. |
| `AvgPool2d` | ✅ | Wired; parity test passes on Metal. |
| `Resize` | ✅ | Wired; parity test passes on Metal. |
| `BatchNorm` | ✅ | Wired; parity test passes on Metal. |
| `InstanceNorm` | ✅ | Wired; parity test passes on Metal. |
| `QuantizeLinear` | ✅ | New byte-addressed GLSL/Metal shaders; HLSL written but unverified. |
| `DequantizeLinear` | ✅ | New byte-addressed GLSL/Metal shaders; HLSL written but unverified. |
| `Add` broadcast | ❌ | Currently throws if operand shapes != output shape. |
| `Mul` broadcast | ❌ | Same as `Add` broadcast. |
| `MatMul` batched | ❌ | Currently rank-2 only. GraphBuilder already allows batched shapes. |
| Float16 | ❌ | All shaders are float-only; backend rejects non-Float32. |

## 3. Proposed Implementation Order

### Phase A — Wire the existing shaders ✅

Done. Generated embedded byte arrays and wired `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Resize`, `BatchNorm`, `InstanceNorm` into `gpu_backend.cpp`; added parity tests; all pass on Metal.

### Phase B — Add quantization ops ✅

Done. Wrote new `quantize_linear.{comp,metal,hlsl}` and `dequantize_linear.{comp,metal,hlsl}` using byte-addressed storage (GLSL via `GL_EXT_shader_8bit_storage`, Metal via `device char*`, HLSL via `ByteAddressBuffer`/`RWByteAddressBuffer` with `InterlockedOr`). Wired them into `gpu_backend.cpp`; added `QuantizeLinear`, `DequantizeLinear`, round-trip, and `QuantizedMatmul` tests — all pass on Metal. HLSL path remains unverified.

### Phase C — Broaden shape support

These are not new `OpKind`s but unlock many more real graphs.

1. **`Add`/`Mul` broadcasting**
   - Option 1 (simplest): emit a runtime reshape/expand via existing ops? Not always possible for arbitrary broadcast.
   - Option 2 (recommended): write `broadcast_add` / `broadcast_mul` shaders that take two input buffers, an output buffer, and a params struct describing each operand's rank/strides. The CPU backend already has `broadcastInputIndex()` logic to copy.
   - Start with the common cases: `[N,C,H,W] + [1,C,1,1]` (channel bias), `[M,N] + [N]` (row-vector), `[M,N] + scalar`.
   - Keep exact-shape fast path as-is; only fall back to broadcast kernel when shapes differ.

2. **Batched `MatMul`**
   - Extend existing `matmul.comp` to handle leading batch dims: treat `A` as `[batch, M, K]`, `B` as `[batch, K, N]` or broadcast `B` across batch.
   - Add `ParamsBatchedMatMul` with `batch, M, K, N` and dispatch `(N, M, batch)`.

### Phase D — Float16 support (cross-cutting)

All current shaders hardcode `float`. To support `Float16` natively:

1. Add `float16_t` / `half` variants of every shader, or make shaders templated/preparsed for both dtypes.
2. In `gpu_backend.cpp`, branch `compileGraph()` on `node.dataType` to pick the right `OpKind`-and-dtype pipeline (or keep separate `OpResources` per dtype).
3. `elementByteSize()` already handles `Float16`.
4. Tests mirror `tests/platform/test_mps_float16_ops.cpp`.

**Recommendation:** defer Phase D until Phase A–C are done and the backend runs at least one real imported model (e.g. `conv_add_relu.onnx`) end-to-end.

## 4. Deliverable Milestones

| Milestone | Status | Definition of done |
|-----------|--------|--------------------|
| M1: Vision ops wired | ✅ | `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Resize`, `BatchNorm`, `InstanceNorm` pass parity tests on Metal. |
| M2: Quantization wired | ✅ | `QuantizeLinear`, `DequantizeLinear` pass; `QuantizedMatmul` graph works end-to-end on Metal. |
| M3: Real imported model | ❌ | An ONNX/TFLite fixture (e.g. `conv_add_relu.onnx`) runs on `GpuGeneric`, not just hand-built graphs. |
| M4: Broadcast/batched matmul | ❌ | Common `Add`/`Mul` broadcast cases and batched `MatMul` pass. |
| M5: Float16 | ❌ | `GpuGeneric` mirrors the MPSGraph Float16 test coverage. |

## 5. Files to Touch

- `src/gpu/shaders/*_spv.hpp` / `*_metallib.hpp` — generate new embedded byte arrays.
- `src/gpu/gpu_backend.cpp` — shader byte routing, `numInputsFor`, `compileGraph`, `dispatch`.
- `src/gpu/gpu_backend.hpp` — no public API change expected.
- `tests/platform/test_gpu_generic_ops.cpp` — new test cases.
- `CMakeLists.txt` / `tests/CMakeLists.txt` — likely no change unless adding a shader build step.
- `TODO.md` — update Phase 3c/3d checklist once work is in progress/done.

## 6. Risks / Open Questions

- **DirectX12 path is unverified.** All `.hlsl` sources are written but have never been compiled. The Metal path can be verified locally; Vulkan via `glslangValidator`; DirectX12 requires a Windows toolchain or at least `dxc`.
- **Shader binary embedding.** `scripts/compile_gpu_shaders.py` now automates SPIR-V/Metal `.metallib` generation, so embedded headers are repeatable. DirectX12 bytecode generation is still manual/out of scope until the HLSL path is verified.
- **Performance.** Current dispatch model is one workgroup per output element with a single active thread. This is correct but leaves GPU utilization on the table. Do not optimize until M1–M3 are done and a benchmark exists.
- **Windows `_WIN32` path.** `gpu_backend.cpp` currently throws for every shader on Windows because no precompiled DirectX bytecode is shipped. Either compile `.hlsl` with `dxcompiler` at build time or commit `.cso` files.

## 7. Suggested Next Steps

1. **Run a real imported model on `GpuGeneric`** (e.g. `conv_add_relu.onnx` / `.tflite`) — this is the first end-to-end validation that the op coverage is sufficient for model import.
2. **Broadcast `Add`/`Mul`** — needed for virtually every imported Conv-bias graph.
3. **Batched `MatMul`** — needed for transformer-style graphs with batch > 1.
4. **Float16 support** — cross-cutting; defer until M3/M4 are done.

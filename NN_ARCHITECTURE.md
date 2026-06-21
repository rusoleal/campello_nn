# Campello — Neural Network Execution Architecture Vision

> **TL;DR:** `campello_nn` is a WebNN-shaped compute-graph abstraction over each platform's official ML accelerator API (DirectML, MPSGraph, NNAPI/XNNPACK, oneDNN/Vulkan). Because graph-format model files (ONNX, TFLite) carry their own topology and need no architecture-specific knowledge to load, `campello_nn` also owns importing those directly into its own `Graph`, plus serializing/caching the result. `campello_llm` (and a future `campello_vision`) sit on top and own what no graph format models: tokenization, weight-only-format loading (safetensors/gguf) for known architectures, KV-cache, and sampling.

---

## 1. Why WebNN as the Reference Model

`campello_gpu` is "inspired by WebGPU" because WebGPU's API shape maps cleanly onto native graphics APIs (DX12, Metal, Vulkan). WebNN plays the same role for ML accelerators: it is a low-level **compute-graph** API designed to map onto DirectML, Core ML/MPS, and NNAPI-class backends.

The key difference from the GPU case: WebNN only gives you tensor ops (matmul, conv, softmax, ...). It says nothing about tokenizers, chat templates, KV-caches, or sampling. That extra layer is architecture-specific and has no browser-API equivalent, so it must be engine-native.

This produces the same two-layer split as rendering — except `campello_nn` also absorbs graph-format model import (see §5), since that needs no architecture-specific knowledge either:

```
campello_nn       — low-level neural compute graph (WebNN-shaped) + ONNX/TFLite import + Graph caching
     ↑                  ↑
campello_llm    campello_vision   — each: weight-only-format loading for known architectures,
(tokenizer, KV-cache,              plus whatever runtime state that domain needs
 sampling, generation loop)        (campello_vision has no generation loop/KV-cache to own)
```

`campello_renderer` doesn't know about glTF at runtime; it only consumes `RenderScene`. Likewise, `campello_nn` doesn't know about transformers, CNNs, tokens, or attention; it only consumes a graph of `Operand`s — whether that graph was built by hand via `GraphBuilder`, wired up by `campello_llm`/`campello_vision` from raw weights, or imported directly from an ONNX/TFLite file.

---

## 2. Backend Mapping

| Platform | Backend | Note |
|----------|---------|------|
| Windows | DirectML | direct match for WebNN's graph model |
| macOS / iOS | **MPSGraph** | MPSGraph is Apple's op-graph build → compile → run API — the structural analog of WebNN. Core ML is model-level, not graph-level, and is a worse fit; use it only as an optional ANE-delegate path reached through MPSGraph. |
| Android | NNAPI successor / vendor delegate, fallback to XNNPACK | OEM coverage is inconsistent; treat as best-effort |
| Linux | oneDNN or Vulkan compute shaders | no official OS API exists — same gap WebNN itself has on Linux |
| Web | passthrough to native browser WebNN | the only platform where this is "free" |

Unlike `campello_gpu`'s graphics backends, this table will not have full parity across all six supported platforms for a while. Linux has no vendor-blessed API at all, and native WebNN implementations are far less mature than Dawn/wgpu-native — early backend work may mean implementing directly against DirectML/MPSGraph rather than tracking a ready-made native WebNN library.

---

## 3. `campello_nn` — Graph Builder

The op set is deliberately the minimum needed to express a transformer block (attention via matmul/softmax, layerNorm, gelu, gather for embeddings) — the same restraint `campello_gpu` applies by not exposing every Metal/DX12 feature, only what the renderer needs.

**Scope update:** the project now also targets vision and multimodal models, not just text-only transformer LLMs. This adds `conv2d`, `maxPool2d`, `avgPool2d`, `resize`, `batchNorm`, and `instanceNorm` to the op set below — the minimum needed to express a CNN/vision-encoder stage plus the resizing/normalization typically used there. `batchNorm` and `instanceNorm` are kept distinct (rather than one generic "normalize over axes" primitive) to match ONNX/PyTorch's `BatchNorm2d`/`InstanceNorm2d` semantics exactly: `batchNorm` takes mean/variance as given inputs (inference-mode running stats), while `instanceNorm` computes them itself, per-`(N,C)`, over the spatial axes. Cross-modal fusion (e.g. vision embeddings feeding into a transformer decoder) needs no new ops: it's the same `matmul`/`softmax` attention machinery already required for text.

**Dtype update:** every op now supports Float16 in addition to Float32 (the CPU backend decodes/encodes at the Float16 boundary internally; MPSGraph runs Float16 natively), and `gather` accepts Uint32 indices alongside Int32. Int8 is real quantization rather than a storage-only dtype: `quantizeLinear`/`dequantizeLinear` (ONNX-shaped, per-tensor scale/zero-point) plus a `quantizedMatmul` convenience that dequantizes a weight then matmuls — weight-only quantization, since neither backend has a native int8×int8 GEMM kernel.

**Model import:** `campello_nn` also imports graph-format model files directly — ONNX first, TFLite as a planned follow-up. Both formats encode full graph topology alongside their weights, not just raw tensors, so importing one is a 1:1 translation from the file's own node list into the equivalent `GraphBuilder` calls (ONNX `Conv` → `conv2d`, `Add` → `add`, ...) — no architecture-specific knowledge required, unlike `Model::load` in §4. This is also why it lives here rather than in `campello_llm`/`campello_vision`: those layers exist specifically for *weight-only* formats (safetensors, gguf, raw state dicts) that carry no topology and therefore need externally-supplied architecture knowledge to wire up.

- **ONNX** (protobuf-encoded) — implemented and validated end-to-end against a real, standard pretrained model (YuNet face detection) on real photos, not just a synthetic fixture. As expected, the first real model surfaced gaps beyond the doc's original op table — `Relu`/`Sigmoid` (added), and import-side handling for `Conv` with a fused bias (the universal case in practice, not an edge case), `Transpose`, `Reshape` (resolving ONNX's `0`/`-1` shape sentinels), `MaxPool`/`AveragePool`, and `Resize` (including its `nearest_mode="floor"`, which needed a small `ResizeDescriptor` extension). Op coverage is still limited to whatever maps onto the table below; further gaps are expected from future real models.
- **TFLite** (FlatBuffers-encoded) — same category, planned follow-up. Comparable effort to ONNX, not free once ONNX import exists: different binary format, different op-naming, NHWC-default layout (vs. this doc's NCHW convention), and quantization fused into op attributes rather than separate nodes.

```cpp
namespace campello_nn {

enum class DeviceType { Cpu, Gpu, Npu, Default };
enum class DataType   { Float32, Float16, Int32, Int8, Uint32 };

struct ContextDescriptor {
    DeviceType deviceType = DeviceType::Default;
};

class Context {
public:
    static std::shared_ptr<Context> create(const ContextDescriptor& desc);
    std::shared_ptr<Tensor> createTensor(const TensorDescriptor& desc);
    Future<void> dispatch(const Graph& graph,
        const std::unordered_map<std::string, std::shared_ptr<Tensor>>& inputs,
        const std::unordered_map<std::string, std::shared_ptr<Tensor>>& outputs);
};

struct TensorDescriptor {
    DataType dataType;
    std::vector<int64_t> shape;
    bool readable = false; // CPU-mappable
    bool writable = false;
};

class Tensor {
public:
    const std::vector<int64_t>& shape() const;
    DataType dataType() const;
    void write(const void* data, size_t size);
    void read(void* data, size_t size) const;
};

class Operand; // opaque graph-node handle

class GraphBuilder {
public:
    explicit GraphBuilder(std::shared_ptr<Context> context);

    Operand input(const std::string& name, const TensorDescriptor& desc);
    Operand constant(const TensorDescriptor& desc, const void* data, size_t size);

    // elementwise / activation
    Operand add(Operand a, Operand b);
    Operand mul(Operand a, Operand b);
    Operand gelu(Operand x);
    Operand softmax(Operand x, int32_t axis);
    Operand layerNorm(Operand x, Operand scale, Operand bias, float eps);

    // linear algebra
    Operand matmul(Operand a, Operand b);
    Operand gemm(Operand a, Operand b, Operand c, float alpha, float beta);

    // shape manipulation
    Operand reshape(Operand x, const std::vector<int64_t>& shape);
    Operand transpose(Operand x, const std::vector<int32_t>& perm);
    Operand concat(const std::vector<Operand>& xs, int32_t axis);
    Operand slice(Operand x, const std::vector<int64_t>& starts, const std::vector<int64_t>& sizes);
    Operand gather(Operand data, Operand indices, int32_t axis); // embedding lookup

    // vision
    Operand conv2d(Operand input, Operand weights, const Conv2dDescriptor& desc);
    Operand maxPool2d(Operand x, const Pool2dDescriptor& desc);
    Operand avgPool2d(Operand x, const Pool2dDescriptor& desc);
    Operand resize(Operand x, const ResizeDescriptor& desc);
    Operand batchNorm(Operand x, Operand mean, Operand variance, Operand scale, Operand bias, float eps);
    Operand instanceNorm(Operand x, Operand scale, Operand bias, float eps);

    // quantization
    Operand quantizeLinear(Operand x, float scale, int32_t zeroPoint);
    Operand dequantizeLinear(Operand x, float scale, int32_t zeroPoint);
    Operand quantizedMatmul(Operand activation, Operand weightInt8, float weightScale, int32_t weightZeroPoint);

    std::shared_ptr<Graph> build(const std::unordered_map<std::string, Operand>& outputs);
};

class Graph {
    // opaque compiled/optimized executable, analogous to a compiled GPU pipeline
};

} // namespace campello_nn
```

`Context` is the analog of `campello_gpu::Device`: it selects and owns a backend. `Graph` is the analog of a compiled pipeline: build once, dispatch many times.

---

## 4. `campello_llm` — Model Execution on Top

`campello_llm` is specifically for **weight-only-format** LLM/transformer weights (safetensors, gguf) — files with no embedded graph topology, where the architecture (LLaMA-style, GPT-style, ...) must be supplied by code that already knows it. Models that ship as ONNX/TFLite don't need `campello_llm` at all; `campello_nn`'s own importer (§3, §5) handles those directly. A future `campello_vision` sibling would play the same role for weight-only-format vision models (e.g. a raw PyTorch state dict with no ONNX export) — same split, same reasoning, no shared code with `campello_llm` beyond what both pull from `campello_nn`.

```cpp
namespace campello_llm {

struct GenerationConfig {
    int32_t maxTokens = 256;
    float temperature = 0.8f;
    float topP = 0.95f;
    int32_t topK = 40;
};

class Model {
public:
    static std::shared_ptr<Model> load(
        std::shared_ptr<campello_nn::Context> context, const std::string& path);

    void generate(const std::string& prompt, const GenerationConfig& config,
                  std::function<void(const std::string& token)> onToken);
};

} // namespace campello_llm
```

`Model::load` is the part with no WebNN equivalent. It reads a weights file (safetensors/gguf) and, for a known architecture (LLaMA-style, GPT-style), wires up the full graph by calling `GraphBuilder` ops per layer with the weights bound as `constant()`s.

`generate()` owns the state WebNN graphs don't model:

- **Prefill** — one graph dispatch over the whole prompt.
- **KV-cache** — explicit `Tensor`s fed back in as inputs each decode step and read back out. WebNN graphs are stateless by default, so this bookkeeping lives in `campello_llm`, not `campello_nn`.
- **Sampling** — temperature / top-k / top-p applied on the CPU after reading the logits tensor back.
- **Streaming** — the decode loop invokes `onToken` per generated token.

---

## 5. Model Import and Graph Caching

`campello_nn` owns two graph-level jobs beyond `GraphBuilder`'s op-by-op API — both are pure graph translation/serialization that need no knowledge of what model they represent, which is why neither needs a separate project the way an earlier draft of this doc sketched (`campello_nn_convert`):

**Importing graph-format model files** (§3's "Model import"). ONNX and, later, TFLite files carry full topology, not just weight values — loading one means walking the file's own node list and calling the matching `GraphBuilder` op per node. No architecture-specific knowledge needed, so it belongs here rather than in `campello_llm`/`campello_vision`.

**Caching the compiled result.** Building the full graph from a weights file at every process startup is wasted work once a model is known to be stable — true whether that graph came from `campello_llm`/`campello_vision`'s per-layer `GraphBuilder` wiring, or from `campello_nn`'s own ONNX/TFLite import above. `campello_nn` can serialize its own compiled `Graph` (weights baked in) to disk and load that serialized form directly next time, skipping reconstruction entirely. This plays the same role shader precompilation plays for `campello_gpu` — pay the build/optimize cost once, not on every launch.

---

## Summary

| Concept | What It Is |
|---------|------------|
| **`campello_nn`** | A WebNN-shaped compute-graph library. Knows tensors and ops — not what a token or a transformer is. Also imports graph-format model files (ONNX, later TFLite) and caches compiled `Graph`s to disk, since both are pure graph-level jobs needing no architecture knowledge. |
| **MPSGraph, not Core ML** | The correct Apple-platform target — it is graph-level, matching WebNN's shape; Core ML is model-level and can't ingest ONNX/TFLite at runtime anyway (needs offline `coremltools` conversion). |
| **`campello_llm`** | Owns tokenization, weight-only-format loading (safetensors/gguf) for known LLM architectures, KV-cache, and sampling. Not needed for models that ship as ONNX/TFLite — `campello_nn` handles those directly. |
| **`campello_vision`** (future) | Same role as `campello_llm`, for weight-only-format vision models. No shared runtime state to own (no KV-cache/generation loop) — mostly just architecture-specific `GraphBuilder` wiring. |
| **Platform parity gap** | Linux and (to a lesser extent) Android lack a mature official graph API, unlike the Metal/DX12/Vulkan parity `campello_gpu` enjoys. |

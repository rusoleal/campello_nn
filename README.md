# campello_nn

A WebNN-shaped neural-network compute-graph library (C++20). Provides a low-level tensor/graph
abstraction intended to map onto each platform's official ML accelerator API — see
[NN_ARCHITECTURE.md](./NN_ARCHITECTURE.md) for the full design rationale.

## 🚀 Part of the Campello Engine

This project is a module within the **Campello** ecosystem.

👉 Main repository: https://github.com/rusoleal/campello

Campello is a modular, composable game engine built as a collection of independent libraries.
Each module is designed to work standalone, but integrates seamlessly into the engine runtime.
`campello_nn` plays the same role for ML accelerators that `campello_gpu` plays for graphics.

## Backend status

| Platform | Backend | Status |
|----------|---------|--------|
| All | CPU reference backend (Float32, Float16, Int8 quantization) | Implemented |
| macOS / iOS | MPSGraph | Implemented (`DeviceType::Gpu`/`Npu`/`Default`) |
| Windows | DirectML | Not started |
| Linux | oneDNN / Vulkan compute | Not started |
| Android | NNAPI successor / vendor delegate, fallback XNNPACK | Not started |
| Web | passthrough to native browser WebNN | Not started |

See [TODO.md](./TODO.md) for the full phased roadmap.

## Requirements

- CMake 3.22.1+
- C++20 compiler

## Build

```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Example

```cpp
#include <campello_nn/context.hpp>
#include <campello_nn/graph_builder.hpp>

using namespace systems::leal::campello_nn;

auto context = Context::create({DeviceType::Cpu});

GraphBuilder builder(context);
auto x   = builder.input("x", {DataType::Float32, {1, 4}});
auto out = builder.gelu(x);
auto graph = builder.build({{"out", out}});

auto input  = context->createTensor({DataType::Float32, {1, 4}, false, true});
auto output = context->createTensor({DataType::Float32, {1, 4}, true, false});
input->write(data, sizeof(data));

auto fence = context->dispatch(*graph, {{"x", input}}, {{"out", output}});
fence->wait();
output->read(result, sizeof(result));
```

## Implemented ops

All ops below are exposed via `GraphBuilder` (`inc/campello_nn/graph_builder.hpp`) and run on
every backend that's `Implemented` in the table above.

| Op | Description |
|----|--------------|
| `add` | Elementwise addition, NumPy/ONNX-style broadcasting |
| `mul` | Elementwise multiplication, NumPy/ONNX-style broadcasting |
| `gelu` | Gaussian Error Linear Unit activation |
| `relu` | Rectified linear unit activation |
| `sigmoid` | Sigmoid activation |
| `softmax` | Softmax over a given axis |
| `layerNorm` | Layer normalization over the last axis (mean/variance computed from `x`) |
| `rmsNorm` | LLaMA-style RMSNorm — `layerNorm` without mean-centering or bias/shift |
| `rotaryEmbedding` | Rotary position embedding (GPT-NeoX/LLaMA "rotate-half" convention) |
| `matmul` | Matrix multiplication |
| `gemm` | General matrix multiply: `alpha * (a @ b) + beta * c` |
| `reshape` | Reshape to an explicit target shape |
| `transpose` | Permute axes |
| `concat` | Concatenate tensors along an axis |
| `slice` | Extract a sub-tensor by start offsets and sizes |
| `gather` | Index lookup along an axis (embedding lookup) |
| `conv2d` | 2D convolution, NCHW input / OIHW weights |
| `maxPool2d` | 2D max pooling |
| `avgPool2d` | 2D average pooling |
| `resize` | Spatial resize (nearest/bilinear) |
| `batchNorm` | Batch normalization, NCHW, using given (inference-mode) mean/variance |
| `instanceNorm` | Instance normalization, NCHW, mean/variance computed per-(N,C) over spatial axes |
| `quantizeLinear` | Per-tensor scale/zero-point quantization, Float32 → Int8 |
| `dequantizeLinear` | Per-tensor scale/zero-point dequantization, Int8 → Float32 |
| `quantizedMatmul` | Weight-only quantized matmul: dequantizes an Int8 weight then matmuls |

## Model import

`campello_nn` can also import ONNX models directly — no architecture-specific code needed, since
ONNX files carry their own graph topology (see [NN_ARCHITECTURE.md](./NN_ARCHITECTURE.md) §3/§5):

```cpp
#include <campello_nn/onnx_importer.hpp>

auto context = Context::create({DeviceType::Cpu});
auto result = importOnnxFromFile(context, "model.onnx"); // or importOnnxFromMemory(...)

auto input = context->createTensor(result.inputs.at("input_name"));
input->write(data, size);

std::unordered_map<std::string, std::shared_ptr<Tensor>> outputs;
for (auto& [name, desc] : result.outputs)
    outputs[name] = context->createTensor(desc);

auto fence = context->dispatch(*result.graph, {{"input_name", input}}, outputs);
fence->wait();
```

Op coverage is scoped to what's been tested against real models so far (see
[TODO.md](./TODO.md) Phase 4a) — validated end-to-end against a real, standard pretrained face
detector (YuNet) running on real photos in `tests/universal/test_yunet_face_detection.cpp`
(build with `-DBUILD_MODEL_TESTS=ON`).

## Graph introspection

Both importers also populate `result.info` (a `GraphInfo`) alongside the compiled `Graph` — the
graph's topology (op list, per-node shape/dtype/attributes, edges), for callers that want to
inspect or visualize what was imported rather than execute it (e.g. a model viewer):

```cpp
#include <campello_nn/onnx_importer.hpp>

auto result = importOnnxFromFile(context, "model.onnx");
for (const NodeInfo& node : result.info.nodes) {
    std::cout << toString(node.kind); // "Conv2d", "Add", "Relu", ...
    if (!node.name.empty()) std::cout << " (" << node.name << ")";
    std::cout << " -> shape [";
    for (int64_t d : node.shape) std::cout << d << " ";
    std::cout << "]\n";
}
```

`GraphInfo` is a display-safe mirror of the internal IR: `NodeInfo::constantByteSize` is the size
of a `Constant` node's data, not the data itself, so inspecting a graph never duplicates a real
model's (potentially hundreds-of-megabytes) weights in memory just to describe its shape.

## Graph caching

A built graph can be serialized to bytes and reloaded later, skipping `GraphBuilder`
reconstruction (op-by-op calls, or re-parsing a source model file):

```cpp
#include <campello_nn/graph_cache.hpp>

// At build/packaging time:
GraphBuilder builder(context);
auto x = builder.input("x", {DataType::Float32, {1, 4}});
auto bytes = builder.serialize({{"out", builder.gelu(x)}});
saveGraphToFile(bytes, "model.cnncache");

// Later, possibly in a different process:
auto result = loadGraphFromFile(context, "model.cnncache"); // or loadGraphFromMemory(...)
auto input = context->createTensor(result.inputs.at("x"));
// ...same dispatch pattern as model import above
```

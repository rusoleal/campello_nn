#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <campello_nn/context.hpp>
#include <campello_nn/operand.hpp>
#include <campello_nn/graph.hpp>
#include <campello_nn/descriptors/tensor_descriptor.hpp>
#include <campello_nn/descriptors/conv2d_descriptor.hpp>
#include <campello_nn/descriptors/pool2d_descriptor.hpp>
#include <campello_nn/descriptors/resize_descriptor.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Builds a compute graph of `Operand`s, then compiles it into a `Graph`.
     *
     * The core op set is the minimum needed to express a transformer block: attention
     * via `matmul`/`softmax`, `layerNorm`, `gelu`, and `gather` for embeddings.
     * `conv2d`/`maxPool2d`/`avgPool2d` extend this to vision/multimodal model support.
     */
    class GraphBuilder
    {
        void *native;

    public:
        explicit GraphBuilder(std::shared_ptr<Context> context);
        ~GraphBuilder();

        Operand input(const std::string &name, const TensorDescriptor &desc);
        Operand constant(const TensorDescriptor &desc, const void *data, size_t size);

        // elementwise / activation
        Operand add(Operand a, Operand b);
        Operand mul(Operand a, Operand b);
        Operand gelu(Operand x);
        Operand relu(Operand x);
        Operand sigmoid(Operand x);
        Operand softmax(Operand x, int32_t axis);
        Operand layerNorm(Operand x, Operand scale, Operand bias, float eps);

        // linear algebra
        Operand matmul(Operand a, Operand b);
        Operand gemm(Operand a, Operand b, Operand c, float alpha, float beta);

        // shape manipulation
        Operand reshape(Operand x, const std::vector<int64_t> &shape);
        Operand transpose(Operand x, const std::vector<int32_t> &perm);
        Operand concat(const std::vector<Operand> &xs, int32_t axis);
        Operand slice(Operand x, const std::vector<int64_t> &starts, const std::vector<int64_t> &sizes);
        Operand gather(Operand data, Operand indices, int32_t axis); // embedding lookup

        // vision
        Operand conv2d(Operand input, Operand weights, const Conv2dDescriptor &desc);
        Operand maxPool2d(Operand x, const Pool2dDescriptor &desc);
        Operand avgPool2d(Operand x, const Pool2dDescriptor &desc);
        Operand resize(Operand x, const ResizeDescriptor &desc);

        // NCHW; mean/variance are given (inference-mode running stats), not computed
        // from `x` — matching ONNX/PyTorch BatchNorm2d at inference time.
        Operand batchNorm(Operand x, Operand mean, Operand variance, Operand scale, Operand bias, float eps);
        // NCHW; mean/variance are computed per-(N,C) over the spatial (H,W) axes, like
        // layerNorm but over spatial axes with channel-broadcast scale/bias.
        Operand instanceNorm(Operand x, Operand scale, Operand bias, float eps);

        // quantization (per-tensor scale/zero-point, matching ONNX QuantizeLinear/
        // DequantizeLinear and MPSGraph's quantizeTensor:/dequantizeTensor: scalar overloads)
        Operand quantizeLinear(Operand x, float scale, int32_t zeroPoint);   // float -> Int8
        Operand dequantizeLinear(Operand x, float scale, int32_t zeroPoint); // Int8 -> Float32
        // Weight-only quantization: dequantizes weightInt8 then matmuls with activation.
        // Neither MPSGraph nor the CPU backend has a dedicated int8xint8 GEMM kernel, so
        // this is the realistic deployment pattern (int8 storage, float32 compute).
        Operand quantizedMatmul(Operand activation, Operand weightInt8, float weightScale, int32_t weightZeroPoint);

        std::shared_ptr<Graph> build(const std::unordered_map<std::string, Operand> &outputs);

        // Serializes the same backend-agnostic IR `build()` would otherwise compile,
        // without compiling it — for caching to disk (see graph_cache.hpp). Bytes are
        // versioned; reload with `deserialize()` or graph_cache.hpp's `loadGraphFrom*`.
        std::vector<uint8_t> serialize(const std::unordered_map<std::string, Operand> &outputs);

        // Compiles a graph directly from bytes produced by `serialize()`, skipping
        // GraphBuilder reconstruction (op-by-op calls or re-parsing a source model
        // file). Throws std::runtime_error on corrupt/unsupported-version data.
        static std::shared_ptr<Graph> deserialize(std::shared_ptr<Context> context, const uint8_t *data, size_t size);
    };

} // namespace systems::leal::campello_nn

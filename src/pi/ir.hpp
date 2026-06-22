#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <campello_nn/constants/data_type.hpp>
#include <campello_nn/descriptors/conv2d_descriptor.hpp>
#include <campello_nn/descriptors/pool2d_descriptor.hpp>
#include <campello_nn/descriptors/resize_descriptor.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Backend-agnostic graph IR. Built by `GraphBuilder`, consumed by `Backend::compileGraph()`.
     *
     * Internal type — never exposed through a public header. Each `Node` already
     * carries its inferred output shape/dtype, so backends do not need to re-derive
     * them at compile time.
     */
    enum class OpKind
    {
        Input,
        Constant,
        Add,
        Mul,
        Gelu,
        Softmax,
        LayerNorm,
        RmsNorm,
        MatMul,
        Gemm,
        Reshape,
        Transpose,
        Concat,
        Slice,
        Gather,
        Conv2d,
        MaxPool2d,
        AvgPool2d,
        Resize,
        BatchNorm,
        InstanceNorm,
        QuantizeLinear,
        DequantizeLinear,
        Relu,
        Sigmoid
    };

    struct Node
    {
        OpKind kind;
        std::vector<size_t> inputs;
        DataType dataType;
        std::vector<int64_t> shape;

        std::string name;                 // Input: binding name
        std::vector<uint8_t> constantBytes; // Constant: raw data

        std::vector<int64_t> intAttr0; // Reshape: target shape; Transpose: perm; Slice: starts
        std::vector<int64_t> intAttr1; // Slice: sizes
        int32_t axis = 0;              // Softmax/Concat/Gather: axis
        float floatAttr0 = 0.f;        // LayerNorm/RmsNorm: eps; Gemm: alpha
        float floatAttr1 = 0.f;        // Gemm: beta

        Conv2dDescriptor convParams;   // Conv2d
        Pool2dDescriptor poolParams;   // MaxPool2d/AvgPool2d
        ResizeDescriptor resizeParams; // Resize
    };

    struct GraphIR
    {
        std::vector<Node> nodes;
        std::vector<std::pair<std::string, size_t>> outputs; // output name -> node index
    };

} // namespace systems::leal::campello_nn

#pragma once

#include <cstddef>
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
     * @brief Public, display-safe mirror of the internal IR's op kinds (`src/pi/ir.hpp`).
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

    /** @brief Human-readable name for `kind`, e.g. `"Conv2d"`. Stable across releases. */
    const char *toString(OpKind kind);

    /**
     * @brief Describes a single node in an imported/built graph, for inspection and
     * visualization (editors, debuggers, model viewers) — not for execution.
     *
     * Mirrors the internal IR's `Node` (`src/pi/ir.hpp`) field-for-field, except
     * `constantByteSize` replaces the actual constant bytes: callers that want to
     * visualize a graph need to know a constant's size, not duplicate potentially
     * hundreds of megabytes of weight data into a second copy.
     */
    struct NodeInfo
    {
        OpKind kind;
        std::vector<size_t> inputs; // indices into the owning GraphInfo::nodes
        DataType dataType;
        std::vector<int64_t> shape;

        std::string name; // Input: binding name; empty for every other kind
        size_t constantByteSize = 0; // Constant: byte size of the (omitted) constant data

        std::vector<int64_t> intAttr0; // Reshape: target shape; Transpose: perm; Slice: starts
        std::vector<int64_t> intAttr1; // Slice: sizes
        int32_t axis = 0;              // Softmax/Concat/Gather: axis
        float floatAttr0 = 0.f;        // LayerNorm/RmsNorm: eps; Gemm: alpha
        float floatAttr1 = 0.f;        // Gemm: beta

        Conv2dDescriptor convParams;   // Conv2d
        Pool2dDescriptor poolParams;   // MaxPool2d/AvgPool2d
        ResizeDescriptor resizeParams; // Resize
    };

    /**
     * @brief Describes a full graph's topology — every node plus its named outputs.
     *
     * Returned alongside an imported `Graph` (see `OnnxImportResult`/`TfliteImportResult`)
     * for callers that need to inspect or visualize what was imported, without any
     * access to the internal IR.
     */
    struct GraphInfo
    {
        std::vector<NodeInfo> nodes;
        std::vector<std::pair<std::string, size_t>> outputs; // output name -> node index
    };

} // namespace systems::leal::campello_nn

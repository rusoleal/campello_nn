#include "graph_info.hpp"

using namespace systems::leal::campello_nn;

const char *systems::leal::campello_nn::toString(OpKind kind)
{
    switch (kind)
    {
    case OpKind::Input:
        return "Input";
    case OpKind::Constant:
        return "Constant";
    case OpKind::Add:
        return "Add";
    case OpKind::Mul:
        return "Mul";
    case OpKind::Gelu:
        return "Gelu";
    case OpKind::Softmax:
        return "Softmax";
    case OpKind::LayerNorm:
        return "LayerNorm";
    case OpKind::RmsNorm:
        return "RmsNorm";
    case OpKind::MatMul:
        return "MatMul";
    case OpKind::Gemm:
        return "Gemm";
    case OpKind::Reshape:
        return "Reshape";
    case OpKind::Transpose:
        return "Transpose";
    case OpKind::Concat:
        return "Concat";
    case OpKind::Slice:
        return "Slice";
    case OpKind::Gather:
        return "Gather";
    case OpKind::Conv2d:
        return "Conv2d";
    case OpKind::MaxPool2d:
        return "MaxPool2d";
    case OpKind::AvgPool2d:
        return "AvgPool2d";
    case OpKind::Resize:
        return "Resize";
    case OpKind::BatchNorm:
        return "BatchNorm";
    case OpKind::InstanceNorm:
        return "InstanceNorm";
    case OpKind::QuantizeLinear:
        return "QuantizeLinear";
    case OpKind::DequantizeLinear:
        return "DequantizeLinear";
    case OpKind::Relu:
        return "Relu";
    case OpKind::Sigmoid:
        return "Sigmoid";
    }
    return "Unknown";
}

namespace
{
    NodeInfo describeNode(const Node &node)
    {
        NodeInfo info;
        info.kind = node.kind;
        info.inputs = node.inputs;
        info.dataType = node.dataType;
        info.shape = node.shape;
        info.name = node.name;
        info.constantByteSize = node.constantBytes.size();
        info.intAttr0 = node.intAttr0;
        info.intAttr1 = node.intAttr1;
        info.axis = node.axis;
        info.floatAttr0 = node.floatAttr0;
        info.floatAttr1 = node.floatAttr1;
        info.convParams = node.convParams;
        info.poolParams = node.poolParams;
        info.resizeParams = node.resizeParams;
        return info;
    }
} // namespace

GraphInfo systems::leal::campello_nn::describeGraphIR(const GraphIR &ir)
{
    GraphInfo info;
    info.nodes.reserve(ir.nodes.size());
    for (const Node &node : ir.nodes)
        info.nodes.push_back(describeNode(node));
    info.outputs = ir.outputs;
    return info;
}

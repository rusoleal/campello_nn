#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <campello_nn/graph_builder.hpp>
#include <campello_nn/float16.hpp>
#include "graph_builder_data.hpp"
#include "context_data.hpp"
#include "resource_data.hpp"
#include "ir_serialization.hpp"

using namespace systems::leal::campello_nn;

namespace
{
    int64_t numElements(const std::vector<int64_t> &shape)
    {
        return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
    }

    void requireSameBuilder(void *builderA, void *builderB)
    {
        if (builderA != builderB)
            throw std::runtime_error("campello_nn: Operand belongs to a different GraphBuilder");
    }

    const Node &nodeOf(const GraphIR &ir, size_t nodeId)
    {
        return ir.nodes[nodeId];
    }

    // NumPy/ONNX-style broadcasting: shapes are aligned from the right (trailing
    // dims); a missing leading dim or a dim of size 1 broadcasts against the
    // other operand's dim. Throws if neither side can broadcast at some position.
    std::vector<int64_t> computeBroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b,
                                                const char *opName)
    {
        size_t rank = std::max(a.size(), b.size());
        std::vector<int64_t> result(rank);
        for (size_t i = 0; i < rank; i++)
        {
            int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
            int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
            if (da == db)
                result[i] = da;
            else if (da == 1)
                result[i] = db;
            else if (db == 1)
                result[i] = da;
            else
                throw std::runtime_error(std::string("campello_nn: ") + opName + "() operand shapes are not broadcast-compatible");
        }
        return result;
    }
}

GraphBuilder::GraphBuilder(std::shared_ptr<Context> context)
{
    native = new GraphBuilderData{std::move(context), GraphIR{}};
}

GraphBuilder::~GraphBuilder()
{
    delete (GraphBuilderData *)native;
}

Operand GraphBuilder::input(const std::string &name, const TensorDescriptor &desc)
{
    auto data = (GraphBuilderData *)native;
    Node node;
    node.kind = OpKind::Input;
    node.dataType = desc.dataType;
    node.shape = desc.shape;
    node.name = name;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::constant(const TensorDescriptor &desc, const void *bytes, size_t size)
{
    auto data = (GraphBuilderData *)native;
    Node node;
    node.kind = OpKind::Constant;
    node.dataType = desc.dataType;
    node.shape = desc.shape;
    node.constantBytes.assign((const uint8_t *)bytes, (const uint8_t *)bytes + size);
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::add(Operand a, Operand b)
{
    requireSameBuilder(a.builder, native);
    requireSameBuilder(b.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &na = nodeOf(data->ir, a.nodeId);
    const Node &nb = nodeOf(data->ir, b.nodeId);
    Node node;
    node.kind = OpKind::Add;
    node.inputs = {a.nodeId, b.nodeId};
    node.dataType = na.dataType;
    node.shape = computeBroadcastShape(na.shape, nb.shape, "add");
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::mul(Operand a, Operand b)
{
    requireSameBuilder(a.builder, native);
    requireSameBuilder(b.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &na = nodeOf(data->ir, a.nodeId);
    const Node &nb = nodeOf(data->ir, b.nodeId);
    Node node;
    node.kind = OpKind::Mul;
    node.inputs = {a.nodeId, b.nodeId};
    node.dataType = na.dataType;
    node.shape = computeBroadcastShape(na.shape, nb.shape, "mul");
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::gelu(Operand x)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    Node node;
    node.kind = OpKind::Gelu;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::relu(Operand x)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    Node node;
    node.kind = OpKind::Relu;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::sigmoid(Operand x)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    Node node;
    node.kind = OpKind::Sigmoid;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::softmax(Operand x, int32_t axis)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    int32_t rank = (int32_t)nx.shape.size();
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        throw std::runtime_error("campello_nn: softmax() axis out of range");
    Node node;
    node.kind = OpKind::Softmax;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    node.axis = axis;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::layerNorm(Operand x, Operand scale, Operand bias, float eps)
{
    requireSameBuilder(x.builder, native);
    requireSameBuilder(scale.builder, native);
    requireSameBuilder(bias.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    const Node &nscale = nodeOf(data->ir, scale.nodeId);
    const Node &nbias = nodeOf(data->ir, bias.nodeId);
    if (nx.shape.empty())
        throw std::runtime_error("campello_nn: layerNorm() input must have rank >= 1");
    int64_t lastDim = nx.shape.back();
    if (numElements(nscale.shape) != lastDim || numElements(nbias.shape) != lastDim)
        throw std::runtime_error("campello_nn: layerNorm() scale/bias must match the input's last dimension");
    Node node;
    node.kind = OpKind::LayerNorm;
    node.inputs = {x.nodeId, scale.nodeId, bias.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    node.floatAttr0 = eps;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::rmsNorm(Operand x, Operand scale, float eps)
{
    requireSameBuilder(x.builder, native);
    requireSameBuilder(scale.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    const Node &nscale = nodeOf(data->ir, scale.nodeId);
    if (nx.shape.empty())
        throw std::runtime_error("campello_nn: rmsNorm() input must have rank >= 1");
    int64_t lastDim = nx.shape.back();
    if (numElements(nscale.shape) != lastDim)
        throw std::runtime_error("campello_nn: rmsNorm() scale must match the input's last dimension");
    Node node;
    node.kind = OpKind::RmsNorm;
    node.inputs = {x.nodeId, scale.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    node.floatAttr0 = eps;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::rotaryEmbedding(Operand x, Operand cosOp, Operand sinOp)
{
    requireSameBuilder(x.builder, native);
    requireSameBuilder(cosOp.builder, native);
    requireSameBuilder(sinOp.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    if (nx.shape.empty())
        throw std::runtime_error("campello_nn: rotaryEmbedding() input must have rank >= 1");
    int64_t lastDim = nx.shape.back();
    if (lastDim % 2 != 0)
        throw std::runtime_error("campello_nn: rotaryEmbedding() input's last dimension must be even");

    int32_t lastAxis = (int32_t)nx.shape.size() - 1;
    int64_t half = lastDim / 2;

    std::vector<int64_t> starts(nx.shape.size(), 0);
    std::vector<int64_t> sizes = nx.shape;
    sizes[lastAxis] = half;
    Operand firstHalf = slice(x, starts, sizes);

    starts[lastAxis] = half;
    Operand secondHalf = slice(x, starts, sizes);

    std::vector<uint8_t> negOneBytes;
    if (nx.dataType == DataType::Float32)
    {
        float v = -1.0f;
        negOneBytes.assign((const uint8_t *)&v, (const uint8_t *)&v + sizeof(v));
    }
    else if (nx.dataType == DataType::Float16)
    {
        uint16_t v = encodeFloat16(-1.0f);
        negOneBytes.assign((const uint8_t *)&v, (const uint8_t *)&v + sizeof(v));
    }
    else
    {
        throw std::runtime_error("campello_nn: rotaryEmbedding() only supports Float32/Float16 input");
    }
    Operand negOne = constant({nx.dataType, {1}, false, false}, negOneBytes.data(), negOneBytes.size());

    Operand negatedSecondHalf = mul(secondHalf, negOne);
    Operand rotatedHalf = concat({negatedSecondHalf, firstHalf}, lastAxis);

    return add(mul(x, cosOp), mul(rotatedHalf, sinOp));
}

Operand GraphBuilder::batchNorm(Operand x, Operand mean, Operand variance, Operand scale, Operand bias, float eps)
{
    requireSameBuilder(x.builder, native);
    requireSameBuilder(mean.builder, native);
    requireSameBuilder(variance.builder, native);
    requireSameBuilder(scale.builder, native);
    requireSameBuilder(bias.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    const Node &nmean = nodeOf(data->ir, mean.nodeId);
    const Node &nvariance = nodeOf(data->ir, variance.nodeId);
    const Node &nscale = nodeOf(data->ir, scale.nodeId);
    const Node &nbias = nodeOf(data->ir, bias.nodeId);
    if (nx.shape.size() != 4)
        throw std::runtime_error("campello_nn: batchNorm() input must be rank-4 (NCHW)");
    int64_t C = nx.shape[1];
    if (numElements(nmean.shape) != C || numElements(nvariance.shape) != C ||
        numElements(nscale.shape) != C || numElements(nbias.shape) != C)
        throw std::runtime_error("campello_nn: batchNorm() mean/variance/scale/bias must match the input's channel count");
    Node node;
    node.kind = OpKind::BatchNorm;
    node.inputs = {x.nodeId, mean.nodeId, variance.nodeId, scale.nodeId, bias.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    node.floatAttr0 = eps;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::instanceNorm(Operand x, Operand scale, Operand bias, float eps)
{
    requireSameBuilder(x.builder, native);
    requireSameBuilder(scale.builder, native);
    requireSameBuilder(bias.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    const Node &nscale = nodeOf(data->ir, scale.nodeId);
    const Node &nbias = nodeOf(data->ir, bias.nodeId);
    if (nx.shape.size() != 4)
        throw std::runtime_error("campello_nn: instanceNorm() input must be rank-4 (NCHW)");
    int64_t C = nx.shape[1];
    if (numElements(nscale.shape) != C || numElements(nbias.shape) != C)
        throw std::runtime_error("campello_nn: instanceNorm() scale/bias must match the input's channel count");
    Node node;
    node.kind = OpKind::InstanceNorm;
    node.inputs = {x.nodeId, scale.nodeId, bias.nodeId};
    node.dataType = nx.dataType;
    node.shape = nx.shape;
    node.floatAttr0 = eps;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::quantizeLinear(Operand x, float scale, int32_t zeroPoint)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    Node node;
    node.kind = OpKind::QuantizeLinear;
    node.inputs = {x.nodeId};
    node.dataType = DataType::Int8;
    node.shape = nx.shape;
    node.floatAttr0 = scale;
    node.floatAttr1 = (float)zeroPoint;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::dequantizeLinear(Operand x, float scale, int32_t zeroPoint)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    Node node;
    node.kind = OpKind::DequantizeLinear;
    node.inputs = {x.nodeId};
    node.dataType = DataType::Float32;
    node.shape = nx.shape;
    node.floatAttr0 = scale;
    node.floatAttr1 = (float)zeroPoint;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::quantizedMatmul(Operand activation, Operand weightInt8, float weightScale, int32_t weightZeroPoint)
{
    Operand dequantizedWeight = dequantizeLinear(weightInt8, weightScale, weightZeroPoint);
    return matmul(activation, dequantizedWeight);
}

Operand GraphBuilder::matmul(Operand a, Operand b)
{
    requireSameBuilder(a.builder, native);
    requireSameBuilder(b.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &na = nodeOf(data->ir, a.nodeId);
    const Node &nb = nodeOf(data->ir, b.nodeId);
    if (na.shape.size() < 2 || nb.shape.size() < 2)
        throw std::runtime_error("campello_nn: matmul() operands must have rank >= 2");
    if (na.shape.size() != nb.shape.size())
        throw std::runtime_error("campello_nn: matmul() operands must have the same rank (no implicit broadcasting yet)");
    size_t rank = na.shape.size();
    for (size_t i = 0; i < rank - 2; i++)
        if (na.shape[i] != nb.shape[i])
            throw std::runtime_error("campello_nn: matmul() batch dimensions must match");
    int64_t M = na.shape[rank - 2];
    int64_t K = na.shape[rank - 1];
    int64_t K2 = nb.shape[rank - 2];
    int64_t N = nb.shape[rank - 1];
    if (K != K2)
        throw std::runtime_error("campello_nn: matmul() inner dimensions must match (a.cols != b.rows)");
    std::vector<int64_t> shape(na.shape.begin(), na.shape.end() - 2);
    shape.push_back(M);
    shape.push_back(N);
    Node node;
    node.kind = OpKind::MatMul;
    node.inputs = {a.nodeId, b.nodeId};
    node.dataType = na.dataType;
    node.shape = shape;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::gemm(Operand a, Operand b, Operand c, float alpha, float beta)
{
    requireSameBuilder(a.builder, native);
    requireSameBuilder(b.builder, native);
    requireSameBuilder(c.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &na = nodeOf(data->ir, a.nodeId);
    const Node &nb = nodeOf(data->ir, b.nodeId);
    const Node &nc = nodeOf(data->ir, c.nodeId);
    if (na.shape.size() != 2 || nb.shape.size() != 2)
        throw std::runtime_error("campello_nn: gemm() a and b must be rank-2");
    int64_t M = na.shape[0], K = na.shape[1], K2 = nb.shape[0], N = nb.shape[1];
    if (K != K2)
        throw std::runtime_error("campello_nn: gemm() inner dimensions must match (a.cols != b.rows)");
    int64_t cElems = numElements(nc.shape);
    if (cElems != M * N && cElems != N && cElems != 1)
        throw std::runtime_error("campello_nn: gemm() c must broadcast to [M, N] (shape [M,N], [N], or [1])");
    Node node;
    node.kind = OpKind::Gemm;
    node.inputs = {a.nodeId, b.nodeId, c.nodeId};
    node.dataType = na.dataType;
    node.shape = {M, N};
    node.floatAttr0 = alpha;
    node.floatAttr1 = beta;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::reshape(Operand x, const std::vector<int64_t> &shape)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    if (numElements(nx.shape) != numElements(shape))
        throw std::runtime_error("campello_nn: reshape() element count must be preserved");
    Node node;
    node.kind = OpKind::Reshape;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = shape;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::transpose(Operand x, const std::vector<int32_t> &perm)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    if (perm.size() != nx.shape.size())
        throw std::runtime_error("campello_nn: transpose() perm size must equal input rank");
    std::vector<bool> seen(perm.size(), false);
    std::vector<int64_t> shape(perm.size());
    for (size_t i = 0; i < perm.size(); i++)
    {
        int32_t p = perm[i];
        if (p < 0 || (size_t)p >= perm.size() || seen[p])
            throw std::runtime_error("campello_nn: transpose() perm must be a permutation of [0, rank)");
        seen[p] = true;
        shape[i] = nx.shape[p];
    }
    Node node;
    node.kind = OpKind::Transpose;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = shape;
    node.intAttr0.assign(perm.begin(), perm.end());
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::concat(const std::vector<Operand> &xs, int32_t axis)
{
    if (xs.empty())
        throw std::runtime_error("campello_nn: concat() requires at least one operand");
    auto data = (GraphBuilderData *)native;
    for (auto &x : xs)
        requireSameBuilder(x.builder, native);
    const Node &first = nodeOf(data->ir, xs[0].nodeId);
    int32_t rank = (int32_t)first.shape.size();
    int32_t resolvedAxis = axis < 0 ? axis + rank : axis;
    if (resolvedAxis < 0 || resolvedAxis >= rank)
        throw std::runtime_error("campello_nn: concat() axis out of range");
    std::vector<int64_t> shape = first.shape;
    int64_t axisSum = first.shape[resolvedAxis];
    std::vector<size_t> inputs = {xs[0].nodeId};
    for (size_t i = 1; i < xs.size(); i++)
    {
        const Node &n = nodeOf(data->ir, xs[i].nodeId);
        if (n.shape.size() != (size_t)rank)
            throw std::runtime_error("campello_nn: concat() operands must have the same rank");
        for (int32_t d = 0; d < rank; d++)
            if (d != resolvedAxis && n.shape[d] != shape[d])
                throw std::runtime_error("campello_nn: concat() operands must match on every non-concat dimension");
        axisSum += n.shape[resolvedAxis];
        inputs.push_back(xs[i].nodeId);
    }
    shape[resolvedAxis] = axisSum;
    Node node;
    node.kind = OpKind::Concat;
    node.inputs = inputs;
    node.dataType = first.dataType;
    node.shape = shape;
    node.axis = resolvedAxis;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::slice(Operand x, const std::vector<int64_t> &starts, const std::vector<int64_t> &sizes)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    if (starts.size() != nx.shape.size() || sizes.size() != nx.shape.size())
        throw std::runtime_error("campello_nn: slice() starts/sizes must match input rank");
    for (size_t i = 0; i < nx.shape.size(); i++)
        if (starts[i] < 0 || sizes[i] < 0 || starts[i] + sizes[i] > nx.shape[i])
            throw std::runtime_error("campello_nn: slice() out of bounds");
    Node node;
    node.kind = OpKind::Slice;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = sizes;
    node.intAttr0 = starts;
    node.intAttr1 = sizes;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::gather(Operand dataOperand, Operand indices, int32_t axis)
{
    requireSameBuilder(dataOperand.builder, native);
    requireSameBuilder(indices.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nd = nodeOf(data->ir, dataOperand.nodeId);
    const Node &ni = nodeOf(data->ir, indices.nodeId);
    int32_t rank = (int32_t)nd.shape.size();
    int32_t resolvedAxis = axis < 0 ? axis + rank : axis;
    if (resolvedAxis < 0 || resolvedAxis >= rank)
        throw std::runtime_error("campello_nn: gather() axis out of range");
    std::vector<int64_t> shape(nd.shape.begin(), nd.shape.begin() + resolvedAxis);
    shape.insert(shape.end(), ni.shape.begin(), ni.shape.end());
    shape.insert(shape.end(), nd.shape.begin() + resolvedAxis + 1, nd.shape.end());
    Node node;
    node.kind = OpKind::Gather;
    node.inputs = {dataOperand.nodeId, indices.nodeId};
    node.dataType = nd.dataType;
    node.shape = shape;
    node.axis = resolvedAxis;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

Operand GraphBuilder::conv2d(Operand input, Operand weights, const Conv2dDescriptor &desc)
{
    requireSameBuilder(input.builder, native);
    requireSameBuilder(weights.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &ni = nodeOf(data->ir, input.nodeId);
    const Node &nw = nodeOf(data->ir, weights.nodeId);
    if (ni.shape.size() != 4)
        throw std::runtime_error("campello_nn: conv2d() input must be rank-4 (NCHW)");
    if (nw.shape.size() != 4)
        throw std::runtime_error("campello_nn: conv2d() weights must be rank-4 (OIHW)");
    int64_t N = ni.shape[0], C = ni.shape[1], H = ni.shape[2], W = ni.shape[3];
    int64_t O = nw.shape[0], Cg = nw.shape[1], KH = nw.shape[2], KW = nw.shape[3];
    if (desc.groups < 1 || C % desc.groups != 0 || O % desc.groups != 0)
        throw std::runtime_error("campello_nn: conv2d() groups must evenly divide input/output channels");
    if (Cg != C / desc.groups)
        throw std::runtime_error("campello_nn: conv2d() weights.shape[1] must equal input channels / groups");
    int64_t outH = (H + desc.paddingTop + desc.paddingBottom - desc.dilationY * (KH - 1) - 1) / desc.strideY + 1;
    int64_t outW = (W + desc.paddingLeft + desc.paddingRight - desc.dilationX * (KW - 1) - 1) / desc.strideX + 1;
    if (outH <= 0 || outW <= 0)
        throw std::runtime_error("campello_nn: conv2d() produces a non-positive output spatial size");
    Node node;
    node.kind = OpKind::Conv2d;
    node.inputs = {input.nodeId, weights.nodeId};
    node.dataType = ni.dataType;
    node.shape = {N, O, outH, outW};
    node.convParams = desc;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

namespace
{
    // Takes the Operand's builder/nodeId already extracted by the caller (a
    // GraphBuilder member function, which has friend access to Operand's private
    // fields/constructor) and returns the new node's index — this free function
    // has neither, so it can't touch Operand's private members directly.
    size_t pool2dImpl(void *native, GraphBuilderData *data, OpKind kind, void *xBuilder, size_t xNodeId,
                       const Pool2dDescriptor &desc, const char *opName)
    {
        requireSameBuilder(xBuilder, native);
        const Node &nx = nodeOf(data->ir, xNodeId);
        if (nx.shape.size() != 4)
            throw std::runtime_error(std::string("campello_nn: ") + opName + "() input must be rank-4 (NCHW)");
        int64_t N = nx.shape[0], C = nx.shape[1], H = nx.shape[2], W = nx.shape[3];
        int64_t outH = (H + desc.paddingTop + desc.paddingBottom - desc.kernelHeight) / desc.strideY + 1;
        int64_t outW = (W + desc.paddingLeft + desc.paddingRight - desc.kernelWidth) / desc.strideX + 1;
        if (outH <= 0 || outW <= 0)
            throw std::runtime_error(std::string("campello_nn: ") + opName + "() produces a non-positive output spatial size");
        Node node;
        node.kind = kind;
        node.inputs = {xNodeId};
        node.dataType = nx.dataType;
        node.shape = {N, C, outH, outW};
        node.poolParams = desc;
        data->ir.nodes.push_back(std::move(node));
        return data->ir.nodes.size() - 1;
    }
}

Operand GraphBuilder::maxPool2d(Operand x, const Pool2dDescriptor &desc)
{
    auto data = (GraphBuilderData *)native;
    size_t nodeId = pool2dImpl(native, data, OpKind::MaxPool2d, x.builder, x.nodeId, desc, "maxPool2d");
    return Operand(native, nodeId);
}

Operand GraphBuilder::avgPool2d(Operand x, const Pool2dDescriptor &desc)
{
    auto data = (GraphBuilderData *)native;
    size_t nodeId = pool2dImpl(native, data, OpKind::AvgPool2d, x.builder, x.nodeId, desc, "avgPool2d");
    return Operand(native, nodeId);
}

Operand GraphBuilder::resize(Operand x, const ResizeDescriptor &desc)
{
    requireSameBuilder(x.builder, native);
    auto data = (GraphBuilderData *)native;
    const Node &nx = nodeOf(data->ir, x.nodeId);
    if (nx.shape.size() != 4)
        throw std::runtime_error("campello_nn: resize() input must be rank-4 (NCHW)");
    if (desc.outputHeight <= 0 || desc.outputWidth <= 0)
        throw std::runtime_error("campello_nn: resize() outputHeight/outputWidth must be positive");
    Node node;
    node.kind = OpKind::Resize;
    node.inputs = {x.nodeId};
    node.dataType = nx.dataType;
    node.shape = {nx.shape[0], nx.shape[1], desc.outputHeight, desc.outputWidth};
    node.resizeParams = desc;
    data->ir.nodes.push_back(std::move(node));
    return Operand(native, data->ir.nodes.size() - 1);
}

std::shared_ptr<Graph> GraphBuilder::build(const std::unordered_map<std::string, Operand> &outputs)
{
    auto data = (GraphBuilderData *)native;
    GraphIR ir = data->ir;
    for (auto &[name, op] : outputs)
    {
        requireSameBuilder(op.builder, native);
        ir.outputs.push_back({name, op.nodeId});
    }

    auto ctxData = (ContextData *)data->context->native;
    void *compiled = ctxData->backend->compileGraph(ir);
    auto gd = new GraphData{ctxData->backend.get(), compiled};
    return std::shared_ptr<Graph>(new Graph((void *)gd));
}

std::vector<uint8_t> GraphBuilder::serialize(const std::unordered_map<std::string, Operand> &outputs)
{
    auto data = (GraphBuilderData *)native;
    GraphIR ir = data->ir;
    for (auto &[name, op] : outputs)
    {
        requireSameBuilder(op.builder, native);
        ir.outputs.push_back({name, op.nodeId});
    }
    return serializeGraphIR(ir);
}

std::shared_ptr<Graph> GraphBuilder::deserialize(std::shared_ptr<Context> context, const uint8_t *data, size_t size)
{
    GraphIR ir = deserializeGraphIR(data, size);
    auto ctxData = (ContextData *)context->native;
    void *compiled = ctxData->backend->compileGraph(ir);
    auto gd = new GraphData{ctxData->backend.get(), compiled};
    return std::shared_ptr<Graph>(new Graph((void *)gd));
}

std::vector<int64_t> systems::leal::campello_nn::internal::operandShapeForImport(const Operand &op)
{
    auto data = (GraphBuilderData *)op.builder;
    return data->ir.nodes[op.nodeId].shape;
}

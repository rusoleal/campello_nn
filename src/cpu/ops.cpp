#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "ops.hpp"
#include "strides.hpp"

using namespace systems::leal::campello_nn;

namespace
{
    // Every CpuValue is genuinely Float32 in memory by the time a kernel runs —
    // cpu_backend.cpp decodes Float16 Input/Constant bytes to Float32 up front and
    // only re-encodes at the very end, so kernels never see Float16 bytes. This just
    // rejects the dtypes that make no sense for elementwise/compute ops (Int32/
    // Uint32/Int8), while accepting both of the dtypes that actually reach here.
    void requireFloat32(const Node &node, const char *opName)
    {
        if (node.dataType != DataType::Float32 && node.dataType != DataType::Float16)
            throw std::runtime_error(std::string("campello_nn: CPU backend only supports Float32/Float16 for ") + opName);
    }

    CpuValue makeFloatValue(const std::vector<int64_t> &shape)
    {
        CpuValue v;
        v.shape = shape;
        v.dataType = DataType::Float32;
        v.bytes.resize(numElements(shape) * sizeof(float));
        return v;
    }

    CpuValue makeInt8Value(const std::vector<int64_t> &shape)
    {
        CpuValue v;
        v.shape = shape;
        v.dataType = DataType::Int8;
        v.bytes.resize(numElements(shape));
        return v;
    }

    // Maps an output multi-index to the corresponding flat index into a
    // (possibly lower-rank, possibly broadcast) operand: a dimension of size 1
    // (or a missing leading dimension) always reads index 0 along that axis,
    // per NumPy/ONNX broadcasting rules — same alignment `computeBroadcastShape`
    // (in graph_builder.cpp) used to compute `node.shape` in the first place.
    int64_t broadcastInputIndex(const std::vector<int64_t> &outIdx, const std::vector<int64_t> &inShape,
                                 const std::vector<int64_t> &inStrides)
    {
        size_t outRank = outIdx.size();
        size_t inRank = inShape.size();
        size_t offset = outRank - inRank;
        int64_t flat = 0;
        for (size_t i = 0; i < inRank; i++)
        {
            int64_t coord = inShape[i] == 1 ? 0 : outIdx[offset + i];
            flat += coord * inStrides[i];
        }
        return flat;
    }

    template <typename BinOp>
    void evalBroadcastBinaryOp(const Node &node, std::vector<CpuValue> &values, CpuValue &out, BinOp op)
    {
        const CpuValue &a = values[node.inputs[0]];
        const CpuValue &b = values[node.inputs[1]];
        out = makeFloatValue(node.shape);

        // Fast path: both operands already match the output shape exactly (the
        // common case, e.g. transformer residual adds) — no broadcast indexing.
        if (a.shape == node.shape && b.shape == node.shape)
        {
            size_t n = out.floatCount();
            for (size_t i = 0; i < n; i++)
                out.f()[i] = op(a.f()[i], b.f()[i]);
            return;
        }

        auto aStrides = rowMajorStrides(a.shape);
        auto bStrides = rowMajorStrides(b.shape);
        int64_t total = numElements(node.shape);
        for (int64_t flat = 0; flat < total; flat++)
        {
            auto outIdx = unravelIndex(flat, node.shape);
            int64_t aIdx = broadcastInputIndex(outIdx, a.shape, aStrides);
            int64_t bIdx = broadcastInputIndex(outIdx, b.shape, bStrides);
            out.f()[flat] = op(a.f()[aIdx], b.f()[bIdx]);
        }
    }

    void evalAdd(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "add");
        evalBroadcastBinaryOp(node, values, out, [](float x, float y)
                               { return x + y; });
    }

    void evalMul(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "mul");
        evalBroadcastBinaryOp(node, values, out, [](float x, float y)
                               { return x * y; });
    }

    void evalGelu(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "gelu");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        size_t n = out.floatCount();
        for (size_t i = 0; i < n; i++)
        {
            float v = x.f()[i];
            out.f()[i] = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
        }
    }

    void evalRelu(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "relu");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        size_t n = out.floatCount();
        for (size_t i = 0; i < n; i++)
            out.f()[i] = std::max(x.f()[i], 0.0f);
    }

    void evalSigmoid(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "sigmoid");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        size_t n = out.floatCount();
        for (size_t i = 0; i < n; i++)
            out.f()[i] = 1.0f / (1.0f + std::exp(-x.f()[i]));
    }

    void evalSoftmax(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "softmax");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        auto strides = rowMajorStrides(node.shape);
        int64_t rank = (int64_t)node.shape.size();
        int64_t axisSize = node.shape[node.axis];
        int64_t axisStride = strides[node.axis];
        int64_t outerTotal = numElements(node.shape) / axisSize;

        for (int64_t o = 0; o < outerTotal; o++)
        {
            int64_t base = 0, remaining = o;
            for (int64_t d = rank - 1; d >= 0; d--)
            {
                if (d == node.axis)
                    continue;
                int64_t dimSize = node.shape[d];
                int64_t coord = remaining % dimSize;
                remaining /= dimSize;
                base += coord * strides[d];
            }

            float maxV = x.f()[base];
            for (int64_t k = 1; k < axisSize; k++)
                maxV = std::max(maxV, x.f()[base + k * axisStride]);
            float sum = 0.f;
            for (int64_t k = 0; k < axisSize; k++)
            {
                float e = std::exp(x.f()[base + k * axisStride] - maxV);
                out.f()[base + k * axisStride] = e;
                sum += e;
            }
            for (int64_t k = 0; k < axisSize; k++)
                out.f()[base + k * axisStride] /= sum;
        }
    }

    void evalLayerNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "layerNorm");
        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &scale = values[node.inputs[1]];
        const CpuValue &bias = values[node.inputs[2]];
        out = makeFloatValue(node.shape);
        int64_t lastDim = node.shape.back();
        int64_t outerTotal = numElements(node.shape) / lastDim;
        float eps = node.floatAttr0;

        for (int64_t o = 0; o < outerTotal; o++)
        {
            const float *row = x.f() + o * lastDim;
            float mean = 0.f;
            for (int64_t k = 0; k < lastDim; k++)
                mean += row[k];
            mean /= (float)lastDim;
            float var = 0.f;
            for (int64_t k = 0; k < lastDim; k++)
            {
                float d = row[k] - mean;
                var += d * d;
            }
            var /= (float)lastDim;
            float invStd = 1.0f / std::sqrt(var + eps);
            float *outRow = out.f() + o * lastDim;
            for (int64_t k = 0; k < lastDim; k++)
                outRow[k] = (row[k] - mean) * invStd * scale.f()[k] + bias.f()[k];
        }
    }

    void evalRmsNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "rmsNorm");
        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &scale = values[node.inputs[1]];
        out = makeFloatValue(node.shape);
        int64_t lastDim = node.shape.back();
        int64_t outerTotal = numElements(node.shape) / lastDim;
        float eps = node.floatAttr0;

        for (int64_t o = 0; o < outerTotal; o++)
        {
            const float *row = x.f() + o * lastDim;
            float meanSquare = 0.f;
            for (int64_t k = 0; k < lastDim; k++)
                meanSquare += row[k] * row[k];
            meanSquare /= (float)lastDim;
            float invRms = 1.0f / std::sqrt(meanSquare + eps);
            float *outRow = out.f() + o * lastDim;
            for (int64_t k = 0; k < lastDim; k++)
                outRow[k] = row[k] * invRms * scale.f()[k];
        }
    }

    void evalBatchNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "batchNorm");
        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &mean = values[node.inputs[1]];
        const CpuValue &variance = values[node.inputs[2]];
        const CpuValue &scale = values[node.inputs[3]];
        const CpuValue &bias = values[node.inputs[4]];
        out = makeFloatValue(node.shape);
        float eps = node.floatAttr0;

        int64_t N = node.shape[0], C = node.shape[1], H = node.shape[2], W = node.shape[3];
        std::vector<float> invStd(C);
        for (int64_t c = 0; c < C; c++)
            invStd[c] = 1.0f / std::sqrt(variance.f()[c] + eps);

        for (int64_t n = 0; n < N; n++)
        {
            for (int64_t c = 0; c < C; c++)
            {
                const float *plane = x.f() + (n * C + c) * H * W;
                float *outPlane = out.f() + (n * C + c) * H * W;
                float m = mean.f()[c], s = scale.f()[c], b = bias.f()[c], inv = invStd[c];
                for (int64_t k = 0; k < H * W; k++)
                    outPlane[k] = (plane[k] - m) * inv * s + b;
            }
        }
    }

    void evalInstanceNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "instanceNorm");
        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &scale = values[node.inputs[1]];
        const CpuValue &bias = values[node.inputs[2]];
        out = makeFloatValue(node.shape);
        float eps = node.floatAttr0;

        int64_t N = node.shape[0], C = node.shape[1], H = node.shape[2], W = node.shape[3];
        int64_t spatial = H * W;

        for (int64_t n = 0; n < N; n++)
        {
            for (int64_t c = 0; c < C; c++)
            {
                const float *plane = x.f() + (n * C + c) * spatial;
                float mean = 0.f;
                for (int64_t k = 0; k < spatial; k++)
                    mean += plane[k];
                mean /= (float)spatial;
                float var = 0.f;
                for (int64_t k = 0; k < spatial; k++)
                {
                    float d = plane[k] - mean;
                    var += d * d;
                }
                var /= (float)spatial;
                float invStd = 1.0f / std::sqrt(var + eps);
                float s = scale.f()[c], b = bias.f()[c];
                float *outPlane = out.f() + (n * C + c) * spatial;
                for (int64_t k = 0; k < spatial; k++)
                    outPlane[k] = (plane[k] - mean) * invStd * s + b;
            }
        }
    }

    void evalMatMul(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "matmul");
        const CpuValue &a = values[node.inputs[0]];
        const CpuValue &b = values[node.inputs[1]];
        out = makeFloatValue(node.shape);

        size_t rank = a.shape.size();
        int64_t M = a.shape[rank - 2];
        int64_t K = a.shape[rank - 1];
        int64_t N = b.shape[rank - 1];
        int64_t batchCount = numElements(node.shape) / (M * N);

        for (int64_t batch = 0; batch < batchCount; batch++)
        {
            const float *aBase = a.f() + batch * M * K;
            const float *bBase = b.f() + batch * K * N;
            float *outBase = out.f() + batch * M * N;
            for (int64_t m = 0; m < M; m++)
            {
                for (int64_t n = 0; n < N; n++)
                {
                    float sum = 0.f;
                    for (int64_t k = 0; k < K; k++)
                        sum += aBase[m * K + k] * bBase[k * N + n];
                    outBase[m * N + n] = sum;
                }
            }
        }
    }

    void evalGemm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "gemm");
        const CpuValue &a = values[node.inputs[0]];
        const CpuValue &b = values[node.inputs[1]];
        const CpuValue &c = values[node.inputs[2]];
        out = makeFloatValue(node.shape);

        int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
        float alpha = node.floatAttr0, beta = node.floatAttr1;
        size_t cElems = c.floatCount();

        for (int64_t m = 0; m < M; m++)
        {
            for (int64_t n = 0; n < N; n++)
            {
                float sum = 0.f;
                for (int64_t k = 0; k < K; k++)
                    sum += a.f()[m * K + k] * b.f()[k * N + n];
                float cv = cElems == 1 ? c.f()[0] : (cElems == (size_t)N ? c.f()[n] : c.f()[m * N + n]);
                out.f()[m * N + n] = alpha * sum + beta * cv;
            }
        }
    }

    void evalReshape(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        out = values[node.inputs[0]];
        out.shape = node.shape;
    }

    void evalTranspose(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "transpose");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        auto inStrides = rowMajorStrides(x.shape);
        const auto &perm = node.intAttr0;
        int64_t total = numElements(node.shape);

        for (int64_t flat = 0; flat < total; flat++)
        {
            auto outIdx = unravelIndex(flat, node.shape);
            std::vector<int64_t> inIdx(perm.size());
            for (size_t i = 0; i < perm.size(); i++)
                inIdx[perm[i]] = outIdx[i];
            out.f()[flat] = x.f()[ravelIndex(inIdx, inStrides)];
        }
    }

    void evalConcat(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "concat");
        out = makeFloatValue(node.shape);
        auto outStrides = rowMajorStrides(node.shape);
        int64_t axisOffset = 0;

        for (size_t inputIdx : node.inputs)
        {
            const CpuValue &in = values[inputIdx];
            int64_t total = numElements(in.shape);
            for (int64_t flat = 0; flat < total; flat++)
            {
                auto idx = unravelIndex(flat, in.shape);
                idx[node.axis] += axisOffset;
                out.f()[ravelIndex(idx, outStrides)] = in.f()[flat];
            }
            axisOffset += in.shape[node.axis];
        }
    }

    void evalSlice(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "slice");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        auto inStrides = rowMajorStrides(x.shape);
        const auto &starts = node.intAttr0;
        int64_t total = numElements(node.shape);

        for (int64_t flat = 0; flat < total; flat++)
        {
            auto idx = unravelIndex(flat, node.shape);
            for (size_t d = 0; d < idx.size(); d++)
                idx[d] += starts[d];
            out.f()[flat] = x.f()[ravelIndex(idx, inStrides)];
        }
    }

    void evalGather(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        const CpuValue &data = values[node.inputs[0]];
        const CpuValue &indices = values[node.inputs[1]];
        if (data.dataType != DataType::Float32)
            throw std::runtime_error("campello_nn: CPU backend only supports Float32 data for gather()");
        if (indices.dataType != DataType::Int32 && indices.dataType != DataType::Uint32)
            throw std::runtime_error("campello_nn: CPU backend requires Int32 or Uint32 indices for gather()");

        out = makeFloatValue(node.shape);
        int64_t axis = node.axis;
        int64_t outerSize = 1;
        for (int64_t d = 0; d < axis; d++)
            outerSize *= data.shape[d];
        int64_t axisSize = data.shape[axis];
        int64_t innerSize = 1;
        for (size_t d = axis + 1; d < data.shape.size(); d++)
            innerSize *= data.shape[d];
        int64_t numIndices = numElements(indices.shape);

        for (int64_t o = 0; o < outerSize; o++)
        {
            for (int64_t ii = 0; ii < numIndices; ii++)
            {
                int64_t idx = indices.dataType == DataType::Int32 ? (int64_t)indices.i32()[ii] : (int64_t)indices.u32()[ii];
                const float *src = data.f() + (o * axisSize + idx) * innerSize;
                float *dst = out.f() + (o * numIndices + ii) * innerSize;
                for (int64_t k = 0; k < innerSize; k++)
                    dst[k] = src[k];
            }
        }
        (void)axisSize;
    }

    void evalQuantizeLinear(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        const CpuValue &x = values[node.inputs[0]];
        // By kernel time, any float-family input (originally Float32 or Float16) is
        // genuinely Float32 in memory — see requireFloat32's comment. Anything else
        // (Int32/Uint32/Int8) means the graph fed a non-float tensor into quantizeLinear.
        if (x.dataType != DataType::Float32)
            throw std::runtime_error("campello_nn: quantizeLinear() requires a float input");

        out = makeInt8Value(node.shape);
        float scale = node.floatAttr0;
        float zeroPoint = node.floatAttr1;
        size_t n = x.floatCount();
        int8_t *o = (int8_t *)out.bytes.data();
        for (size_t i = 0; i < n; i++)
        {
            float q = std::round(x.f()[i] / scale) + zeroPoint;
            q = std::clamp(q, -128.f, 127.f);
            o[i] = (int8_t)q;
        }
    }

    void evalDequantizeLinear(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        const CpuValue &x = values[node.inputs[0]];
        if (x.dataType != DataType::Int8)
            throw std::runtime_error("campello_nn: dequantizeLinear() requires an Int8 input");

        out = makeFloatValue(node.shape);
        float scale = node.floatAttr0;
        float zeroPoint = node.floatAttr1;
        size_t n = numElements(node.shape);
        const int8_t *xi = (const int8_t *)x.bytes.data();
        for (size_t i = 0; i < n; i++)
            out.f()[i] = scale * ((float)xi[i] - zeroPoint);
    }

    void evalConv2d(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "conv2d");
        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &w = values[node.inputs[1]];
        out = makeFloatValue(node.shape);
        const Conv2dDescriptor &p = node.convParams;

        int64_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
        int64_t O = w.shape[0], Cg = w.shape[1], KH = w.shape[2], KW = w.shape[3];
        int64_t outH = node.shape[2], outW = node.shape[3];
        int64_t inPerGroup = C / p.groups;
        int64_t outPerGroup = O / p.groups;

        for (int64_t n = 0; n < N; n++)
        {
            for (int64_t o = 0; o < O; o++)
            {
                int64_t group = o / outPerGroup;
                int64_t inChannelBase = group * inPerGroup;
                for (int64_t oh = 0; oh < outH; oh++)
                {
                    for (int64_t ow = 0; ow < outW; ow++)
                    {
                        float sum = 0.f;
                        for (int64_t ci = 0; ci < Cg; ci++)
                        {
                            int64_t c = inChannelBase + ci;
                            for (int64_t kh = 0; kh < KH; kh++)
                            {
                                int64_t ih = oh * p.strideY - p.paddingTop + kh * p.dilationY;
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int64_t kw = 0; kw < KW; kw++)
                                {
                                    int64_t iw = ow * p.strideX - p.paddingLeft + kw * p.dilationX;
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    float xv = x.f()[((n * C + c) * H + ih) * W + iw];
                                    float wv = w.f()[((o * Cg + ci) * KH + kh) * KW + kw];
                                    sum += xv * wv;
                                }
                            }
                        }
                        out.f()[((n * O + o) * outH + oh) * outW + ow] = sum;
                    }
                }
            }
        }
    }

    void evalPool2d(const Node &node, std::vector<CpuValue> &values, CpuValue &out, bool isMax)
    {
        requireFloat32(node, isMax ? "maxPool2d" : "avgPool2d");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        const Pool2dDescriptor &p = node.poolParams;

        int64_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
        int64_t outH = node.shape[2], outW = node.shape[3];

        for (int64_t n = 0; n < N; n++)
        {
            for (int64_t c = 0; c < C; c++)
            {
                for (int64_t oh = 0; oh < outH; oh++)
                {
                    for (int64_t ow = 0; ow < outW; ow++)
                    {
                        float acc = isMax ? -std::numeric_limits<float>::infinity() : 0.f;
                        int64_t count = 0;
                        for (int64_t kh = 0; kh < p.kernelHeight; kh++)
                        {
                            int64_t ih = oh * p.strideY - p.paddingTop + kh;
                            if (ih < 0 || ih >= H)
                                continue;
                            for (int64_t kw = 0; kw < p.kernelWidth; kw++)
                            {
                                int64_t iw = ow * p.strideX - p.paddingLeft + kw;
                                if (iw < 0 || iw >= W)
                                    continue;
                                float v = x.f()[((n * C + c) * H + ih) * W + iw];
                                acc = isMax ? std::max(acc, v) : acc + v;
                                count++;
                            }
                        }
                        out.f()[((n * C + c) * outH + oh) * outW + ow] = isMax ? acc : (count > 0 ? acc / count : 0.f);
                    }
                }
            }
        }
    }

    // Maps a destination coordinate back to a source coordinate, per MPSGraphResizeOps'
    // documented centerResult/alignCorners semantics.
    float resizeSrcCoord(int64_t dst, int64_t inSize, int64_t outSize, bool centerResult, bool alignCorners)
    {
        if (alignCorners)
        {
            float scale = outSize > 1 ? (float)(inSize - 1) / (float)(outSize - 1) : 0.f;
            return dst * scale;
        }
        float scale = (float)inSize / (float)outSize;
        float src = centerResult ? (dst + 0.5f) * scale - 0.5f : dst * scale;
        return std::clamp(src, 0.f, (float)(inSize - 1));
    }

    void evalResize(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "resize");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        const ResizeDescriptor &p = node.resizeParams;

        int64_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
        int64_t outH = node.shape[2], outW = node.shape[3];

        for (int64_t n = 0; n < N; n++)
        {
            for (int64_t c = 0; c < C; c++)
            {
                const float *plane = x.f() + (n * C + c) * H * W;
                float *outPlane = out.f() + (n * C + c) * outH * outW;
                for (int64_t oh = 0; oh < outH; oh++)
                {
                    float srcH = resizeSrcCoord(oh, H, outH, p.centerResult, p.alignCorners);
                    for (int64_t ow = 0; ow < outW; ow++)
                    {
                        float srcW = resizeSrcCoord(ow, W, outW, p.centerResult, p.alignCorners);
                        float value;
                        if (p.mode == ResizeMode::Nearest)
                        {
                            int64_t ih, iw;
                            if (p.nearestRoundsDown)
                            {
                                ih = std::clamp((int64_t)std::floor(srcH), (int64_t)0, H - 1);
                                iw = std::clamp((int64_t)std::floor(srcW), (int64_t)0, W - 1);
                            }
                            else
                            {
                                ih = std::clamp((int64_t)std::round(srcH), (int64_t)0, H - 1);
                                iw = std::clamp((int64_t)std::round(srcW), (int64_t)0, W - 1);
                            }
                            value = plane[ih * W + iw];
                        }
                        else
                        {
                            int64_t h0 = std::clamp((int64_t)std::floor(srcH), (int64_t)0, H - 1);
                            int64_t h1 = std::min(h0 + 1, H - 1);
                            int64_t w0 = std::clamp((int64_t)std::floor(srcW), (int64_t)0, W - 1);
                            int64_t w1 = std::min(w0 + 1, W - 1);
                            float fh = srcH - h0;
                            float fw = srcW - w0;
                            float top = plane[h0 * W + w0] * (1 - fw) + plane[h0 * W + w1] * fw;
                            float bottom = plane[h1 * W + w0] * (1 - fw) + plane[h1 * W + w1] * fw;
                            value = top * (1 - fh) + bottom * fh;
                        }
                        outPlane[oh * outW + ow] = value;
                    }
                }
            }
        }
    }
}

void systems::leal::campello_nn::evalNode(const Node &node, size_t selfIndex, std::vector<CpuValue> &values)
{
    CpuValue out;
    switch (node.kind)
    {
    case OpKind::Add: evalAdd(node, values, out); break;
    case OpKind::Mul: evalMul(node, values, out); break;
    case OpKind::Gelu: evalGelu(node, values, out); break;
    case OpKind::Relu: evalRelu(node, values, out); break;
    case OpKind::Sigmoid: evalSigmoid(node, values, out); break;
    case OpKind::Softmax: evalSoftmax(node, values, out); break;
    case OpKind::LayerNorm: evalLayerNorm(node, values, out); break;
    case OpKind::RmsNorm: evalRmsNorm(node, values, out); break;
    case OpKind::BatchNorm: evalBatchNorm(node, values, out); break;
    case OpKind::InstanceNorm: evalInstanceNorm(node, values, out); break;
    case OpKind::MatMul: evalMatMul(node, values, out); break;
    case OpKind::Gemm: evalGemm(node, values, out); break;
    case OpKind::Reshape: evalReshape(node, values, out); break;
    case OpKind::Transpose: evalTranspose(node, values, out); break;
    case OpKind::Concat: evalConcat(node, values, out); break;
    case OpKind::Slice: evalSlice(node, values, out); break;
    case OpKind::Gather: evalGather(node, values, out); break;
    case OpKind::QuantizeLinear: evalQuantizeLinear(node, values, out); break;
    case OpKind::DequantizeLinear: evalDequantizeLinear(node, values, out); break;
    case OpKind::Conv2d: evalConv2d(node, values, out); break;
    case OpKind::MaxPool2d: evalPool2d(node, values, out, true); break;
    case OpKind::AvgPool2d: evalPool2d(node, values, out, false); break;
    case OpKind::Resize: evalResize(node, values, out); break;
    case OpKind::Input:
    case OpKind::Constant:
        throw std::runtime_error("campello_nn: Input/Constant nodes must be pre-populated before evalNode()");
    }
    values[selfIndex] = std::move(out);
}

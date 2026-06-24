#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "ops.hpp"
#include "strides.hpp"
#include "thread_pool.hpp"
#include "simd_kernels.hpp"

namespace
{
    // kElementwiseGrain/kRowGrain/kNCUnitGrain/kMatmulRowGrain moved to
    // simd_kernels.hpp (shared with the Arch-templated kernel bodies there).
    // kConvUnitGrain stays here — conv2d/pool2d aren't vectorized/dispatched.
    constexpr int64_t kConvUnitGrain = 4;
}

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

    // Each wrapper below dispatches to the Arch-templated kernel in
    // simd_kernels.hpp: AVX2+FMA3 if available at runtime (cached check, see
    // cpuSupportsAvx2Fma()), else the safe SSE2/NEON default_arch baseline —
    // mechanically the same code as before this round, just reached through a
    // template parameter instead of the fixed FloatBatch/kSimdWidth aliases.
    void evalAdd(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "add");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalBroadcastBinaryOpImpl<xsimd::fma3<xsimd::avx2>>(node, values, out, AddOp{});
        else
#endif
            evalBroadcastBinaryOpImpl<xsimd::default_arch>(node, values, out, AddOp{});
    }

    void evalMul(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "mul");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalBroadcastBinaryOpImpl<xsimd::fma3<xsimd::avx2>>(node, values, out, MulOp{});
        else
#endif
            evalBroadcastBinaryOpImpl<xsimd::default_arch>(node, values, out, MulOp{});
    }

    void evalGelu(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "gelu");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalGeluImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalGeluImpl<xsimd::default_arch>(node, values, out);
    }

    void evalRelu(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "relu");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalReluImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalReluImpl<xsimd::default_arch>(node, values, out);
    }

    void evalSigmoid(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "sigmoid");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalSigmoidImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalSigmoidImpl<xsimd::default_arch>(node, values, out);
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

        parallelFor(0, outerTotal, kRowGrain, [&](int64_t obegin, int64_t oend)
                    {
            for (int64_t o = obegin; o < oend; o++)
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
            } });
    }

    void evalLayerNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "layerNorm");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalLayerNormImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalLayerNormImpl<xsimd::default_arch>(node, values, out);
    }

    void evalRmsNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "rmsNorm");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalRmsNormImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalRmsNormImpl<xsimd::default_arch>(node, values, out);
    }

    void evalBatchNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "batchNorm");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalBatchNormImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalBatchNormImpl<xsimd::default_arch>(node, values, out);
    }

    void evalInstanceNorm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "instanceNorm");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalInstanceNormImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalInstanceNormImpl<xsimd::default_arch>(node, values, out);
    }

    void evalMatMul(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "matmul");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalMatMulImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalMatMulImpl<xsimd::default_arch>(node, values, out);
    }

    void evalGemm(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        requireFloat32(node, "gemm");
#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
        if (cpuSupportsAvx2Fma())
            evalGemmImpl<xsimd::fma3<xsimd::avx2>>(node, values, out);
        else
#endif
            evalGemmImpl<xsimd::default_arch>(node, values, out);
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

        parallelFor(0, N * O, kConvUnitGrain, [&](int64_t begin, int64_t end)
                    {
            for (int64_t no = begin; no < end; no++)
            {
                int64_t n = no / O;
                int64_t o = no % O;
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
                        out.f()[no * outH * outW + oh * outW + ow] = sum;
                    }
                }
            } });
    }

    void evalPool2d(const Node &node, std::vector<CpuValue> &values, CpuValue &out, bool isMax)
    {
        requireFloat32(node, isMax ? "maxPool2d" : "avgPool2d");
        const CpuValue &x = values[node.inputs[0]];
        out = makeFloatValue(node.shape);
        const Pool2dDescriptor &p = node.poolParams;

        int64_t N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
        int64_t outH = node.shape[2], outW = node.shape[3];

        parallelFor(0, N * C, kNCUnitGrain, [&](int64_t begin, int64_t end)
                    {
            for (int64_t nc = begin; nc < end; nc++)
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
                                float v = x.f()[nc * H * W + ih * W + iw];
                                acc = isMax ? std::max(acc, v) : acc + v;
                                count++;
                            }
                        }
                        out.f()[nc * outH * outW + oh * outW + ow] = isMax ? acc : (count > 0 ? acc / count : 0.f);
                    }
                }
            } });
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

        parallelFor(0, N * C, kNCUnitGrain, [&](int64_t begin, int64_t end)
                    {
            for (int64_t nc = begin; nc < end; nc++)
            {
                const float *plane = x.f() + nc * H * W;
                float *outPlane = out.f() + nc * outH * outW;
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
            } });
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

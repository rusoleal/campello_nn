#pragma once

#include <algorithm>
#include <cmath>
#include "ops.hpp"
#include "strides.hpp"
#include "simd.hpp"
#include "thread_pool.hpp"

namespace systems::leal::campello_nn
{

    // Grain sizes: see ops.cpp's original comment (kept there too, for
    // kConvUnitGrain, which conv2d/pool2d still use directly in ops.cpp without
    // an Arch-templated/AVX2-dispatched version). Large enough that today's
    // small unit-test shapes stay on parallelFor's serial fast-path, small
    // enough that real model-sized tensors get split across threads. Not
    // empirically tuned — revisit once Phase 6's benchmark harness exists.
    constexpr int64_t kElementwiseGrain = 65536;
    constexpr int64_t kRowGrain = 64;
    constexpr int64_t kNCUnitGrain = 8;
    constexpr int64_t kMatmulRowGrain = 8;

    // Named functor types (not lambdas): evalAdd/evalMul pass these into
    // evalBroadcastBinaryOpImpl. Lambda closure types are anonymous and can't
    // appear in the extern-template / explicit-instantiation declarations below,
    // which need a nameable type to instantiate evalBroadcastBinaryOpImpl for
    // xsimd::fma3<xsimd::avx2> in a different translation unit.
    struct AddOp
    {
        template <class T>
        T operator()(T x, T y) const { return x + y; }
    };

    struct MulOp
    {
        template <class T>
        T operator()(T x, T y) const { return x * y; }
    };

    // Maps an output multi-index to the corresponding flat index into a
    // (possibly lower-rank, possibly broadcast) operand — see ops.cpp's
    // original comment on this; duplicated here since both the default-arch
    // and AVX2 instantiations of evalBroadcastBinaryOpImpl need it.
    inline int64_t simdBroadcastInputIndex(const std::vector<int64_t> &outIdx, const std::vector<int64_t> &inShape,
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

    inline CpuValue simdMakeFloatValue(const std::vector<int64_t> &shape)
    {
        CpuValue v;
        v.shape = shape;
        v.dataType = DataType::Float32;
        v.bytes.resize(numElements(shape) * sizeof(float));
        return v;
    }

    // --- Arch-templated kernel bodies -------------------------------------
    // Each is the exact same logic as the corresponding eval* function added
    // in the SIMD round (see TODO.md) — this round is pure dispatch
    // infrastructure, not new vectorization. `Batch`/`width` replace the fixed
    // `FloatBatch`/`kSimdWidth` from simd.hpp so the same body compiles against
    // any xsimd arch tag, not just the default one.

    template <class Arch, class BinOp>
    void evalBroadcastBinaryOpImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out, BinOp op)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &a = values[node.inputs[0]];
        const CpuValue &b = values[node.inputs[1]];
        out = simdMakeFloatValue(node.shape);

        if (a.shape == node.shape && b.shape == node.shape)
        {
            int64_t n = (int64_t)out.floatCount();
            parallelFor(0, n, kElementwiseGrain, [&](int64_t begin, int64_t end)
                        {
                int64_t i = begin;
                for (; i + width <= end; i += width)
                {
                    auto va = Batch::load_unaligned(a.f() + i);
                    auto vb = Batch::load_unaligned(b.f() + i);
                    op(va, vb).store_unaligned(out.f() + i);
                }
                for (; i < end; i++)
                    out.f()[i] = op(a.f()[i], b.f()[i]); });
            return;
        }

        auto aStrides = rowMajorStrides(a.shape);
        auto bStrides = rowMajorStrides(b.shape);
        int64_t total = numElements(node.shape);
        parallelFor(0, total, kElementwiseGrain, [&](int64_t begin, int64_t end)
                    {
            for (int64_t flat = begin; flat < end; flat++)
            {
                auto outIdx = unravelIndex(flat, node.shape);
                int64_t aIdx = simdBroadcastInputIndex(outIdx, a.shape, aStrides);
                int64_t bIdx = simdBroadcastInputIndex(outIdx, b.shape, bStrides);
                out.f()[flat] = op(a.f()[aIdx], b.f()[bIdx]);
            } });
    }

    template <class Arch>
    void evalGeluImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &x = values[node.inputs[0]];
        out = simdMakeFloatValue(node.shape);
        int64_t n = (int64_t)out.floatCount();
        parallelFor(0, n, kElementwiseGrain, [&](int64_t begin, int64_t end)
                    {
            int64_t i = begin;
            for (; i + width <= end; i += width)
            {
                auto v = Batch::load_unaligned(x.f() + i);
                (0.5f * v * (1.0f + xsimd::erf(v * 0.70710678118654752f))).store_unaligned(out.f() + i);
            }
            for (; i < end; i++)
            {
                float v = x.f()[i];
                out.f()[i] = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
            } });
    }

    template <class Arch>
    void evalReluImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &x = values[node.inputs[0]];
        out = simdMakeFloatValue(node.shape);
        int64_t n = (int64_t)out.floatCount();
        parallelFor(0, n, kElementwiseGrain, [&](int64_t begin, int64_t end)
                    {
            int64_t i = begin;
            for (; i + width <= end; i += width)
            {
                auto v = Batch::load_unaligned(x.f() + i);
                xsimd::max(v, Batch(0.0f)).store_unaligned(out.f() + i);
            }
            for (; i < end; i++)
                out.f()[i] = std::max(x.f()[i], 0.0f); });
    }

    template <class Arch>
    void evalSigmoidImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &x = values[node.inputs[0]];
        out = simdMakeFloatValue(node.shape);
        int64_t n = (int64_t)out.floatCount();
        parallelFor(0, n, kElementwiseGrain, [&](int64_t begin, int64_t end)
                    {
            int64_t i = begin;
            for (; i + width <= end; i += width)
            {
                auto v = Batch::load_unaligned(x.f() + i);
                (1.0f / (1.0f + xsimd::exp(-v))).store_unaligned(out.f() + i);
            }
            for (; i < end; i++)
                out.f()[i] = 1.0f / (1.0f + std::exp(-x.f()[i])); });
    }

    template <class Arch>
    void evalLayerNormImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &scale = values[node.inputs[1]];
        const CpuValue &bias = values[node.inputs[2]];
        out = simdMakeFloatValue(node.shape);
        int64_t lastDim = node.shape.back();
        int64_t outerTotal = numElements(node.shape) / lastDim;
        float eps = node.floatAttr0;

        parallelFor(0, outerTotal, kRowGrain, [&](int64_t obegin, int64_t oend)
                    {
            for (int64_t o = obegin; o < oend; o++)
            {
                const float *row = x.f() + o * lastDim;
                int64_t k = 0;
                Batch sumv(0.0f);
                for (; k + width <= lastDim; k += width)
                    sumv = sumv + Batch::load_unaligned(row + k);
                float mean = xsimd::reduce_add(sumv);
                for (; k < lastDim; k++)
                    mean += row[k];
                mean /= (float)lastDim;

                Batch meanv(mean);
                Batch varv(0.0f);
                for (k = 0; k + width <= lastDim; k += width)
                {
                    auto d = Batch::load_unaligned(row + k) - meanv;
                    varv = varv + d * d;
                }
                float var = xsimd::reduce_add(varv);
                for (; k < lastDim; k++)
                {
                    float d = row[k] - mean;
                    var += d * d;
                }
                var /= (float)lastDim;
                float invStd = 1.0f / std::sqrt(var + eps);

                float *outRow = out.f() + o * lastDim;
                Batch invStdv(invStd);
                for (k = 0; k + width <= lastDim; k += width)
                {
                    auto rv = Batch::load_unaligned(row + k);
                    auto sv = Batch::load_unaligned(scale.f() + k);
                    auto bv = Batch::load_unaligned(bias.f() + k);
                    ((rv - meanv) * invStdv * sv + bv).store_unaligned(outRow + k);
                }
                for (; k < lastDim; k++)
                    outRow[k] = (row[k] - mean) * invStd * scale.f()[k] + bias.f()[k];
            } });
    }

    template <class Arch>
    void evalRmsNormImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &scale = values[node.inputs[1]];
        out = simdMakeFloatValue(node.shape);
        int64_t lastDim = node.shape.back();
        int64_t outerTotal = numElements(node.shape) / lastDim;
        float eps = node.floatAttr0;

        parallelFor(0, outerTotal, kRowGrain, [&](int64_t obegin, int64_t oend)
                    {
            for (int64_t o = obegin; o < oend; o++)
            {
                const float *row = x.f() + o * lastDim;
                int64_t k = 0;
                Batch sqv(0.0f);
                for (; k + width <= lastDim; k += width)
                {
                    auto rv = Batch::load_unaligned(row + k);
                    sqv = sqv + rv * rv;
                }
                float meanSquare = xsimd::reduce_add(sqv);
                for (; k < lastDim; k++)
                    meanSquare += row[k] * row[k];
                meanSquare /= (float)lastDim;
                float invRms = 1.0f / std::sqrt(meanSquare + eps);

                float *outRow = out.f() + o * lastDim;
                Batch invRmsv(invRms);
                for (k = 0; k + width <= lastDim; k += width)
                {
                    auto rv = Batch::load_unaligned(row + k);
                    auto sv = Batch::load_unaligned(scale.f() + k);
                    (rv * invRmsv * sv).store_unaligned(outRow + k);
                }
                for (; k < lastDim; k++)
                    outRow[k] = row[k] * invRms * scale.f()[k];
            } });
    }

    template <class Arch>
    void evalBatchNormImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &mean = values[node.inputs[1]];
        const CpuValue &variance = values[node.inputs[2]];
        const CpuValue &scale = values[node.inputs[3]];
        const CpuValue &bias = values[node.inputs[4]];
        out = simdMakeFloatValue(node.shape);
        float eps = node.floatAttr0;

        int64_t N = node.shape[0], C = node.shape[1], H = node.shape[2], W = node.shape[3];
        std::vector<float> invStd(C);
        for (int64_t c = 0; c < C; c++)
            invStd[c] = 1.0f / std::sqrt(variance.f()[c] + eps);

        parallelFor(0, N * C, kNCUnitGrain, [&](int64_t begin, int64_t end)
                    {
            for (int64_t nc = begin; nc < end; nc++)
            {
                int64_t c = nc % C;
                const float *plane = x.f() + nc * H * W;
                float *outPlane = out.f() + nc * H * W;
                float m = mean.f()[c], s = scale.f()[c], b = bias.f()[c], inv = invStd[c];
                int64_t spatial = H * W;
                Batch mv(m), sv(s), bv(b), invv(inv);
                int64_t k = 0;
                for (; k + width <= spatial; k += width)
                {
                    auto pv = Batch::load_unaligned(plane + k);
                    ((pv - mv) * invv * sv + bv).store_unaligned(outPlane + k);
                }
                for (; k < spatial; k++)
                    outPlane[k] = (plane[k] - m) * inv * s + b;
            } });
    }

    template <class Arch>
    void evalInstanceNormImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &x = values[node.inputs[0]];
        const CpuValue &scale = values[node.inputs[1]];
        const CpuValue &bias = values[node.inputs[2]];
        out = simdMakeFloatValue(node.shape);
        float eps = node.floatAttr0;

        int64_t N = node.shape[0], C = node.shape[1], H = node.shape[2], W = node.shape[3];
        int64_t spatial = H * W;

        parallelFor(0, N * C, kNCUnitGrain, [&](int64_t begin, int64_t end)
                    {
            for (int64_t nc = begin; nc < end; nc++)
            {
                int64_t c = nc % C;
                const float *plane = x.f() + nc * spatial;
                int64_t k = 0;
                Batch sumv(0.0f);
                for (; k + width <= spatial; k += width)
                    sumv = sumv + Batch::load_unaligned(plane + k);
                float mean = xsimd::reduce_add(sumv);
                for (; k < spatial; k++)
                    mean += plane[k];
                mean /= (float)spatial;

                Batch meanv(mean);
                Batch varv(0.0f);
                for (k = 0; k + width <= spatial; k += width)
                {
                    auto d = Batch::load_unaligned(plane + k) - meanv;
                    varv = varv + d * d;
                }
                float var = xsimd::reduce_add(varv);
                for (; k < spatial; k++)
                {
                    float d = plane[k] - mean;
                    var += d * d;
                }
                var /= (float)spatial;
                float invStd = 1.0f / std::sqrt(var + eps);
                float s = scale.f()[c], b = bias.f()[c];
                float *outPlane = out.f() + nc * spatial;
                Batch invStdv(invStd), sv(s), bv(b);
                for (k = 0; k + width <= spatial; k += width)
                {
                    auto pv = Batch::load_unaligned(plane + k);
                    ((pv - meanv) * invStdv * sv + bv).store_unaligned(outPlane + k);
                }
                for (; k < spatial; k++)
                    outPlane[k] = (plane[k] - mean) * invStd * s + b;
            } });
    }

    template <class Arch>
    void evalMatMulImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &a = values[node.inputs[0]];
        const CpuValue &b = values[node.inputs[1]];
        out = simdMakeFloatValue(node.shape);

        size_t rank = a.shape.size();
        int64_t M = a.shape[rank - 2];
        int64_t K = a.shape[rank - 1];
        int64_t N = b.shape[rank - 1];
        int64_t batchCount = numElements(node.shape) / (M * N);

        parallelFor(0, batchCount * M, kMatmulRowGrain, [&](int64_t begin, int64_t end)
                    {
            for (int64_t idx = begin; idx < end; idx++)
            {
                int64_t batch = idx / M;
                int64_t m = idx % M;
                const float *aRow = a.f() + batch * M * K + m * K;
                const float *bBase = b.f() + batch * K * N;
                float *outRow = out.f() + batch * M * N + m * N;

                for (int64_t k = 0; k < K; k++)
                {
                    float aVal = aRow[k];
                    Batch aBatch(aVal);
                    const float *bRow = bBase + k * N;
                    int64_t n = 0;
                    if (k == 0)
                    {
                        for (; n + width <= N; n += width)
                            (aBatch * Batch::load_unaligned(bRow + n)).store_unaligned(outRow + n);
                        for (; n < N; n++)
                            outRow[n] = aVal * bRow[n];
                    }
                    else
                    {
                        for (; n + width <= N; n += width)
                        {
                            auto ov = Batch::load_unaligned(outRow + n);
                            xsimd::fma(aBatch, Batch::load_unaligned(bRow + n), ov).store_unaligned(outRow + n);
                        }
                        for (; n < N; n++)
                            outRow[n] += aVal * bRow[n];
                    }
                }
            } });
    }

    template <class Arch>
    void evalGemmImpl(const Node &node, std::vector<CpuValue> &values, CpuValue &out)
    {
        using Batch = xsimd::batch<float, Arch>;
        constexpr int64_t width = Batch::size;

        const CpuValue &a = values[node.inputs[0]];
        const CpuValue &b = values[node.inputs[1]];
        const CpuValue &c = values[node.inputs[2]];
        out = simdMakeFloatValue(node.shape);

        int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
        float alpha = node.floatAttr0, beta = node.floatAttr1;
        size_t cElems = c.floatCount();

        parallelFor(0, M, kMatmulRowGrain, [&](int64_t mbegin, int64_t mend)
                    {
            for (int64_t m = mbegin; m < mend; m++)
            {
                const float *aRow = a.f() + m * K;
                float *outRow = out.f() + m * N;

                for (int64_t k = 0; k < K; k++)
                {
                    float aVal = aRow[k];
                    Batch aBatch(aVal);
                    const float *bRow = b.f() + k * N;
                    int64_t n = 0;
                    if (k == 0)
                    {
                        for (; n + width <= N; n += width)
                            (aBatch * Batch::load_unaligned(bRow + n)).store_unaligned(outRow + n);
                        for (; n < N; n++)
                            outRow[n] = aVal * bRow[n];
                    }
                    else
                    {
                        for (; n + width <= N; n += width)
                        {
                            auto ov = Batch::load_unaligned(outRow + n);
                            xsimd::fma(aBatch, Batch::load_unaligned(bRow + n), ov).store_unaligned(outRow + n);
                        }
                        for (; n < N; n++)
                            outRow[n] += aVal * bRow[n];
                    }
                }

                const float *cRow = cElems == (size_t)(M * N) ? c.f() + m * N : nullptr;
                Batch alphaBatch(alpha), betaBatch(beta);
                int64_t n = 0;
                for (; n + width <= N; n += width)
                {
                    auto sumv = Batch::load_unaligned(outRow + n);
                    Batch cv = cElems == 1 ? Batch(c.f()[0])
                               : cElems == (size_t)N ? Batch::load_unaligned(c.f() + n)
                                                      : Batch::load_unaligned(cRow + n);
                    (alphaBatch * sumv + betaBatch * cv).store_unaligned(outRow + n);
                }
                for (; n < N; n++)
                {
                    float cv = cElems == 1 ? c.f()[0] : (cElems == (size_t)N ? c.f()[n] : c.f()[m * N + n]);
                    outRow[n] = alpha * outRow[n] + beta * cv;
                }
            } });
    }

#if !(defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#define CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED 1
#endif

#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED

    /// Cached (checked once, not per-call) runtime cpuid check for AVX2+FMA3.
    inline bool cpuSupportsAvx2Fma()
    {
        static const bool supported = xsimd::available_architectures().has(xsimd::fma3<xsimd::avx2> {});
        return supported;
    }

    // Definitions live in simd_kernels_avx2.cpp, compiled with -mavx2/-mfma
    // (or /arch:AVX2 on MSVC) — referencing xsimd::fma3<xsimd::avx2> codegen
    // from a TU not built with those flags would be invalid, hence `extern`.
    extern template void evalBroadcastBinaryOpImpl<xsimd::fma3<xsimd::avx2>, AddOp>(
        const Node &, std::vector<CpuValue> &, CpuValue &, AddOp);
    extern template void evalBroadcastBinaryOpImpl<xsimd::fma3<xsimd::avx2>, MulOp>(
        const Node &, std::vector<CpuValue> &, CpuValue &, MulOp);
    extern template void evalGeluImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalReluImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalSigmoidImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalLayerNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalRmsNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalBatchNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalInstanceNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalMatMulImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    extern template void evalGemmImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);

#else

    constexpr bool cpuSupportsAvx2Fma() { return false; }

#endif

} // namespace systems::leal::campello_nn

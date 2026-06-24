#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include "test_helpers.hpp"

// Distinct purpose from test_cpu_threading.cpp: that file's large shapes
// already incidentally exercise some of these vectorized code paths, but
// wasn't designed to target SIMD specifically. These tests use shapes
// deliberately *not* a multiple of the SIMD width (4 on this build's SSE2
// baseline, but also not a multiple of 8/16 for AVX2/AVX-512 builds elsewhere)
// to specifically exercise each op's scalar-remainder tail, plus gemm's three
// `c`-broadcast shapes, none of which test_cpu_threading.cpp covers.

TEST(CpuSimd, GeluOddSize)
{
    const int n = 59;
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {n}});
    auto graph = builder.build({{"out", builder.gelu(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {n}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {n}, true, false});

    std::vector<float> xv(n);
    for (int i = 0; i < n; i++)
        xv[i] = (float)(i - n / 2) * 0.1f;
    tx->write(xv.data(), xv.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(n);
    tout->read(result.data(), result.size() * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        float expected = 0.5f * xv[i] * (1.0f + std::erf(xv[i] * 0.70710678118654752f));
        EXPECT_NEAR(result[i], expected, 1e-4f) << "at index " << i;
    }
}

TEST(CpuSimd, SigmoidOddSize)
{
    const int n = 67;
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {n}});
    auto graph = builder.build({{"out", builder.sigmoid(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {n}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {n}, true, false});

    std::vector<float> xv(n);
    for (int i = 0; i < n; i++)
        xv[i] = (float)(i - n / 2) * 0.07f;
    tx->write(xv.data(), xv.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(n);
    tout->read(result.data(), result.size() * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        float expected = 1.0f / (1.0f + std::exp(-xv[i]));
        EXPECT_NEAR(result[i], expected, 1e-4f) << "at index " << i;
    }
}

TEST(CpuSimd, RmsNormOddLastDim)
{
    const int64_t rows = 5, lastDim = 59;
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {rows, lastDim}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {lastDim}});
    auto graph = builder.build({{"out", builder.rmsNorm(x, scale, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {rows, lastDim}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {lastDim}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {rows, lastDim}, true, false});

    std::vector<float> xv(rows * lastDim), scaleV(lastDim);
    for (int64_t o = 0; o < rows; o++)
        for (int64_t k = 0; k < lastDim; k++)
            xv[o * lastDim + k] = std::cos((float)o * 0.3f + (float)k * 0.07f);
    for (int64_t k = 0; k < lastDim; k++)
        scaleV[k] = 1.0f + (float)k * 0.005f;
    tx->write(xv.data(), xv.size() * sizeof(float));
    tscale->write(scaleV.data(), scaleV.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(rows * lastDim);
    tout->read(result.data(), result.size() * sizeof(float));

    for (int64_t o = 0; o < rows; o++)
    {
        double meanSquare = 0.0;
        for (int64_t k = 0; k < lastDim; k++)
            meanSquare += (double)xv[o * lastDim + k] * xv[o * lastDim + k];
        meanSquare /= (double)lastDim;
        double invRms = 1.0 / std::sqrt(meanSquare + 1e-5);
        for (int64_t k = 0; k < lastDim; k++)
        {
            double expected = xv[o * lastDim + k] * invRms * scaleV[k];
            EXPECT_NEAR(result[o * lastDim + k], (float)expected, 1e-4f) << "at row " << o << " k " << k;
        }
    }
}

TEST(CpuSimd, InstanceNormOddSpatialBatchOne)
{
    const int64_t N = 1, C = 3, H = 7, W = 9; // spatial = 63, not a multiple of 4 or 8
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {N, C, H, W}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {C}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {C}});
    auto graph = builder.build({{"out", builder.instanceNorm(x, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {N, C, H, W}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {C}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {C}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {N, C, H, W}, true, false});

    int64_t spatial = H * W;
    std::vector<float> xv(N * C * spatial), scaleV(C), biasV(C);
    for (size_t i = 0; i < xv.size(); i++)
        xv[i] = std::sin((float)i * 0.05f) * 10.0f;
    for (int64_t c = 0; c < C; c++)
    {
        scaleV[c] = 1.0f + (float)c * 0.1f;
        biasV[c] = (float)c * 0.2f;
    }
    tx->write(xv.data(), xv.size() * sizeof(float));
    tscale->write(scaleV.data(), scaleV.size() * sizeof(float));
    tbias->write(biasV.data(), biasV.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}, {"bias", tbias}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(N * C * spatial);
    tout->read(result.data(), result.size() * sizeof(float));

    for (int64_t c = 0; c < C; c++)
    {
        const float *plane = xv.data() + c * spatial;
        double mean = 0.0;
        for (int64_t k = 0; k < spatial; k++)
            mean += plane[k];
        mean /= (double)spatial;
        double var = 0.0;
        for (int64_t k = 0; k < spatial; k++)
        {
            double d = plane[k] - mean;
            var += d * d;
        }
        var /= (double)spatial;
        double invStd = 1.0 / std::sqrt(var + 1e-5);
        for (int64_t k = 0; k < spatial; k++)
        {
            double expected = (plane[k] - mean) * invStd * scaleV[c] + biasV[c];
            EXPECT_NEAR(result[c * spatial + k], (float)expected, 1e-3f) << "at c=" << c << " k=" << k;
        }
    }
}

TEST(CpuSimd, BatchNormOddSpatialBatchOne)
{
    const int64_t N = 1, C = 3, H = 7, W = 9; // spatial = 63, not a multiple of 4 or 8
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {N, C, H, W}});
    auto mean = builder.input("mean", {cnn::DataType::Float32, {C}});
    auto variance = builder.input("variance", {cnn::DataType::Float32, {C}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {C}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {C}});
    auto graph = builder.build({{"out", builder.batchNorm(x, mean, variance, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {N, C, H, W}, false, true});
    auto tmean = context->createTensor({cnn::DataType::Float32, {C}, false, true});
    auto tvariance = context->createTensor({cnn::DataType::Float32, {C}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {C}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {C}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {N, C, H, W}, true, false});

    int64_t spatial = H * W;
    std::vector<float> xv(N * C * spatial), meanV(C), varianceV(C), scaleV(C), biasV(C);
    for (size_t i = 0; i < xv.size(); i++)
        xv[i] = (float)(i % 17) - 8.0f;
    for (int64_t c = 0; c < C; c++)
    {
        meanV[c] = 1.0f + (float)c;
        varianceV[c] = 2.0f + (float)c * 0.5f;
        scaleV[c] = 1.0f + (float)c * 0.1f;
        biasV[c] = (float)c * 0.2f;
    }
    tx->write(xv.data(), xv.size() * sizeof(float));
    tmean->write(meanV.data(), meanV.size() * sizeof(float));
    tvariance->write(varianceV.data(), varianceV.size() * sizeof(float));
    tscale->write(scaleV.data(), scaleV.size() * sizeof(float));
    tbias->write(biasV.data(), biasV.size() * sizeof(float));

    auto fence = context->dispatch(
        *graph,
        {{"x", tx}, {"mean", tmean}, {"variance", tvariance}, {"scale", tscale}, {"bias", tbias}},
        {{"out", tout}});
    fence->wait();

    std::vector<float> result(N * C * spatial);
    tout->read(result.data(), result.size() * sizeof(float));

    for (int64_t c = 0; c < C; c++)
    {
        float invStd = 1.0f / std::sqrt(varianceV[c] + 1e-5f);
        for (int64_t k = 0; k < spatial; k++)
        {
            float expected = (xv[c * spatial + k] - meanV[c]) * invStd * scaleV[c] + biasV[c];
            EXPECT_FLOAT_EQ(result[c * spatial + k], expected) << "at c=" << c << " k=" << k;
        }
    }
}

TEST(CpuSimd, MatMulOddSizes)
{
    const int64_t M = 17, K = 23, N = 37; // none are multiples of 4 or 8
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {M, K}});
    auto b = builder.input("b", {cnn::DataType::Float32, {K, N}});
    auto graph = builder.build({{"out", builder.matmul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {M, K}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {K, N}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {M, N}, true, false});

    std::vector<float> av(M * K), bv(K * N);
    for (size_t i = 0; i < av.size(); i++)
        av[i] = (float)((int)(i % 11) - 5);
    for (size_t i = 0; i < bv.size(); i++)
        bv[i] = (float)((int)(i % 9) - 4);
    ta->write(av.data(), av.size() * sizeof(float));
    tb->write(bv.data(), bv.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(M * N);
    tout->read(result.data(), result.size() * sizeof(float));

    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++)
        {
            float sum = 0.f;
            for (int64_t k = 0; k < K; k++)
                sum += av[m * K + k] * bv[k * N + n];
            EXPECT_FLOAT_EQ(result[m * N + n], sum) << "at m=" << m << " n=" << n;
        }
}

TEST(CpuSimd, GemmAllCBroadcastShapesOddSizes)
{
    const int64_t M = 13, K = 19, N = 29; // none are multiples of 4 or 8
    const float alpha = 1.5f, beta = 0.5f;

    std::vector<float> av(M * K), bv(K * N);
    for (size_t i = 0; i < av.size(); i++)
        av[i] = (float)((int)(i % 7) - 3);
    for (size_t i = 0; i < bv.size(); i++)
        bv[i] = (float)((int)(i % 5) - 2);

    std::vector<float> cScalar = {2.5f};
    std::vector<float> cRow(N);
    for (int64_t n = 0; n < N; n++)
        cRow[n] = (float)n * 0.1f;
    std::vector<float> cMatrix(M * N);
    for (size_t i = 0; i < cMatrix.size(); i++)
        cMatrix[i] = (float)(i % 13) * 0.1f;

    std::vector<std::pair<const char *, std::vector<float> *>> cases = {
        {"scalar", &cScalar}, {"row", &cRow}, {"matrix", &cMatrix}};

    for (auto &[label, cPtr] : cases)
    {
        auto context = makeCpuContext();
        cnn::GraphBuilder builder(context);
        auto a = builder.input("a", {cnn::DataType::Float32, {M, K}});
        auto b = builder.input("b", {cnn::DataType::Float32, {K, N}});
        std::vector<int64_t> cShape = cPtr->size() == 1 ? std::vector<int64_t>{1, 1}
                                       : cPtr->size() == (size_t)N ? std::vector<int64_t>{1, N}
                                                                    : std::vector<int64_t>{M, N};
        auto c = builder.input("c", {cnn::DataType::Float32, cShape});
        auto graph = builder.build({{"out", builder.gemm(a, b, c, alpha, beta)}});

        auto ta = context->createTensor({cnn::DataType::Float32, {M, K}, false, true});
        auto tb = context->createTensor({cnn::DataType::Float32, {K, N}, false, true});
        auto tc = context->createTensor({cnn::DataType::Float32, cShape, false, true});
        auto tout = context->createTensor({cnn::DataType::Float32, {M, N}, true, false});

        ta->write(av.data(), av.size() * sizeof(float));
        tb->write(bv.data(), bv.size() * sizeof(float));
        tc->write(cPtr->data(), cPtr->size() * sizeof(float));

        auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}, {"c", tc}}, {{"out", tout}});
        fence->wait();

        std::vector<float> result(M * N);
        tout->read(result.data(), result.size() * sizeof(float));

        size_t cElems = cPtr->size();
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++)
            {
                float sum = 0.f;
                for (int64_t k = 0; k < K; k++)
                    sum += av[m * K + k] * bv[k * N + n];
                float cv = cElems == 1 ? (*cPtr)[0] : (cElems == (size_t)N ? (*cPtr)[n] : (*cPtr)[m * N + n]);
                float expected = alpha * sum + beta * cv;
                EXPECT_FLOAT_EQ(result[m * N + n], expected) << "case=" << label << " m=" << m << " n=" << n;
            }
    }
}

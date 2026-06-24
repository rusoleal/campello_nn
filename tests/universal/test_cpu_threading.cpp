#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include "test_helpers.hpp"

// These tests use shapes deliberately sized above each op's parallelFor grain
// threshold (see src/cpu/ops.cpp's kElementwiseGrain/kRowGrain/kMatmulRowGrain/
// kConvUnitGrain/kNCUnitGrain constants) — every other test in this suite uses
// small shapes that always take parallelFor's serial fast-path, so without
// these, the actual multi-threaded code path in src/cpu/thread_pool.cpp would
// never run during `ctest`. Expected values are computed independently in each
// test (a brute-force reference loop), not by re-deriving the production
// formula, to actually catch indexing mistakes in the parallel rewrite.

TEST(CpuThreading, AddLargeElementwise)
{
    const int n = 300000; // well above kElementwiseGrain (65536) * 2
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {n}});
    auto b = builder.input("b", {cnn::DataType::Float32, {n}});
    auto graph = builder.build({{"out", builder.add(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {n}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {n}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {n}, true, false});

    std::vector<float> av(n), bv(n), expected(n);
    for (int i = 0; i < n; i++)
    {
        av[i] = (float)(i % 997) * 0.01f;
        bv[i] = (float)((i * 7 + 3) % 991) * 0.02f;
        expected[i] = av[i] + bv[i];
    }
    ta->write(av.data(), av.size() * sizeof(float));
    tb->write(bv.data(), bv.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(n);
    tout->read(result.data(), result.size() * sizeof(float));
    for (int i = 0; i < n; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "at index " << i;
}

TEST(CpuThreading, MatMulLargeBatched)
{
    const int64_t batch = 4, M = 64, K = 64, N = 64; // batch*M = 256, well above kMatmulRowGrain (8) * 2
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {batch, M, K}});
    auto b = builder.input("b", {cnn::DataType::Float32, {batch, K, N}});
    auto graph = builder.build({{"out", builder.matmul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {batch, M, K}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {batch, K, N}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {batch, M, N}, true, false});

    std::vector<float> av(batch * M * K), bv(batch * K * N);
    for (size_t i = 0; i < av.size(); i++)
        av[i] = (float)((int)(i % 13) - 6);
    for (size_t i = 0; i < bv.size(); i++)
        bv[i] = (float)((int)(i % 7) - 3);
    ta->write(av.data(), av.size() * sizeof(float));
    tb->write(bv.data(), bv.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(batch * M * N);
    tout->read(result.data(), result.size() * sizeof(float));

    // Independent brute-force reference, same float accumulation order as the
    // production kernel (k innermost), so results should match exactly.
    for (int64_t bi = 0; bi < batch; bi++)
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++)
            {
                float sum = 0.f;
                for (int64_t k = 0; k < K; k++)
                    sum += av[bi * M * K + m * K + k] * bv[bi * K * N + k * N + n];
                EXPECT_FLOAT_EQ(result[bi * M * N + m * N + n], sum)
                    << "at batch=" << bi << " m=" << m << " n=" << n;
            }
}

TEST(CpuThreading, LayerNormManyRows)
{
    const int64_t rows = 256, lastDim = 32; // rows well above kRowGrain (64) * 2
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {rows, lastDim}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {lastDim}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {lastDim}});
    auto graph = builder.build({{"out", builder.layerNorm(x, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {rows, lastDim}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {lastDim}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {lastDim}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {rows, lastDim}, true, false});

    std::vector<float> xv(rows * lastDim), scaleV(lastDim), biasV(lastDim);
    for (int64_t o = 0; o < rows; o++)
        for (int64_t k = 0; k < lastDim; k++)
            xv[o * lastDim + k] = std::sin((float)o * 0.1f + (float)k * 0.05f);
    for (int64_t k = 0; k < lastDim; k++)
    {
        scaleV[k] = 1.0f + (float)k * 0.01f;
        biasV[k] = (float)k * 0.02f;
    }
    tx->write(xv.data(), xv.size() * sizeof(float));
    tscale->write(scaleV.data(), scaleV.size() * sizeof(float));
    tbias->write(biasV.data(), biasV.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}, {"bias", tbias}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(rows * lastDim);
    tout->read(result.data(), result.size() * sizeof(float));

    for (int64_t o = 0; o < rows; o++)
    {
        double mean = 0.0;
        for (int64_t k = 0; k < lastDim; k++)
            mean += xv[o * lastDim + k];
        mean /= (double)lastDim;
        double var = 0.0;
        for (int64_t k = 0; k < lastDim; k++)
        {
            double d = xv[o * lastDim + k] - mean;
            var += d * d;
        }
        var /= (double)lastDim;
        double invStd = 1.0 / std::sqrt(var + 1e-5);
        for (int64_t k = 0; k < lastDim; k++)
        {
            double expected = (xv[o * lastDim + k] - mean) * invStd * scaleV[k] + biasV[k];
            EXPECT_NEAR(result[o * lastDim + k], (float)expected, 1e-3f) << "at row " << o << " k " << k;
        }
    }
}

TEST(CpuThreading, Conv2dManyOutputChannelsBatchOne)
{
    // N=1 deliberately: this is the common inference case, and the case that
    // would get zero parallelism if the outer loop were wrapped without first
    // flattening N*O (see ops.cpp's evalConv2d).
    const int64_t N = 1, C = 2, H = 9, W = 9, O = 16, KH = 3, KW = 3;
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {N, C, H, W}});
    auto w = builder.input("w", {cnn::DataType::Float32, {O, C, KH, KW}});
    auto graph = builder.build({{"out", builder.conv2d(x, w, cnn::Conv2dDescriptor{})}});

    int64_t outH = H - KH + 1, outW = W - KW + 1;
    auto tx = context->createTensor({cnn::DataType::Float32, {N, C, H, W}, false, true});
    auto tw = context->createTensor({cnn::DataType::Float32, {O, C, KH, KW}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {N, O, outH, outW}, true, false});

    std::vector<float> xv(N * C * H * W), wv(O * C * KH * KW);
    for (size_t i = 0; i < xv.size(); i++)
        xv[i] = (float)((int)(i % 11) - 5) * 0.1f;
    for (size_t i = 0; i < wv.size(); i++)
        wv[i] = (float)((int)(i % 5) - 2) * 0.2f;
    tx->write(xv.data(), xv.size() * sizeof(float));
    tw->write(wv.data(), wv.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"w", tw}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(N * O * outH * outW);
    tout->read(result.data(), result.size() * sizeof(float));

    for (int64_t o = 0; o < O; o++)
        for (int64_t oh = 0; oh < outH; oh++)
            for (int64_t ow = 0; ow < outW; ow++)
            {
                float sum = 0.f;
                for (int64_t c = 0; c < C; c++)
                    for (int64_t kh = 0; kh < KH; kh++)
                        for (int64_t kw = 0; kw < KW; kw++)
                            sum += xv[(c * H + (oh + kh)) * W + (ow + kw)] *
                                   wv[((o * C + c) * KH + kh) * KW + kw];
                EXPECT_FLOAT_EQ(result[(o * outH + oh) * outW + ow], sum)
                    << "at o=" << o << " oh=" << oh << " ow=" << ow;
            }
}

TEST(CpuThreading, AvgPool2dManyChannelsBatchOne)
{
    // N=1 deliberately, same rationale as the Conv2d test above (flattened
    // N*C, not just the outer N loop).
    const int64_t N = 1, C = 32, H = 8, W = 8;
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {N, C, H, W}});
    cnn::Pool2dDescriptor desc;
    desc.kernelHeight = 2;
    desc.kernelWidth = 2;
    desc.strideX = 2;
    desc.strideY = 2;
    auto graph = builder.build({{"out", builder.avgPool2d(x, desc)}});

    int64_t outH = H / 2, outW = W / 2;
    auto tx = context->createTensor({cnn::DataType::Float32, {N, C, H, W}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {N, C, outH, outW}, true, false});

    std::vector<float> xv(N * C * H * W);
    for (size_t i = 0; i < xv.size(); i++)
        xv[i] = (float)(i % 23);
    tx->write(xv.data(), xv.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(N * C * outH * outW);
    tout->read(result.data(), result.size() * sizeof(float));

    for (int64_t c = 0; c < C; c++)
        for (int64_t oh = 0; oh < outH; oh++)
            for (int64_t ow = 0; ow < outW; ow++)
            {
                float sum = 0.f;
                for (int64_t kh = 0; kh < 2; kh++)
                    for (int64_t kw = 0; kw < 2; kw++)
                        sum += xv[(c * H + (oh * 2 + kh)) * W + (ow * 2 + kw)];
                EXPECT_FLOAT_EQ(result[(c * outH + oh) * outW + ow], sum / 4.0f)
                    << "at c=" << c << " oh=" << oh << " ow=" << ow;
            }
}

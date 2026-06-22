#include <cmath>
#include <gtest/gtest.h>
#include "../universal/test_helpers.hpp"

// Mirrors tests/universal/test_cpu_ops.cpp, but runs against the MPSGraph
// backend (DeviceType::Gpu) to check the IR-to-MPSGraph op mapping is correct,
// not just that it compiles.

TEST(MpsOps, Add)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 2}});
    auto graph = builder.build({{"out", builder.add(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float av[4] = {1, 2, 3, 4};
    float bv[4] = {10, 20, 30, 40};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], av[i] + bv[i]);
}

TEST(MpsOps, Mul)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3}});
    auto graph = builder.build({{"out", builder.mul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3}, true, false});

    float av[3] = {2, 3, 4};
    float bv[3] = {5, 6, 7};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[3];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(result[i], av[i] * bv[i]);
}

TEST(MpsOps, AddBroadcastRowVector)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3}});
    auto graph = builder.build({{"out", builder.add(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 3}, true, false});

    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[3] = {10, 20, 30};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {11, 22, 33, 14, 25, 36};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, MulBroadcastColumnVector)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 1}});
    auto graph = builder.build({{"out", builder.mul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 1}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 3}, true, false});

    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[2] = {2, 3};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {2, 4, 6, 12, 15, 18};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, Gelu)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.gelu(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4}, true, false});

    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
    {
        float expected = 0.5f * xv[i] * (1.0f + std::erf(xv[i] * 0.70710678118654752f));
        EXPECT_NEAR(result[i], expected, 1e-3f);
    }
}

TEST(MpsOps, Relu)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.relu(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4}, true, false});

    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    float expected[4] = {0, 0, 0.5f, 2.f};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, Sigmoid)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.sigmoid(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4}, true, false});

    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
    {
        float expected = 1.0f / (1.0f + std::exp(-xv[i]));
        EXPECT_NEAR(result[i], expected, 1e-3f);
    }
}

TEST(MpsOps, Softmax)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 3}});
    auto graph = builder.build({{"out", builder.softmax(x, -1)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 3}, true, false});

    float xv[6] = {1, 2, 3, 1, 1, 1};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));

    for (int row = 0; row < 2; row++)
    {
        float sum = result[row * 3 + 0] + result[row * 3 + 1] + result[row * 3 + 2];
        EXPECT_NEAR(sum, 1.0f, 1e-4f);
    }
    EXPECT_NEAR(result[3], result[4], 1e-4f);
    EXPECT_NEAR(result[4], result[5], 1e-4f);
    EXPECT_LT(result[0], result[1]);
    EXPECT_LT(result[1], result[2]);
}

TEST(MpsOps, LayerNorm)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {4}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.layerNorm(x, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});

    float xv[4] = {1, 2, 3, 4};
    float scaleV[4] = {1, 1, 1, 1};
    float biasV[4] = {0, 0, 0, 0};
    tx->write(xv, sizeof(xv));
    tscale->write(scaleV, sizeof(scaleV));
    tbias->write(biasV, sizeof(biasV));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}, {"bias", tbias}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));

    float mean = 2.5f;
    float var = 1.25f;
    float invStd = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(result[i], (xv[i] - mean) * invStd, 1e-3f);
}

TEST(MpsOps, RmsNorm)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.rmsNorm(x, scale, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});

    float xv[4] = {1, 2, 3, 4};
    float scaleV[4] = {2, 1, 0.5f, 1};
    tx->write(xv, sizeof(xv));
    tscale->write(scaleV, sizeof(scaleV));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));

    float meanSquare = (1.f * 1.f + 2.f * 2.f + 3.f * 3.f + 4.f * 4.f) / 4.f;
    float invRms = 1.0f / std::sqrt(meanSquare + 1e-5f);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(result[i], xv[i] * invRms * scaleV[i], 1e-3f);
}

TEST(MpsOps, RotaryEmbedding)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto cos = builder.input("cos", {cnn::DataType::Float32, {4}});
    auto sin = builder.input("sin", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.rotaryEmbedding(x, cos, sin)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4}, false, true});
    auto tcos = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tsin = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});

    float xv[4] = {1, 2, 3, 4};
    float cosV[4] = {0.6f, 0.6f, 0.6f, 0.6f};
    float sinV[4] = {0.8f, 0.8f, 0.8f, 0.8f};
    tx->write(xv, sizeof(xv));
    tcos->write(cosV, sizeof(cosV));
    tsin->write(sinV, sizeof(sinV));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"cos", tcos}, {"sin", tsin}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));

    float expected[4] = {1 * 0.6f + -3 * 0.8f, 2 * 0.6f + -4 * 0.8f, 3 * 0.6f + 1 * 0.8f, 4 * 0.6f + 2 * 0.8f};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(result[i], expected[i], 1e-3f);
}

TEST(MpsOps, BatchNorm)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 2, 2, 2}});
    auto mean = builder.input("mean", {cnn::DataType::Float32, {2}});
    auto variance = builder.input("variance", {cnn::DataType::Float32, {2}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {2}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {2}});
    auto graph = builder.build({{"out", builder.batchNorm(x, mean, variance, scale, bias, 0.0f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, false, true});
    auto tmean = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tvariance = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, true, false});

    float xv[8] = {1, 2, 3, 4, 10, 20, 30, 40};
    float meanV[2] = {2.5f, 25.f};
    float varianceV[2] = {0.25f, 0.25f};
    float scaleV[2] = {2.f, 0.5f};
    float biasV[2] = {1.f, -1.f};
    tx->write(xv, sizeof(xv));
    tmean->write(meanV, sizeof(meanV));
    tvariance->write(varianceV, sizeof(varianceV));
    tscale->write(scaleV, sizeof(scaleV));
    tbias->write(biasV, sizeof(biasV));

    auto fence = context->dispatch(
        *graph,
        {{"x", tx}, {"mean", tmean}, {"variance", tvariance}, {"scale", tscale}, {"bias", tbias}},
        {{"out", tout}});
    fence->wait();

    float result[8];
    tout->read(result, sizeof(result));
    float expected[8] = {-5, -1, 3, 7, -16, -6, 4, 14};
    for (int i = 0; i < 8; i++)
        EXPECT_NEAR(result[i], expected[i], 1e-3f);
}

TEST(MpsOps, InstanceNorm)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 2, 2, 2}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {2}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {2}});
    auto graph = builder.build({{"out", builder.instanceNorm(x, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, true, false});

    float xv[8] = {1, 2, 3, 4, 10, 20, 30, 40};
    float scaleV[2] = {1, 1};
    float biasV[2] = {0, 0};
    tx->write(xv, sizeof(xv));
    tscale->write(scaleV, sizeof(scaleV));
    tbias->write(biasV, sizeof(biasV));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}, {"bias", tbias}}, {{"out", tout}});
    fence->wait();

    float result[8];
    tout->read(result, sizeof(result));
    float mean0 = 2.5f, var0 = 1.25f, invStd0 = 1.0f / std::sqrt(var0 + 1e-5f);
    float mean1 = 25.f, var1 = 125.f, invStd1 = 1.0f / std::sqrt(var1 + 1e-5f);
    float expected[8];
    for (int i = 0; i < 4; i++)
        expected[i] = (xv[i] - mean0) * invStd0;
    for (int i = 4; i < 8; i++)
        expected[i] = (xv[i] - mean1) * invStd1;
    for (int i = 0; i < 8; i++)
        EXPECT_NEAR(result[i], expected[i], 1e-3f);
}

TEST(MpsOps, MatMul)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3, 2}});
    auto graph = builder.build({{"out", builder.matmul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[6] = {7, 8, 9, 10, 11, 12};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    EXPECT_FLOAT_EQ(result[0], 58);
    EXPECT_FLOAT_EQ(result[1], 64);
    EXPECT_FLOAT_EQ(result[2], 139);
    EXPECT_FLOAT_EQ(result[3], 154);
}

TEST(MpsOps, Gemm)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 2}});
    auto c = builder.input("c", {cnn::DataType::Float32, {2, 2}});
    auto graph = builder.build({{"out", builder.gemm(a, b, c, 2.0f, 0.5f)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tc = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float av[4] = {1, 2, 3, 4};
    float bv[4] = {1, 0, 0, 1};
    float cv[4] = {10, 10, 10, 10};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));
    tc->write(cv, sizeof(cv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}, {"c", tc}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], 2.0f * av[i] + 0.5f * cv[i]);
}

TEST(MpsOps, Reshape)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 3}});
    auto graph = builder.build({{"out", builder.reshape(x, {3, 2})}});

    auto tx = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3, 2}, true, false});

    float xv[6] = {1, 2, 3, 4, 5, 6};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], xv[i]);
}

TEST(MpsOps, Transpose)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 3}});
    auto graph = builder.build({{"out", builder.transpose(x, {1, 0})}});

    auto tx = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3, 2}, true, false});

    float xv[6] = {1, 2, 3, 4, 5, 6};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, Concat)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {1, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {1, 3}});
    auto graph = builder.build({{"out", builder.concat({a, b}, 1)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {1, 2}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {1, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 5}, true, false});

    float av[2] = {1, 2};
    float bv[3] = {3, 4, 5};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[5];
    tout->read(result, sizeof(result));
    float expected[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, Slice)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.slice(x, {1}, {2})}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2}, true, false});

    float xv[4] = {10, 20, 30, 40};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[2];
    tout->read(result, sizeof(result));
    EXPECT_FLOAT_EQ(result[0], 20);
    EXPECT_FLOAT_EQ(result[1], 30);
}

TEST(MpsOps, Conv2d)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 3, 3}});
    auto w = builder.input("w", {cnn::DataType::Float32, {1, 1, 2, 2}});
    auto graph = builder.build({{"out", builder.conv2d(x, w, cnn::Conv2dDescriptor{})}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, false, true});
    auto tw = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, true, false});

    float xv[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float wv[4] = {1, 0, 0, 1};
    tx->write(xv, sizeof(xv));
    tw->write(wv, sizeof(wv));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"w", tw}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    float expected[4] = {6, 8, 12, 14};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, MaxPool2d)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 4, 4}});
    cnn::Pool2dDescriptor desc;
    desc.kernelHeight = 2;
    desc.kernelWidth = 2;
    desc.strideX = 2;
    desc.strideY = 2;
    auto graph = builder.build({{"out", builder.maxPool2d(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 4, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, true, false});

    float xv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    float expected[4] = {6, 8, 14, 16};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, AvgPool2d)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 4, 4}});
    cnn::Pool2dDescriptor desc;
    desc.kernelHeight = 2;
    desc.kernelWidth = 2;
    desc.strideX = 2;
    desc.strideY = 2;
    auto graph = builder.build({{"out", builder.avgPool2d(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 4, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, true, false});

    float xv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    float expected[4] = {3.5f, 5.5f, 11.5f, 13.5f};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, ResizeBilinearAlignCorners)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 2, 2}});
    cnn::ResizeDescriptor desc;
    desc.outputHeight = 3;
    desc.outputWidth = 3;
    desc.mode = cnn::ResizeMode::Bilinear;
    desc.alignCorners = true;
    auto graph = builder.build({{"out", builder.resize(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, true, false});

    float xv[4] = {1, 2, 3, 4};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[9];
    tout->read(result, sizeof(result));
    float expected[9] = {1, 1.5f, 2, 2, 2.5f, 3, 3, 3.5f, 4};
    for (int i = 0; i < 9; i++)
        EXPECT_NEAR(result[i], expected[i], 1e-3f);
}

TEST(MpsOps, ResizeNearestAlignCorners)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 2, 2}});
    cnn::ResizeDescriptor desc;
    desc.outputHeight = 4;
    desc.outputWidth = 4;
    desc.mode = cnn::ResizeMode::Nearest;
    desc.alignCorners = true;
    auto graph = builder.build({{"out", builder.resize(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 4, 4}, true, false});

    float xv[4] = {1, 2, 3, 4};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[16];
    tout->read(result, sizeof(result));
    float expected[16] = {
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4};
    for (int i = 0; i < 16; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, ResizeNearestFloorAsymmetric)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 2, 2}});
    cnn::ResizeDescriptor desc;
    desc.outputHeight = 3;
    desc.outputWidth = 3;
    desc.mode = cnn::ResizeMode::Nearest;
    desc.centerResult = false;
    desc.alignCorners = false;
    desc.nearestRoundsDown = true;
    auto graph = builder.build({{"out", builder.resize(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, true, false});

    float xv[4] = {1, 2, 3, 4};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[9];
    tout->read(result, sizeof(result));
    float expected[9] = {1, 1, 2, 1, 1, 2, 3, 3, 4};
    for (int i = 0; i < 9; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, Gather)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto data = builder.input("data", {cnn::DataType::Float32, {4, 2}});
    auto indices = builder.input("indices", {cnn::DataType::Int32, {3}});
    auto graph = builder.build({{"out", builder.gather(data, indices, 0)}});

    auto tdata = context->createTensor({cnn::DataType::Float32, {4, 2}, false, true});
    auto tindices = context->createTensor({cnn::DataType::Int32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3, 2}, true, false});

    float dataV[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    int32_t idxV[3] = {2, 0, 3};
    tdata->write(dataV, sizeof(dataV));
    tindices->write(idxV, sizeof(idxV));

    auto fence = context->dispatch(*graph, {{"data", tdata}, {"indices", tindices}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {2, 2, 0, 0, 3, 3};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(MpsOps, GatherUint32Indices)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto data = builder.input("data", {cnn::DataType::Float32, {4, 2}});
    auto indices = builder.input("indices", {cnn::DataType::Uint32, {3}});
    auto graph = builder.build({{"out", builder.gather(data, indices, 0)}});

    auto tdata = context->createTensor({cnn::DataType::Float32, {4, 2}, false, true});
    auto tindices = context->createTensor({cnn::DataType::Uint32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3, 2}, true, false});

    float dataV[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    uint32_t idxV[3] = {2, 0, 3};
    tdata->write(dataV, sizeof(dataV));
    tindices->write(idxV, sizeof(idxV));

    auto fence = context->dispatch(*graph, {{"data", tdata}, {"indices", tindices}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {2, 2, 0, 0, 3, 3};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

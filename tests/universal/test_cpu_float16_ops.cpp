#include <cmath>
#include <gtest/gtest.h>
#include "test_helpers.hpp"

// Float16 coverage across the elementwise/activation/linalg/normalization op
// categories — the CPU backend decodes Float16 to Float32 at the Input/Constant
// boundary and re-encodes only at the final output, so these tests exercise that
// boundary-conversion path, not per-op special-casing (there is none).

TEST(CpuFloat16Ops, Add)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float16, {4}});
    auto b = builder.input("b", {cnn::DataType::Float16, {4}});
    auto graph = builder.build({{"out", builder.add(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float16, {4}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float16, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float16, {4}, true, false});

    auto av = toHalf({1, 2, 3, 4});
    auto bv = toHalf({10, 20, 30, 40});
    ta->write(av.data(), av.size() * sizeof(uint16_t));
    tb->write(bv.data(), bv.size() * sizeof(uint16_t));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    std::vector<uint16_t> result(4);
    tout->read(result.data(), result.size() * sizeof(uint16_t));
    auto resultF = fromHalf(result);
    float expected[4] = {11, 22, 33, 44};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(resultF[i], expected[i], 1e-3f);
}

TEST(CpuFloat16Ops, Mul)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float16, {3}});
    auto b = builder.input("b", {cnn::DataType::Float16, {3}});
    auto graph = builder.build({{"out", builder.mul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float16, {3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float16, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float16, {3}, true, false});

    auto av = toHalf({2, 3, 4});
    auto bv = toHalf({5, 6, 7});
    ta->write(av.data(), av.size() * sizeof(uint16_t));
    tb->write(bv.data(), bv.size() * sizeof(uint16_t));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    std::vector<uint16_t> result(3);
    tout->read(result.data(), result.size() * sizeof(uint16_t));
    auto resultF = fromHalf(result);
    float expected[3] = {10, 18, 28};
    for (int i = 0; i < 3; i++)
        EXPECT_NEAR(resultF[i], expected[i], 1e-3f);
}

TEST(CpuFloat16Ops, Gelu)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float16, {4}});
    auto graph = builder.build({{"out", builder.gelu(x)}});

    auto tx = context->createTensor({cnn::DataType::Float16, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float16, {4}, true, false});

    std::vector<float> xv = {-2.f, -0.5f, 0.5f, 2.f};
    auto xh = toHalf(xv);
    tx->write(xh.data(), xh.size() * sizeof(uint16_t));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    std::vector<uint16_t> result(4);
    tout->read(result.data(), result.size() * sizeof(uint16_t));
    auto resultF = fromHalf(result);
    for (int i = 0; i < 4; i++)
    {
        float expected = 0.5f * xv[i] * (1.0f + std::erf(xv[i] * 0.70710678118654752f));
        EXPECT_NEAR(resultF[i], expected, 5e-2f);
    }
}

TEST(CpuFloat16Ops, MatMul)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float16, {2, 2}});
    auto b = builder.input("b", {cnn::DataType::Float16, {2, 2}});
    auto graph = builder.build({{"out", builder.matmul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float16, {2, 2}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float16, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float16, {2, 2}, true, false});

    auto av = toHalf({1, 2, 3, 4});
    auto bv = toHalf({5, 6, 7, 8});
    ta->write(av.data(), av.size() * sizeof(uint16_t));
    tb->write(bv.data(), bv.size() * sizeof(uint16_t));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    std::vector<uint16_t> result(4);
    tout->read(result.data(), result.size() * sizeof(uint16_t));
    auto resultF = fromHalf(result);
    // [1,2;3,4] @ [5,6;7,8] = [19,22;43,50]
    float expected[4] = {19, 22, 43, 50};
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(resultF[i], expected[i], 1e-2f);
}

TEST(CpuFloat16Ops, LayerNorm)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float16, {1, 4}});
    auto scale = builder.input("scale", {cnn::DataType::Float16, {4}});
    auto bias = builder.input("bias", {cnn::DataType::Float16, {4}});
    auto graph = builder.build({{"out", builder.layerNorm(x, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float16, {1, 4}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float16, {4}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float16, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float16, {1, 4}, true, false});

    std::vector<float> xv = {1, 2, 3, 4};
    auto xh = toHalf(xv);
    auto scaleH = toHalf({1, 1, 1, 1});
    auto biasH = toHalf({0, 0, 0, 0});
    tx->write(xh.data(), xh.size() * sizeof(uint16_t));
    tscale->write(scaleH.data(), scaleH.size() * sizeof(uint16_t));
    tbias->write(biasH.data(), biasH.size() * sizeof(uint16_t));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}, {"bias", tbias}}, {{"out", tout}});
    fence->wait();

    std::vector<uint16_t> result(4);
    tout->read(result.data(), result.size() * sizeof(uint16_t));
    auto resultF = fromHalf(result);

    float mean = 2.5f, var = 1.25f;
    float invStd = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(resultF[i], (xv[i] - mean) * invStd, 5e-2f);
}

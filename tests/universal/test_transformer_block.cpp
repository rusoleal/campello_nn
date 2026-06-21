#include <cmath>
#include <gtest/gtest.h>
#include "test_helpers.hpp"

// End-to-end smoke test for the building blocks of a transformer layer:
// a linear projection (matmul + bias add), a GELU activation, and a
// LayerNorm — composed into one graph and dispatched once.
TEST(EndToEnd, LinearGeluLayerNormBlock)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);

    auto x = builder.input("x", {cnn::DataType::Float32, {1, 2}});
    auto w = builder.input("w", {cnn::DataType::Float32, {2, 2}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {1, 2}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {2}});
    auto lnBias = builder.input("lnBias", {cnn::DataType::Float32, {2}});

    auto linear = builder.add(builder.matmul(x, w), bias);
    auto activated = builder.gelu(linear);
    auto out = builder.layerNorm(activated, scale, lnBias, 1e-5f);
    auto graph = builder.build({{"out", out}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 2}, false, true});
    auto tw = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {1, 2}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tlnBias = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 2}, true, false});

    float xv[2] = {1, 2};
    float wv[4] = {1, 0, 0, 1}; // identity
    float biasV[2] = {0.1f, 0.2f};
    float scaleV[2] = {1, 1};
    float lnBiasV[2] = {0, 0};

    tx->write(xv, sizeof(xv));
    tw->write(wv, sizeof(wv));
    tbias->write(biasV, sizeof(biasV));
    tscale->write(scaleV, sizeof(scaleV));
    tlnBias->write(lnBiasV, sizeof(lnBiasV));

    auto fence = context->dispatch(
        *graph,
        {{"x", tx}, {"w", tw}, {"bias", tbias}, {"scale", tscale}, {"lnBias", tlnBias}},
        {{"out", tout}});
    fence->wait();

    float result[2];
    tout->read(result, sizeof(result));

    // Independently compute the expected result: linear -> gelu -> layerNorm.
    float linearV[2] = {
        xv[0] * wv[0] + xv[1] * wv[2] + biasV[0],
        xv[0] * wv[1] + xv[1] * wv[3] + biasV[1]};
    float geluV[2];
    for (int i = 0; i < 2; i++)
        geluV[i] = 0.5f * linearV[i] * (1.0f + std::erf(linearV[i] * 0.70710678118654752f));
    float mean = (geluV[0] + geluV[1]) / 2.0f;
    float var = ((geluV[0] - mean) * (geluV[0] - mean) + (geluV[1] - mean) * (geluV[1] - mean)) / 2.0f;
    float invStd = 1.0f / std::sqrt(var + 1e-5f);
    float expected[2] = {(geluV[0] - mean) * invStd, (geluV[1] - mean) * invStd};

    EXPECT_NEAR(result[0], expected[0], 1e-4f);
    EXPECT_NEAR(result[1], expected[1], 1e-4f);
}

#include <gtest/gtest.h>
#include "../universal/test_helpers.hpp"

TEST(MpsQuantizationOps, QuantizeLinear)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {5}});
    auto graph = builder.build({{"out", builder.quantizeLinear(x, 0.1f, 10)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {5}, false, true});
    auto tout = context->createTensor({cnn::DataType::Int8, {5}, true, false});

    float xv[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    int8_t result[5];
    tout->read(result, sizeof(result));
    int8_t expected[5] = {-10, 0, 10, 20, 30};
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(result[i], expected[i]);
}

TEST(MpsQuantizationOps, DequantizeLinear)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Int8, {5}});
    auto graph = builder.build({{"out", builder.dequantizeLinear(x, 0.1f, 10)}});

    auto tx = context->createTensor({cnn::DataType::Int8, {5}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {5}, true, false});

    int8_t xv[5] = {-10, 0, 10, 20, 30};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[5];
    tout->read(result, sizeof(result));
    float expected[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 0; i < 5; i++)
        EXPECT_NEAR(result[i], expected[i], 1e-3f);
}

TEST(MpsQuantizationOps, QuantizedMatmul)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto activation = builder.input("activation", {cnn::DataType::Float32, {2, 2}});
    auto weight = builder.input("weight", {cnn::DataType::Int8, {2, 2}});
    auto graph = builder.build({{"out", builder.quantizedMatmul(activation, weight, 1.0f, 0)}});

    auto tactivation = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tweight = context->createTensor({cnn::DataType::Int8, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float activationV[4] = {1, 2, 3, 4};
    int8_t weightV[4] = {1, 0, 0, 1};
    tactivation->write(activationV, sizeof(activationV));
    tweight->write(weightV, sizeof(weightV));

    auto fence = context->dispatch(*graph, {{"activation", tactivation}, {"weight", tweight}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(result[i], activationV[i], 1e-3f);
}

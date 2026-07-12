#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include "test_helpers.hpp"

TEST(CpuQuantizationOps, QuantizeLinear)
{
    auto context = makeCpuContext();
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
    // q = round(x/scale) + zeroPoint
    int8_t expected[5] = {-10, 0, 10, 20, 30};
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(result[i], expected[i]);
}

TEST(CpuQuantizationOps, DequantizeLinear)
{
    auto context = makeCpuContext();
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
    // out = scale * (q - zeroPoint)
    float expected[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 0; i < 5; i++)
        EXPECT_NEAR(result[i], expected[i], 1e-5f);
}

TEST(CpuQuantizationOps, QuantizedMatmul)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto activation = builder.input("activation", {cnn::DataType::Float32, {2, 2}});
    auto weight = builder.input("weight", {cnn::DataType::Int8, {2, 2}});
    auto graph = builder.build({{"out", builder.quantizedMatmul(activation, weight, 1.0f, 0)}});

    auto tactivation = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tweight = context->createTensor({cnn::DataType::Int8, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float activationV[4] = {1, 2, 3, 4};
    // scale=1, zeroPoint=0 -> dequantized weight equals these raw int8 values exactly:
    // the 2x2 identity matrix, so quantizedMatmul(activation, weight) == activation.
    int8_t weightV[4] = {1, 0, 0, 1};
    tactivation->write(activationV, sizeof(activationV));
    tweight->write(weightV, sizeof(weightV));

    auto fence = context->dispatch(*graph, {{"activation", tactivation}, {"weight", tweight}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], activationV[i]);
}

TEST(CpuQuantizationOps, GgmlQuantizedMatmul_Q8_0)
{
    // Build a Q8_0 quantized weight for a [outFeatures=4, inFeatures=8] linear
    // layer. Q8_0 packs 32 elements per block, so one block covers the whole
    // matrix. The bytes are: fp16 scale (2 bytes) + 32 int8 quant values.
    constexpr int64_t kOutFeatures = 4;
    constexpr int64_t kInFeatures = 8;
    constexpr size_t kBlockElements = 32;
    std::vector<float> weightGguf(static_cast<size_t>(kOutFeatures * kInFeatures));
    for (size_t i = 0; i < weightGguf.size(); ++i)
        weightGguf[i] = static_cast<float>(static_cast<int>(i % 7) - 3); // small pattern in [-3, 3]

    float maxAbs = 0.f;
    for (float v : weightGguf)
        maxAbs = std::max(maxAbs, std::abs(v));
    float scale = maxAbs / 127.0f;

    std::vector<uint8_t> weightBytes;
    uint16_t scaleBits = cnn::encodeFloat16(scale);
    float decodedScale = cnn::decodeFloat16(scaleBits);
    weightBytes.insert(weightBytes.end(), reinterpret_cast<const uint8_t *>(&scaleBits),
                       reinterpret_cast<const uint8_t *>(&scaleBits) + sizeof(scaleBits));
    std::vector<int8_t> quantValues(weightGguf.size());
    for (size_t i = 0; i < weightGguf.size(); ++i)
    {
        int q = static_cast<int>(std::round(weightGguf[i] / decodedScale));
        q = std::clamp(q, -127, 127);
        quantValues[i] = static_cast<int8_t>(q);
        weightBytes.push_back(static_cast<uint8_t>(quantValues[i]));
    }

    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto activation = builder.input("activation", {cnn::DataType::Float32, {2, kInFeatures}});
    auto weightConstant = builder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightBytes.size())}, false, false},
                                           weightBytes.data(), weightBytes.size());
    auto graph = builder.build({{"out", builder.ggmlQuantizedMatmul(activation, weightConstant, 8,
                                                                      {kInFeatures, kOutFeatures})}});

    auto tactivation = context->createTensor({cnn::DataType::Float32, {2, kInFeatures}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, kOutFeatures}, true, false});

    std::vector<float> activationV(static_cast<size_t>(2 * kInFeatures));
    for (size_t i = 0; i < activationV.size(); ++i)
        activationV[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
    tactivation->write(activationV.data(), activationV.size() * sizeof(float));

    auto fence = context->dispatch(*graph, {{"activation", tactivation}}, {{"out", tout}});
    fence->wait();

    std::vector<float> result(activationV.size() / kInFeatures * kOutFeatures);
    tout->read(result.data(), result.size() * sizeof(float));

    // Build the transposed weight matrix [kInFeatures, kOutFeatures] used by the matmul.
    std::vector<float> weightTransposed(static_cast<size_t>(kInFeatures * kOutFeatures));
    for (int64_t o = 0; o < kOutFeatures; ++o)
        for (int64_t i = 0; i < kInFeatures; ++i)
            weightTransposed[static_cast<size_t>(i * kOutFeatures + o)] =
                static_cast<float>(quantValues[static_cast<size_t>(o * kInFeatures + i)]) * decodedScale;

    for (int64_t b = 0; b < 2; ++b)
    {
        for (int64_t n = 0; n < kOutFeatures; ++n)
        {
            float expected = 0.f;
            for (int64_t k = 0; k < kInFeatures; ++k)
            {
                expected += activationV[static_cast<size_t>(b * kInFeatures + k)] *
                            weightTransposed[static_cast<size_t>(k * kOutFeatures + n)];
            }
            EXPECT_NEAR(result[static_cast<size_t>(b * kOutFeatures + n)], expected, 1e-4f)
                << " mismatch at b=" << b << " n=" << n;
        }
    }
}

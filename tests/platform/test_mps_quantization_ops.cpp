#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
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

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q8_0)
{
    constexpr int64_t kOutFeatures = 4;
    constexpr int64_t kInFeatures = 8;
    constexpr size_t kBlockElements = 32;
    std::vector<float> weightGguf(static_cast<size_t>(kOutFeatures * kInFeatures));
    for (size_t i = 0; i < weightGguf.size(); ++i)
        weightGguf[i] = static_cast<float>(static_cast<int>(i % 7) - 3);

    float maxAbs = 0.f;
    for (float v : weightGguf)
        maxAbs = std::max(maxAbs, std::abs(v));
    float scale = maxAbs / 127.0f;
    uint16_t scaleBits = cnn::encodeFloat16(scale);
    float decodedScale = cnn::decodeFloat16(scaleBits);

    std::vector<uint8_t> weightBytes;
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

    auto context = makeGpuContext();
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
            EXPECT_NEAR(result[static_cast<size_t>(b * kOutFeatures + n)], expected, 1e-3f)
                << " mismatch at b=" << b << " n=" << n;
        }
    }
}

namespace
{
    std::vector<uint8_t> makeQ4KWeight(int64_t outFeatures, int64_t inFeatures,
                                       const std::vector<float> &values)
    {
        constexpr int kSuperBlockSize = 256;
        constexpr int kSuperBlockBytes = 144;
        int64_t total = outFeatures * inFeatures;
        if (total % kSuperBlockSize != 0)
            throw std::runtime_error("makeQ4KWeight: element count must be a multiple of 256");
        if (static_cast<int64_t>(values.size()) != total)
            throw std::runtime_error("makeQ4KWeight: values size mismatch");

        std::vector<uint8_t> bytes;
        bytes.reserve(static_cast<size_t>((total / kSuperBlockSize) * kSuperBlockBytes));

        float d = 0.5f;
        float min = 0.1f;
        uint16_t dBits = cnn::encodeFloat16(d);
        uint16_t minBits = cnn::encodeFloat16(min);

        // scales[12] chosen so every getScaleMinK4() returns sc=1, m=0.
        uint8_t scales[12] = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};

        for (int64_t sb = 0; sb < total / kSuperBlockSize; ++sb)
        {
            bytes.insert(bytes.end(), reinterpret_cast<const uint8_t *>(&dBits),
                         reinterpret_cast<const uint8_t *>(&dBits) + sizeof(dBits));
            bytes.insert(bytes.end(), reinterpret_cast<const uint8_t *>(&minBits),
                         reinterpret_cast<const uint8_t *>(&minBits) + sizeof(minBits));
            bytes.insert(bytes.end(), scales, scales + 12);

            uint8_t qs[128] = {};
            int64_t base = sb * kSuperBlockSize;
            for (int offset = 0; offset < kSuperBlockSize; ++offset)
            {
                int flatIdx = static_cast<int>(base + offset);
                int nibble = static_cast<int>(std::clamp(values[flatIdx] / d, 0.0f, 15.0f));
                int group = offset / 64;
                int subGroup = (offset % 64) / 32;
                int l = offset % 32;
                int idx = group * 32 + l;
                if (subGroup == 0)
                    qs[idx] = (qs[idx] & 0xF0u) | static_cast<uint8_t>(nibble);
                else
                    qs[idx] = (qs[idx] & 0x0Fu) | static_cast<uint8_t>(nibble << 4);
            }
            bytes.insert(bytes.end(), qs, qs + 128);
        }
        return bytes;
    }
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q4_K)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 256;
    std::vector<float> weightGguf(static_cast<size_t>(kOutFeatures * kInFeatures));
    for (size_t i = 0; i < weightGguf.size(); ++i)
        weightGguf[i] = static_cast<float>((i % 15) + 1) * 0.5f;

    std::vector<uint8_t> weightBytes = makeQ4KWeight(kOutFeatures, kInFeatures, weightGguf);

    // CPU reference result as ground truth.
    auto cpuContext = makeCpuContext();
    cnn::GraphBuilder cpuBuilder(cpuContext);
    auto cpuActivation = cpuBuilder.input("activation", {cnn::DataType::Float32, {1, kInFeatures}});
    auto cpuWeight = cpuBuilder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightBytes.size())}, false, false},
                                         weightBytes.data(), weightBytes.size());
    auto cpuGraph = cpuBuilder.build({{"out", cpuBuilder.ggmlQuantizedMatmul(cpuActivation, cpuWeight, 12,
                                                                                {kInFeatures, kOutFeatures})}});

    std::vector<float> activationV(static_cast<size_t>(kInFeatures));
    for (size_t i = 0; i < activationV.size(); ++i)
        activationV[i] = static_cast<float>(static_cast<int>(i % 5) - 2);

    auto cpuIn = cpuContext->createTensor({cnn::DataType::Float32, {1, kInFeatures}, false, true});
    auto cpuOut = cpuContext->createTensor({cnn::DataType::Float32, {1, kOutFeatures}, true, false});
    cpuIn->write(activationV.data(), activationV.size() * sizeof(float));
    auto cpuFence = cpuContext->dispatch(*cpuGraph, {{"activation", cpuIn}}, {{"out", cpuOut}});
    cpuFence->wait();
    std::vector<float> expected(kOutFeatures);
    cpuOut->read(expected.data(), expected.size() * sizeof(float));

    // MPS segmented path.
    auto gpuContext = makeGpuContext();
    cnn::GraphBuilder gpuBuilder(gpuContext);
    auto gpuActivation = gpuBuilder.input("activation", {cnn::DataType::Float32, {1, kInFeatures}});
    auto gpuWeight = gpuBuilder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightBytes.size())}, false, false},
                                         weightBytes.data(), weightBytes.size());
    auto gpuGraph = gpuBuilder.build({{"out", gpuBuilder.ggmlQuantizedMatmul(gpuActivation, gpuWeight, 12,
                                                                                {kInFeatures, kOutFeatures})}});

    auto gpuIn = gpuContext->createTensor({cnn::DataType::Float32, {1, kInFeatures}, false, true});
    auto gpuOut = gpuContext->createTensor({cnn::DataType::Float32, {1, kOutFeatures}, true, false});
    gpuIn->write(activationV.data(), activationV.size() * sizeof(float));
    auto gpuFence = gpuContext->dispatch(*gpuGraph, {{"activation", gpuIn}}, {{"out", gpuOut}});
    gpuFence->wait();
    std::vector<float> result(kOutFeatures);
    gpuOut->read(result.data(), result.size() * sizeof(float));

    for (int64_t n = 0; n < kOutFeatures; ++n)
        EXPECT_NEAR(result[n], expected[n], 1e-3f) << " mismatch at n=" << n;
}

namespace
{
    // Fills `bytes` with a deterministic pseudo-random byte stream (a tiny LCG,
    // not meant to be a good RNG - just reproducible across runs/platforms).
    void appendRandomBytes(std::vector<uint8_t> &bytes, size_t count, uint32_t &state)
    {
        for (size_t i = 0; i < count; ++i)
        {
            state = state * 1664525u + 1013904223u;
            bytes.push_back(static_cast<uint8_t>(state >> 24));
        }
    }

    void appendHalf(std::vector<uint8_t> &bytes, float v)
    {
        uint16_t bits = cnn::encodeFloat16(v);
        bytes.insert(bytes.end(), reinterpret_cast<const uint8_t *>(&bits),
                     reinterpret_cast<const uint8_t *>(&bits) + sizeof(bits));
    }

    void appendFloat32(std::vector<uint8_t> &bytes, float v)
    {
        bytes.insert(bytes.end(), reinterpret_cast<const uint8_t *>(&v),
                     reinterpret_cast<const uint8_t *>(&v) + sizeof(v));
    }

    // Runs the same raw GGML-quantized bytes through the CPU backend (ground truth,
    // already covered by CpuQuantizationOps tests) and the MPS backend, and checks
    // they agree. This is a bit-unpacking parity check, not a quantization-accuracy
    // check, so the byte contents don't need to represent a "real" quantized tensor.
    void expectMpsMatchesCpuForGgmlType(int32_t ggmlType, int64_t outFeatures, int64_t inFeatures,
                                        const std::vector<uint8_t> &weightBytes)
    {
        std::vector<float> activationV(static_cast<size_t>(inFeatures));
        for (size_t i = 0; i < activationV.size(); ++i)
            activationV[i] = (static_cast<float>(static_cast<int>(i % 9) - 4)) * 0.37f;

        auto cpuContext = makeCpuContext();
        cnn::GraphBuilder cpuBuilder(cpuContext);
        auto cpuActivation = cpuBuilder.input("activation", {cnn::DataType::Float32, {1, inFeatures}});
        auto cpuWeight = cpuBuilder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightBytes.size())}, false, false},
                                             weightBytes.data(), weightBytes.size());
        auto cpuGraph = cpuBuilder.build({{"out", cpuBuilder.ggmlQuantizedMatmul(cpuActivation, cpuWeight, ggmlType,
                                                                                    {inFeatures, outFeatures})}});
        auto cpuIn = cpuContext->createTensor({cnn::DataType::Float32, {1, inFeatures}, false, true});
        auto cpuOut = cpuContext->createTensor({cnn::DataType::Float32, {1, outFeatures}, true, false});
        cpuIn->write(activationV.data(), activationV.size() * sizeof(float));
        auto cpuFence = cpuContext->dispatch(*cpuGraph, {{"activation", cpuIn}}, {{"out", cpuOut}});
        cpuFence->wait();
        std::vector<float> expected(static_cast<size_t>(outFeatures));
        cpuOut->read(expected.data(), expected.size() * sizeof(float));

        auto gpuContext = makeGpuContext();
        cnn::GraphBuilder gpuBuilder(gpuContext);
        auto gpuActivation = gpuBuilder.input("activation", {cnn::DataType::Float32, {1, inFeatures}});
        auto gpuWeight = gpuBuilder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightBytes.size())}, false, false},
                                             weightBytes.data(), weightBytes.size());
        auto gpuGraph = gpuBuilder.build({{"out", gpuBuilder.ggmlQuantizedMatmul(gpuActivation, gpuWeight, ggmlType,
                                                                                    {inFeatures, outFeatures})}});
        auto gpuIn = gpuContext->createTensor({cnn::DataType::Float32, {1, inFeatures}, false, true});
        auto gpuOut = gpuContext->createTensor({cnn::DataType::Float32, {1, outFeatures}, true, false});
        gpuIn->write(activationV.data(), activationV.size() * sizeof(float));
        auto gpuFence = gpuContext->dispatch(*gpuGraph, {{"activation", gpuIn}}, {{"out", gpuOut}});
        gpuFence->wait();
        std::vector<float> result(static_cast<size_t>(outFeatures));
        gpuOut->read(result.data(), result.size() * sizeof(float));

        for (int64_t n = 0; n < outFeatures; ++n)
        {
            float tol = std::abs(expected[static_cast<size_t>(n)]) * 1e-3f + 1e-3f;
            EXPECT_NEAR(result[static_cast<size_t>(n)], expected[static_cast<size_t>(n)], tol)
                << "ggmlType=" << ggmlType << " mismatch at n=" << n;
        }
    }
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q4_0)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 32; // 1 block of 32
    uint32_t state = 11111u;
    std::vector<uint8_t> weightBytes;
    for (int b = 0; b < 2; ++b)
    {
        appendHalf(weightBytes, 0.10f + 0.01f * b);
        appendRandomBytes(weightBytes, 16, state);
    }
    expectMpsMatchesCpuForGgmlType(2, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q4_1)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 32; // 2 blocks of 32
    uint32_t state = 12345u;
    std::vector<uint8_t> weightBytes;
    for (int b = 0; b < 2; ++b)
    {
        appendHalf(weightBytes, 0.10f + 0.01f * b);
        appendHalf(weightBytes, -0.20f + 0.01f * b);
        appendRandomBytes(weightBytes, 16, state);
    }
    expectMpsMatchesCpuForGgmlType(3, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q5_0)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 32;
    uint32_t state = 23456u;
    std::vector<uint8_t> weightBytes;
    for (int b = 0; b < 2; ++b)
    {
        appendHalf(weightBytes, 0.15f + 0.02f * b);
        appendRandomBytes(weightBytes, 4, state);
        appendRandomBytes(weightBytes, 16, state);
    }
    expectMpsMatchesCpuForGgmlType(6, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q5_1)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 32;
    uint32_t state = 34567u;
    std::vector<uint8_t> weightBytes;
    for (int b = 0; b < 2; ++b)
    {
        appendHalf(weightBytes, 0.12f + 0.01f * b);
        appendHalf(weightBytes, -0.10f + 0.01f * b);
        appendRandomBytes(weightBytes, 4, state);
        appendRandomBytes(weightBytes, 16, state);
    }
    expectMpsMatchesCpuForGgmlType(7, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q8_1)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 32;
    uint32_t state = 45678u;
    std::vector<uint8_t> weightBytes;
    for (int b = 0; b < 2; ++b)
    {
        appendHalf(weightBytes, 0.05f + 0.01f * b);
        appendRandomBytes(weightBytes, 2, state); // 's' field - unused by the dequant kernel
        appendRandomBytes(weightBytes, 32, state);
    }
    expectMpsMatchesCpuForGgmlType(9, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q2_K)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 256; // 2 super-blocks of 256
    uint32_t state = 56789u;
    std::vector<uint8_t> weightBytes;
    for (int sb = 0; sb < 2; ++sb)
    {
        appendRandomBytes(weightBytes, 16, state); // scales
        appendRandomBytes(weightBytes, 64, state); // qs
        appendHalf(weightBytes, 0.20f + 0.01f * sb);
        appendHalf(weightBytes, 0.05f + 0.01f * sb);
    }
    expectMpsMatchesCpuForGgmlType(10, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q3_K)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 256;
    uint32_t state = 67890u;
    std::vector<uint8_t> weightBytes;
    for (int sb = 0; sb < 2; ++sb)
    {
        appendRandomBytes(weightBytes, 32, state); // hmask
        appendRandomBytes(weightBytes, 64, state); // qs
        appendRandomBytes(weightBytes, 12, state); // scales
        appendHalf(weightBytes, 0.30f + 0.01f * sb);
    }
    expectMpsMatchesCpuForGgmlType(11, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q5_K)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 256;
    uint32_t state = 78901u;
    std::vector<uint8_t> weightBytes;
    for (int sb = 0; sb < 2; ++sb)
    {
        appendHalf(weightBytes, 0.25f + 0.01f * sb);
        appendHalf(weightBytes, 0.02f + 0.01f * sb);
        appendRandomBytes(weightBytes, 12, state);  // scales
        appendRandomBytes(weightBytes, 32, state);  // qh
        appendRandomBytes(weightBytes, 128, state); // ql
    }
    expectMpsMatchesCpuForGgmlType(13, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q6_K)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 256;
    uint32_t state = 89012u;
    std::vector<uint8_t> weightBytes;
    for (int sb = 0; sb < 2; ++sb)
    {
        appendRandomBytes(weightBytes, 128, state); // ql
        appendRandomBytes(weightBytes, 64, state);  // qh
        appendRandomBytes(weightBytes, 16, state);  // scales
        appendHalf(weightBytes, 0.18f + 0.01f * sb);
    }
    expectMpsMatchesCpuForGgmlType(14, kOutFeatures, kInFeatures, weightBytes);
}

TEST(MpsQuantizationOps, GgmlQuantizedMatmul_Q8_K)
{
    constexpr int64_t kOutFeatures = 2;
    constexpr int64_t kInFeatures = 256;
    uint32_t state = 90123u;
    std::vector<uint8_t> weightBytes;
    for (int sb = 0; sb < 2; ++sb)
    {
        appendFloat32(weightBytes, 0.03f + 0.005f * sb);
        appendRandomBytes(weightBytes, 256, state); // qs
        appendRandomBytes(weightBytes, 32, state);  // bsums - unused by the dequant kernel
    }
    expectMpsMatchesCpuForGgmlType(15, kOutFeatures, kInFeatures, weightBytes);
}

// mps_backend.mm's compileGraph() groups consecutive GgmlQuantizedMatmul nodes
// with no plain op between them (e.g. q/k/v or gate/up projections, which all
// read the same upstream activation and write to different outputs) into one
// segment, encoded into a single command buffer/encoder with one
// commit+waitUntilCompleted instead of one round trip per node. None of the
// tests above exercise that merged-segment path -- each builds a graph with
// only one ggmlQuantizedMatmul call. This one builds three, sharing the same
// activation input but different weights/outputs (mirroring q/k/v), and checks
// each output independently against the CPU reference -- if encoding multiple
// dispatches into one shared encoder ever clobbered a buffer binding or
// mis-ordered a dependency, exactly one of these would come back wrong.
TEST(MpsQuantizationOps, GgmlQuantizedMatmul_ThreeAdjacentNodesShareOneSegment)
{
    constexpr int64_t kInFeatures = 256;
    constexpr int64_t kOutFeaturesQ = 2;
    constexpr int64_t kOutFeaturesK = 3;
    constexpr int64_t kOutFeaturesV = 4;

    auto makeWeights = [](int64_t outFeatures, float scaleBase) {
        std::vector<float> weightGguf(static_cast<size_t>(outFeatures * kInFeatures));
        for (size_t i = 0; i < weightGguf.size(); ++i)
            weightGguf[i] = static_cast<float>((i % 15) + 1) * scaleBase;
        return makeQ4KWeight(outFeatures, kInFeatures, weightGguf);
    };
    std::vector<uint8_t> weightQ = makeWeights(kOutFeaturesQ, 0.5f);
    std::vector<uint8_t> weightK = makeWeights(kOutFeaturesK, 0.3f);
    std::vector<uint8_t> weightV = makeWeights(kOutFeaturesV, 0.7f);

    std::vector<float> activationV(static_cast<size_t>(kInFeatures));
    for (size_t i = 0; i < activationV.size(); ++i)
        activationV[i] = static_cast<float>(static_cast<int>(i % 5) - 2);

    auto runThree = [&](std::shared_ptr<cnn::Context> context, std::vector<float> &outQ, std::vector<float> &outK,
                        std::vector<float> &outV) {
        cnn::GraphBuilder builder(context);
        auto activation = builder.input("activation", {cnn::DataType::Float32, {1, kInFeatures}});
        auto wQ = builder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightQ.size())}, false, false},
                                   weightQ.data(), weightQ.size());
        auto wK = builder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightK.size())}, false, false},
                                   weightK.data(), weightK.size());
        auto wV = builder.constant({cnn::DataType::Int8, {static_cast<int64_t>(weightV.size())}, false, false},
                                   weightV.data(), weightV.size());
        // Declared back-to-back with no plain op between them, same as q/k/v.
        auto q = builder.ggmlQuantizedMatmul(activation, wQ, 12, {kInFeatures, kOutFeaturesQ});
        auto k = builder.ggmlQuantizedMatmul(activation, wK, 12, {kInFeatures, kOutFeaturesK});
        auto v = builder.ggmlQuantizedMatmul(activation, wV, 12, {kInFeatures, kOutFeaturesV});
        auto graph = builder.build({{"q", q}, {"k", k}, {"v", v}});

        auto tIn = context->createTensor({cnn::DataType::Float32, {1, kInFeatures}, false, true});
        auto tQ = context->createTensor({cnn::DataType::Float32, {1, kOutFeaturesQ}, true, false});
        auto tK = context->createTensor({cnn::DataType::Float32, {1, kOutFeaturesK}, true, false});
        auto tV = context->createTensor({cnn::DataType::Float32, {1, kOutFeaturesV}, true, false});
        tIn->write(activationV.data(), activationV.size() * sizeof(float));

        auto fence = context->dispatch(*graph, {{"activation", tIn}}, {{"q", tQ}, {"k", tK}, {"v", tV}});
        fence->wait();

        outQ.resize(kOutFeaturesQ);
        outK.resize(kOutFeaturesK);
        outV.resize(kOutFeaturesV);
        tQ->read(outQ.data(), outQ.size() * sizeof(float));
        tK->read(outK.data(), outK.size() * sizeof(float));
        tV->read(outV.data(), outV.size() * sizeof(float));
    };

    std::vector<float> expectedQ, expectedK, expectedV;
    runThree(makeCpuContext(), expectedQ, expectedK, expectedV);

    std::vector<float> resultQ, resultK, resultV;
    runThree(makeGpuContext(), resultQ, resultK, resultV);

    for (int64_t n = 0; n < kOutFeaturesQ; ++n)
        EXPECT_NEAR(resultQ[n], expectedQ[n], 1e-3f) << "q mismatch at n=" << n;
    for (int64_t n = 0; n < kOutFeaturesK; ++n)
        EXPECT_NEAR(resultK[n], expectedK[n], 1e-3f) << "k mismatch at n=" << n;
    for (int64_t n = 0; n < kOutFeaturesV; ++n)
        EXPECT_NEAR(resultV[n], expectedV[n], 1e-3f) << "v mismatch at n=" << n;
}

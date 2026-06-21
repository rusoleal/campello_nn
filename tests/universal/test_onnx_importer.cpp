#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include <campello_nn/onnx_importer.hpp>
#include "test_helpers.hpp"

// Imports a small, real ONNX graph (Conv -> Add -> Relu, generated and validated
// with Python's onnx package) and checks the result against the same hand-computed
// values used elsewhere in this suite (see CpuOps.Conv2d).
TEST(OnnxImporter, ImportConvAddRelu)
{
    auto context = makeCpuContext();
    std::string path = std::string(CAMPELLO_NN_TEST_FIXTURES_DIR) + "/conv_add_relu.onnx";
    auto result = cnn::importOnnxFromFile(context, path);

    ASSERT_EQ(result.inputs.size(), 1u);
    ASSERT_EQ(result.outputs.size(), 1u);
    ASSERT_TRUE(result.inputs.count("x"));
    ASSERT_TRUE(result.outputs.count("out"));

    std::vector<int64_t> expectedInShape = {1, 1, 4, 4};
    std::vector<int64_t> expectedOutShape = {1, 1, 3, 3};
    EXPECT_EQ(result.inputs["x"].shape, expectedInShape);
    EXPECT_EQ(result.inputs["x"].dataType, cnn::DataType::Float32);
    EXPECT_EQ(result.outputs["out"].shape, expectedOutShape);

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 4, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, true, false});

    float xv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*result.graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float out[9];
    tout->read(out, sizeof(out));
    // conv (diagonal-pick kernel) -> [7,9,11,15,17,19,23,25,27], +100 bias, relu (no-op, all positive)
    float expected[9] = {107, 109, 111, 115, 117, 119, 123, 125, 127};
    for (int i = 0; i < 9; i++)
        EXPECT_FLOAT_EQ(out[i], expected[i]);
}

// Same fixture, but via the primary importOnnxFromMemory() entry point — the one
// that matters for callers without a real filesystem path (Android assets, etc.).
TEST(OnnxImporter, ImportFromMemory)
{
    std::string path = std::string(CAMPELLO_NN_TEST_FIXTURES_DIR) + "/conv_add_relu.onnx";
    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto context = makeCpuContext();
    auto result = cnn::importOnnxFromMemory(context, bytes.data(), bytes.size());

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 4, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, true, false});

    float xv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*result.graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float out[9];
    tout->read(out, sizeof(out));
    float expected[9] = {107, 109, 111, 115, 117, 119, 123, 125, 127};
    for (int i = 0; i < 9; i++)
        EXPECT_FLOAT_EQ(out[i], expected[i]);
}

#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include <campello_nn/tflite_importer.hpp>
#include "test_helpers.hpp"

// Imports a small, real TFLite graph — Conv2D(no bias) -> Add(broadcast bias) ->
// Relu, computing the exact same thing as tests/fixtures/conv_add_relu.onnx (see
// test_onnx_importer.cpp) so the two importers can be checked against identical
// expected values. Generated from tests/fixtures/conv_add_relu_tflite.json via:
//
//   flatc --binary tflite_schema.fbs conv_add_relu_tflite.json
//
// (both files live in tests/fixtures/; tflite_schema.fbs is google-ai-edge/LiteRT's
// schema.fbs, see fixtures/NOTICE.md). TFLite tensors are NHWC, but since every
// non-batch/non-spatial dim here is 1 (single channel), the NHWC [1,4,4,1] input
// and NCHW [1,1,4,4] (ONNX's) are byte-for-byte identical — this fixture doesn't
// exercise a real N>1-channel NHWC<->NCHW byte reordering, just the import
// pipeline's op coverage and shape bookkeeping end-to-end.
TEST(TfliteImporter, ImportConvAddRelu)
{
    auto context = makeCpuContext();
    std::string path = std::string(CAMPELLO_NN_TEST_FIXTURES_DIR) + "/conv_add_relu.tflite";
    auto result = cnn::importTfliteFromFile(context, path);

    ASSERT_EQ(result.inputs.size(), 1u);
    ASSERT_EQ(result.outputs.size(), 1u);
    ASSERT_TRUE(result.inputs.count("x"));
    ASSERT_TRUE(result.outputs.count("out"));

    std::vector<int64_t> expectedInShape = {1, 4, 4, 1}; // NHWC, matching the file's own declared shape
    std::vector<int64_t> expectedOutShape = {1, 3, 3, 1};
    EXPECT_EQ(result.inputs["x"].shape, expectedInShape);
    EXPECT_EQ(result.inputs["x"].dataType, cnn::DataType::Float32);
    EXPECT_EQ(result.outputs["out"].shape, expectedOutShape);

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4, 4, 1}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 3, 3, 1}, true, false});

    float xv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*result.graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float out[9];
    tout->read(out, sizeof(out));
    // Same computation as OnnxImporter.ImportConvAddRelu: conv (diagonal-pick
    // kernel) -> [7,9,11,15,17,19,23,25,27], +100 bias, relu (no-op, all positive).
    float expected[9] = {107, 109, 111, 115, 117, 119, 123, 125, 127};
    for (int i = 0; i < 9; i++)
        EXPECT_FLOAT_EQ(out[i], expected[i]);
}

// Same fixture, but via the primary importTfliteFromMemory() entry point — the
// one that matters for callers without a real filesystem path (Android assets, etc.).
TEST(TfliteImporter, ImportFromMemory)
{
    std::string path = std::string(CAMPELLO_NN_TEST_FIXTURES_DIR) + "/conv_add_relu.tflite";
    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto context = makeCpuContext();
    auto result = cnn::importTfliteFromMemory(context, bytes.data(), bytes.size());

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4, 4, 1}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 3, 3, 1}, true, false});

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

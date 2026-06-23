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

// Checks GraphInfo (the public topology-inspection API — see graph_info.hpp) against
// the same conv_add_relu.tflite fixture. Unlike the ONNX importer, TFLite's rank-4
// activations are converted NHWC<->NCHW at each graph boundary (see
// tflite_importer.hpp), so the topology includes two extra Transpose nodes the ONNX
// importer's GraphInfo doesn't have (see OnnxImporter.GraphInfoDescribesTopology):
// one right after the input, one right before the output.
TEST(TfliteImporter, GraphInfoDescribesTopology)
{
    auto context = makeCpuContext();
    std::string path = std::string(CAMPELLO_NN_TEST_FIXTURES_DIR) + "/conv_add_relu.tflite";
    auto result = cnn::importTfliteFromFile(context, path);

    ASSERT_EQ(result.info.nodes.size(), 8u);

    EXPECT_EQ(result.info.nodes[0].kind, cnn::OpKind::Input);
    EXPECT_EQ(result.info.nodes[0].name, "x");
    EXPECT_EQ(result.info.nodes[1].kind, cnn::OpKind::Transpose); // NHWC -> NCHW
    EXPECT_EQ(result.info.nodes[2].kind, cnn::OpKind::Constant);  // conv weight, OIHW
    EXPECT_EQ(result.info.nodes[3].kind, cnn::OpKind::Conv2d);
    EXPECT_EQ(result.info.nodes[4].kind, cnn::OpKind::Constant); // bias
    EXPECT_EQ(result.info.nodes[5].kind, cnn::OpKind::Add);
    EXPECT_EQ(result.info.nodes[6].kind, cnn::OpKind::Relu);
    EXPECT_EQ(result.info.nodes[7].kind, cnn::OpKind::Transpose); // NCHW -> NHWC

    std::vector<int64_t> expectedOutShape = {1, 3, 3, 1}; // NHWC, matching the file's own declared shape
    EXPECT_EQ(result.info.nodes[7].shape, expectedOutShape);

    ASSERT_EQ(result.info.outputs.size(), 1u);
    EXPECT_EQ(result.info.outputs[0].first, "out");
    EXPECT_EQ(result.info.outputs[0].second, 7u);
}

// DEPTHWISE_CONV_2D, the op-code this importer initially shipped without
// support for (see TODO.md Phase 4b) — its weight layout ([1,filter_height,
// filter_width,output_depth], confirmed against TFLite's own reference kernel,
// tflite/kernels/internal/reference/depthwiseconv_float.h) is verified here by
// hand-computing the expected output independently, not just trusting the
// layout. 1x1 kernel (no spatial mixing) over 2 input channels, depth_multiplier
// 1, weight=[2,3] (per-channel scalar), bias=[100,200], fused RELU (a no-op
// here since everything stays positive).
TEST(TfliteImporter, ImportDepthwiseConv2d)
{
    auto context = makeCpuContext();
    std::string path = std::string(CAMPELLO_NN_TEST_FIXTURES_DIR) + "/depthwise_conv2d.tflite";
    auto result = cnn::importTfliteFromFile(context, path);

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, true, false});

    // NHWC: (h,w) pixels each holding [ch0,ch1] = (1,10),(2,20),(3,30),(4,40).
    float xv[8] = {1, 10, 2, 20, 3, 30, 4, 40};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*result.graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float out[8];
    tout->read(out, sizeof(out));
    // ch0 = x_ch0*2 + 100, ch1 = x_ch1*3 + 200, interleaved NHWC.
    float expected[8] = {102, 230, 104, 260, 106, 290, 108, 320};
    for (int i = 0; i < 8; i++)
        EXPECT_FLOAT_EQ(out[i], expected[i]);
}

// BATCH_MATMUL with adj_y=true — exercises the transpose-insertion path
// (GraphBuilder::matmul() has no transpose flag of its own, so adj_x/adj_y
// become an explicit transpose() of the last two axes before the matmul; see
// tflite_importer.cpp's swapLastTwoAxes()). adj_y semantics confirmed against
// tflite/kernels/batch_matmul.cc ("transpose the last two dimensions").
TEST(TfliteImporter, ImportBatchMatmulAdjY)
{
    auto context = makeCpuContext();
    std::string path = std::string(CAMPELLO_NN_TEST_FIXTURES_DIR) + "/batch_matmul.tflite";
    auto result = cnn::importTfliteFromFile(context, path);

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    // a = [[1,2],[3,4]]; b (constant, stored as-is) = [[5,6],[7,8]].
    float av[4] = {1, 2, 3, 4};
    ta->write(av, sizeof(av));

    auto fence = context->dispatch(*result.graph, {{"a", ta}}, {{"out", tout}});
    fence->wait();

    float out[4];
    tout->read(out, sizeof(out));
    // adj_y=true means the actual right-multiplicand is b^T = [[5,7],[6,8]].
    // a @ b^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]] = [[17,23],[39,53]].
    float expected[4] = {17, 23, 39, 53};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(out[i], expected[i]);
}

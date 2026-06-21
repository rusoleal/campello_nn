#include <gtest/gtest.h>
#include <filesystem>
#include <campello_nn/graph_cache.hpp>
#include "test_helpers.hpp"

TEST(GraphCache, RoundTripGelu)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto bytes = builder.serialize({{"out", builder.gelu(x)}});

    auto cached = cnn::loadGraphFromMemory(context, bytes.data(), bytes.size());
    ASSERT_EQ(cached.inputs.size(), 1u);
    ASSERT_EQ(cached.outputs.size(), 1u);
    EXPECT_EQ(cached.inputs.at("x").shape, (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(cached.outputs.at("out").shape, (std::vector<int64_t>{1, 4}));

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});
    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*cached.graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();
    float result[4];
    tout->read(result, sizeof(result));

    // Same graph built directly (no cache round trip), for comparison.
    cnn::GraphBuilder freshBuilder(context);
    auto freshX = freshBuilder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto freshGraph = freshBuilder.build({{"out", freshBuilder.gelu(freshX)}});
    auto freshOut = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});
    auto freshFence = context->dispatch(*freshGraph, {{"x", tx}}, {{"out", freshOut}});
    freshFence->wait();
    float expected[4];
    freshOut->read(expected, sizeof(expected));

    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GraphCache, RoundTripConvPoolResizeDescriptors)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 3, 3}});
    auto w = builder.input("w", {cnn::DataType::Float32, {1, 1, 2, 2}});
    auto conv = builder.conv2d(x, w, cnn::Conv2dDescriptor{});
    cnn::ResizeDescriptor resizeDesc;
    resizeDesc.outputHeight = 3;
    resizeDesc.outputWidth = 3;
    resizeDesc.mode = cnn::ResizeMode::Nearest;
    resizeDesc.nearestRoundsDown = true;
    auto resized = builder.resize(conv, resizeDesc);
    auto bytes = builder.serialize({{"out", resized}});

    auto cached = cnn::loadGraphFromMemory(context, bytes.data(), bytes.size());
    EXPECT_EQ(cached.outputs.at("out").shape, (std::vector<int64_t>{1, 1, 3, 3}));

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, false, true});
    auto tw = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, true, false});
    float xv[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float wv[4] = {1, 0, 0, 1};
    tx->write(xv, sizeof(xv));
    tw->write(wv, sizeof(wv));

    auto fence = context->dispatch(*cached.graph, {{"x", tx}, {"w", tw}}, {{"out", tout}});
    fence->wait();
    float result[9];
    tout->read(result, sizeof(result));

    cnn::GraphBuilder freshBuilder(context);
    auto freshX = freshBuilder.input("x", {cnn::DataType::Float32, {1, 1, 3, 3}});
    auto freshW = freshBuilder.input("w", {cnn::DataType::Float32, {1, 1, 2, 2}});
    auto freshConv = freshBuilder.conv2d(freshX, freshW, cnn::Conv2dDescriptor{});
    auto freshResized = freshBuilder.resize(freshConv, resizeDesc);
    auto freshGraph = freshBuilder.build({{"out", freshResized}});
    auto freshOut = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, true, false});
    auto freshFence = context->dispatch(*freshGraph, {{"x", tx}, {"w", tw}}, {{"out", freshOut}});
    freshFence->wait();
    float expected[9];
    freshOut->read(expected, sizeof(expected));

    for (int i = 0; i < 9; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GraphCache, RoundTripPreservesConstantBytes)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 3}});
    float constVals[3] = {10.f, 20.f, 30.f};
    auto c = builder.constant({cnn::DataType::Float32, {1, 3}, false, false}, constVals, sizeof(constVals));
    auto bytes = builder.serialize({{"out", builder.add(x, c)}});

    auto cached = cnn::loadGraphFromMemory(context, bytes.data(), bytes.size());
    auto tx = context->createTensor({cnn::DataType::Float32, {1, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 3}, true, false});
    float xv[3] = {1, 2, 3};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*cached.graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();
    float result[3];
    tout->read(result, sizeof(result));
    float expected[3] = {11, 22, 33};
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GraphCache, FileRoundTrip)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto bytes = builder.serialize({{"out", builder.relu(x)}});

    auto path = (std::filesystem::temp_directory_path() / "campello_nn_test_graph_cache.bin").string();
    cnn::saveGraphToFile(bytes, path);
    auto cached = cnn::loadGraphFromFile(context, path);
    std::filesystem::remove(path);

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});
    float xv[4] = {-1.f, -0.5f, 0.5f, 1.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*cached.graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();
    float result[4];
    tout->read(result, sizeof(result));
    float expected[4] = {0.f, 0.f, 0.5f, 1.f};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GraphCache, LoadGraphFromFileMissingFileThrows)
{
    auto context = makeCpuContext();
    EXPECT_THROW(cnn::loadGraphFromFile(context, "/nonexistent/path/campello_nn_missing.bin"), std::runtime_error);
}

TEST(GraphCache, CorruptMagicThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto bytes = builder.serialize({{"out", builder.gelu(x)}});
    bytes[0] ^= 0xFF; // corrupt the magic header
    EXPECT_THROW(cnn::loadGraphFromMemory(context, bytes.data(), bytes.size()), std::runtime_error);
}

TEST(GraphCache, TruncatedBufferThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto bytes = builder.serialize({{"out", builder.gelu(x)}});
    bytes.resize(bytes.size() / 2);
    EXPECT_THROW(cnn::loadGraphFromMemory(context, bytes.data(), bytes.size()), std::runtime_error);
}

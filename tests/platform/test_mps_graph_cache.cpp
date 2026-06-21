#include <gtest/gtest.h>
#include <campello_nn/graph_cache.hpp>
#include "../universal/test_helpers.hpp"

// Confirms a serialized graph compiles and runs correctly against the real
// MPSGraph/GPU backend too, not just the CPU reference backend (see
// tests/universal/test_graph_cache.cpp for the broader CPU-only coverage).
TEST(MpsOps, GraphCacheRoundTrip)
{
    auto context = makeGpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3}});
    auto bytes = builder.serialize({{"out", builder.add(a, b)}});

    auto cached = cnn::loadGraphFromMemory(context, bytes.data(), bytes.size());

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 3}, true, false});
    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[3] = {10, 20, 30};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*cached.graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();
    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {11, 22, 33, 14, 25, 36};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

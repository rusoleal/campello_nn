#include <gtest/gtest.h>
#include "../universal/test_helpers.hpp"

// Exercises DeviceType::GpuGeneric (src/gpu/gpu_backend.cpp) — the
// campello_gpu-based backend, available on every platform unlike the
// platform-native MpsOps/DirectMlOps tests this mirrors. Vertical slice only:
// Relu, exact-shape Add, and rank-2 unbatched MatMul (see TODO.md).

TEST(GpuGenericOps, Relu)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.relu(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4}, true, false});

    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    float expected[4] = {0, 0, 0.5f, 2.f};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GpuGenericOps, AddExactShape)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 2}});
    auto graph = builder.build({{"out", builder.add(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float av[4] = {1, 2, 3, 4};
    float bv[4] = {10, 20, 30, 40};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], av[i] + bv[i]);
}

TEST(GpuGenericOps, MatMul)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3, 2}});
    auto graph = builder.build({{"out", builder.matmul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[6] = {7, 8, 9, 10, 11, 12};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    // [1,2,3; 4,5,6] @ [7,8; 9,10; 11,12] = [58,64; 139,154]
    EXPECT_FLOAT_EQ(result[0], 58);
    EXPECT_FLOAT_EQ(result[1], 64);
    EXPECT_FLOAT_EQ(result[2], 139);
    EXPECT_FLOAT_EQ(result[3], 154);
}

// Specifically exercises the open question recorded in TODO.md: does
// campello_gpu auto-track resource hazards between two dispatches in the same
// compute pass (WebGPU-style APIs are supposed to, but ComputePassEncoder
// exposes no explicit barrier call to confirm it)? A single-op test wouldn't
// catch a missing barrier between relu's write and add's read of the same
// buffer — this graph specifically chains them.
TEST(GpuGenericOps, ChainedReluThenAdd)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto b = builder.input("b", {cnn::DataType::Float32, {4}});
    auto relu = builder.relu(x);
    auto graph = builder.build({{"out", builder.add(relu, b)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4}, true, false});

    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    float bv[4] = {100.f, 100.f, 100.f, 100.f};
    tx->write(xv, sizeof(xv));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    // relu(x) = {0, 0, 0.5, 2}; + b = {100, 100, 100.5, 102}
    float expected[4] = {100.f, 100.f, 100.5f, 102.f};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

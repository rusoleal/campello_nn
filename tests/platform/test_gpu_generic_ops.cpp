

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <campello_nn/onnx_importer.hpp>
#include "../universal/test_helpers.hpp"

// Exercises DeviceType::GpuGeneric (src/gpu/gpu_backend.cpp) — the
// campello_gpu-based backend, available on every platform unlike the
// platform-native MpsOps/DirectMlOps tests this mirrors.

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

TEST(GpuGenericOps, MatMulBatched)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    // batch=2, M=2, K=3, N=4
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 3, 4}});
    auto graph = builder.build({{"out", builder.matmul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 2, 3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 3, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2, 4}, true, false});

    float av[12] = {
        1, 2, 3,
        4, 5, 6,
        // second batch: identity-ish scaled
        1, 0, 0,
        0, 1, 0};
    float bv[24] = {
        1, 0, 0, 1,
        0, 1, 0, 0,
        0, 0, 1, 0,
        // second batch
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[16];
    tout->read(result, sizeof(result));
    // batch 0: a[0] @ b[0]
    EXPECT_FLOAT_EQ(result[0], 1);
    EXPECT_FLOAT_EQ(result[1], 2);
    EXPECT_FLOAT_EQ(result[2], 3);
    EXPECT_FLOAT_EQ(result[3], 1);
    EXPECT_FLOAT_EQ(result[4], 4);
    EXPECT_FLOAT_EQ(result[5], 5);
    EXPECT_FLOAT_EQ(result[6], 6);
    EXPECT_FLOAT_EQ(result[7], 4);
    // batch 1: a[1] @ b[1]
    EXPECT_FLOAT_EQ(result[8], 7);
    EXPECT_FLOAT_EQ(result[9], 8);
    EXPECT_FLOAT_EQ(result[10], 9);
    EXPECT_FLOAT_EQ(result[11], 10);
    EXPECT_FLOAT_EQ(result[12], 11);
    EXPECT_FLOAT_EQ(result[13], 12);
    EXPECT_FLOAT_EQ(result[14], 13);
    EXPECT_FLOAT_EQ(result[15], 14);
}

TEST(GpuGenericOps, MulExactShape)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3}});
    auto graph = builder.build({{"out", builder.mul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3}, true, false});

    float av[3] = {2, 3, 4};
    float bv[3] = {5, 6, 7};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[3];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(result[i], av[i] * bv[i]);
}

TEST(GpuGenericOps, Sigmoid)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.sigmoid(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4}, true, false});

    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
    {
        float expected = 1.0f / (1.0f + std::exp(-xv[i]));
        EXPECT_NEAR(result[i], expected, 1e-3f);
    }
}

TEST(GpuGenericOps, Gelu)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.gelu(x)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4}, true, false});

    float xv[4] = {-2.f, -0.5f, 0.5f, 2.f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    for (int i = 0; i < 4; i++)
    {
        // Reference uses libm's std::erf; the shader uses a polynomial
        // approximation (max error ~1.5e-7 — see gelu.comp's comment), hence
        // the tolerance rather than EXPECT_FLOAT_EQ.
        float expected = 0.5f * xv[i] * (1.0f + std::erf(xv[i] * 0.70710678118654752f));
        EXPECT_NEAR(result[i], expected, 1e-3f);
    }
}

TEST(GpuGenericOps, LayerNorm)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {4}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.layerNorm(x, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tbias = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});

    float xv[4] = {1, 2, 3, 4};
    float scaleV[4] = {1, 1, 1, 1};
    float biasV[4] = {0, 0, 0, 0};
    tx->write(xv, sizeof(xv));
    tscale->write(scaleV, sizeof(scaleV));
    tbias->write(biasV, sizeof(biasV));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}, {"bias", tbias}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));

    float mean = 2.5f;
    float var = 1.25f; // mean of (x-mean)^2 for {1,2,3,4}
    float invStd = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(result[i], (xv[i] - mean) * invStd, 1e-4f);
}

TEST(GpuGenericOps, RmsNorm)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.rmsNorm(x, scale, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 4}, false, true});
    auto tscale = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 4}, true, false});

    float xv[4] = {1, 2, 3, 4};
    float scaleV[4] = {2, 1, 0.5f, 1};
    tx->write(xv, sizeof(xv));
    tscale->write(scaleV, sizeof(scaleV));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"scale", tscale}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));

    float meanSquare = (1.f * 1.f + 2.f * 2.f + 3.f * 3.f + 4.f * 4.f) / 4.f;
    float invRms = 1.0f / std::sqrt(meanSquare + 1e-5f);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(result[i], xv[i] * invRms * scaleV[i], 1e-4f);
}

// Same hazard-tracking rationale as ChainedReluThenAdd below, but exercising
// a different op pair (a row-reduction op feeding into an elementwise one).
TEST(GpuGenericOps, ChainedGeluThenAdd)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto b = builder.input("b", {cnn::DataType::Float32, {4}});
    auto gelu = builder.gelu(x);
    auto graph = builder.build({{"out", builder.add(gelu, b)}});

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
    for (int i = 0; i < 4; i++)
    {
        float gelu_i = 0.5f * xv[i] * (1.0f + std::erf(xv[i] * 0.70710678118654752f));
        EXPECT_NEAR(result[i], gelu_i + bv[i], 1e-3f);
    }
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

// Reshape is a zero-cost alias (see gpu_backend.cpp) — chains it into add()
// to confirm dispatch() resolves the aliased buffer correctly rather than
// just checking Reshape in isolation (which wouldn't catch a bad alias).
TEST(GpuGenericOps, ReshapeThenAdd)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 2}});
    auto reshaped = builder.reshape(x, {2, 2});
    auto graph = builder.build({{"out", builder.add(reshaped, b)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float xv[4] = {1, 2, 3, 4};
    float bv[4] = {10, 20, 30, 40};
    tx->write(xv, sizeof(xv));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"x", tx}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    float expected[4] = {11, 22, 33, 44};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

// Rank-3 permute (not just the rank-2 swap a naive implementation might get
// away with) — exercises the generic divisor/gatherStride decode in
// transpose.comp for every dim, not just one.
TEST(GpuGenericOps, TransposeRank3)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 3, 4}});
    auto graph = builder.build({{"out", builder.transpose(x, {2, 0, 1})}});

    auto tx = context->createTensor({cnn::DataType::Float32, {2, 3, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4, 2, 3}, true, false});

    float xv[24];
    for (int i = 0; i < 24; i++)
        xv[i] = (float)i;
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[24];
    tout->read(result, sizeof(result));
    // out[k, i, j] = x[i, j, k] for i in 0..2, j in 0..3, k in 0..4
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 4; k++)
                EXPECT_FLOAT_EQ(result[(k * 2 + i) * 3 + j], xv[(i * 3 + j) * 4 + k]);
}

// Slices both dimensions of a rank-2 tensor (not just a single-axis slice on
// a rank-1 tensor) to exercise the per-dim baseOffset/divisor/multiplier
// math in slice.comp for more than one dim at once.
TEST(GpuGenericOps, Slice2D)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {3, 4}});
    auto graph = builder.build({{"out", builder.slice(x, {1, 1}, {2, 2})}});

    auto tx = context->createTensor({cnn::DataType::Float32, {3, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float xv[12];
    for (int i = 0; i < 12; i++)
        xv[i] = (float)i;
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    // x rows 1..2, cols 1..2: [[5,6],[9,10]]
    float expected[4] = {5, 6, 9, 10};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

// Three inputs (not two) along axis 0, specifically to exercise
// CompiledNode::concatPieces' multi-dispatch loop beyond the trivial
// two-piece case.
TEST(GpuGenericOps, ConcatThreeInputs)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {1, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 2}});
    auto c = builder.input("c", {cnn::DataType::Float32, {1, 2}});
    auto graph = builder.build({{"out", builder.concat({a, b, c}, 0)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {1, 2}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tc = context->createTensor({cnn::DataType::Float32, {1, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {4, 2}, true, false});

    float av[2] = {1, 2};
    float bv[4] = {3, 4, 5, 6};
    float cv[2] = {7, 8};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));
    tc->write(cv, sizeof(cv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}, {"c", tc}}, {{"out", tout}});
    fence->wait();

    float result[8];
    tout->read(result, sizeof(result));
    float expected[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 8; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GpuGenericOps, Gather)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto data = builder.input("data", {cnn::DataType::Float32, {4, 2}});
    auto indices = builder.input("indices", {cnn::DataType::Int32, {3}});
    auto graph = builder.build({{"out", builder.gather(data, indices, 0)}});

    auto tdata = context->createTensor({cnn::DataType::Float32, {4, 2}, false, true});
    auto tindices = context->createTensor({cnn::DataType::Int32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3, 2}, true, false});

    float dataV[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    int32_t idxV[3] = {2, 0, 3};
    tdata->write(dataV, sizeof(dataV));
    tindices->write(idxV, sizeof(idxV));

    auto fence = context->dispatch(*graph, {{"data", tdata}, {"indices", tindices}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {2, 2, 0, 0, 3, 3};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GpuGenericOps, GatherUint32Indices)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto data = builder.input("data", {cnn::DataType::Float32, {4, 2}});
    auto indices = builder.input("indices", {cnn::DataType::Uint32, {3}});
    auto graph = builder.build({{"out", builder.gather(data, indices, 0)}});

    auto tdata = context->createTensor({cnn::DataType::Float32, {4, 2}, false, true});
    auto tindices = context->createTensor({cnn::DataType::Uint32, {3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3, 2}, true, false});

    float dataV[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    uint32_t idxV[3] = {2, 0, 3};
    tdata->write(dataV, sizeof(dataV));
    tindices->write(idxV, sizeof(idxV));

    auto fence = context->dispatch(*graph, {{"data", tdata}, {"indices", tindices}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    float expected[6] = {2, 2, 0, 0, 3, 3};
    for (int i = 0; i < 6; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GpuGenericOps, Gemm)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 2}});
    auto c = builder.input("c", {cnn::DataType::Float32, {2, 2}});
    auto graph = builder.build({{"out", builder.gemm(a, b, c, 2.0f, 0.5f)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tc = context->createTensor({cnn::DataType::Float32, {2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float av[4] = {1, 2, 3, 4};
    float bv[4] = {1, 0, 0, 1}; // identity
    float cv[4] = {10, 10, 10, 10};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));
    tc->write(cv, sizeof(cv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}, {"c", tc}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    // alpha * (a @ I) + beta * c = 2*a + 0.5*10
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], 2.0f * av[i] + 0.5f * cv[i]);
}

// C as a single row (cElems == N, broadcasting over every output row) — a
// different branch of gemm.comp's cv selection than the full-[M,N] case
// the previous test exercises.
TEST(GpuGenericOps, GemmBroadcastCRow)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3, 2}});
    auto c = builder.input("c", {cnn::DataType::Float32, {2}});
    auto graph = builder.build({{"out", builder.gemm(a, b, c, 1.0f, 1.0f)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3, 2}, false, true});
    auto tc = context->createTensor({cnn::DataType::Float32, {2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 2}, true, false});

    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[6] = {1, 0, 0, 1, 1, 1};
    float cv[2] = {100, 200};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));
    tc->write(cv, sizeof(cv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}, {"c", tc}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    // a @ b: row0 = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]; row1 = [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
    float expected[4] = {4 + 100, 5 + 200, 10 + 100, 11 + 200};
    for (int i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(result[i], expected[i]);
}

TEST(GpuGenericOps, SoftmaxLastAxis)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 3}});
    auto graph = builder.build({{"out", builder.softmax(x, -1)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {2, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 3}, true, false});

    float xv[6] = {1, 2, 3, 1, 1, 1};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));

    for (int row = 0; row < 2; row++)
    {
        float sum = result[row * 3 + 0] + result[row * 3 + 1] + result[row * 3 + 2];
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
    EXPECT_NEAR(result[3], result[4], 1e-5f);
    EXPECT_NEAR(result[4], result[5], 1e-5f);
    EXPECT_LT(result[0], result[1]);
    EXPECT_LT(result[1], result[2]);
}

// Softmax over axis 0 (not the last dim) of a rank-3 tensor — specifically
// exercises softmax.comp's generic outer-decode (skipping a non-last
// dimension when unraveling the workgroup's row index), which a last-axis-
// only test like SoftmaxLastAxis above can't catch a bug in.
TEST(GpuGenericOps, SoftmaxNonLastAxis)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 1, 3}});
    auto graph = builder.build({{"out", builder.softmax(x, 0)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {2, 1, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 1, 3}, true, false});

    // x[0,0,:] = {1,2,3}, x[1,0,:] = {4,4,4} — softmax is taken over the
    // size-2 axis 0, independently for each of the 3 columns.
    float xv[6] = {1, 2, 3, 4, 4, 4};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[6];
    tout->read(result, sizeof(result));
    for (int col = 0; col < 3; col++)
    {
        float e0 = std::exp(xv[col] - std::max(xv[col], xv[3 + col]));
        float e1 = std::exp(xv[3 + col] - std::max(xv[col], xv[3 + col]));
        float expected0 = e0 / (e0 + e1);
        float expected1 = e1 / (e0 + e1);
        EXPECT_NEAR(result[col], expected0, 1e-5f);          // out[0, 0, col]
        EXPECT_NEAR(result[3 + col], expected1, 1e-5f);      // out[1, 0, col]
        EXPECT_NEAR(result[col] + result[3 + col], 1.0f, 1e-5f);
    }
}

TEST(GpuGenericOps, Conv2d)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 3, 3}});
    // OIHW: 1 output channel, 1 input channel, 2x2 kernel.
    float wv[4] = {1, 2, 3, 4};
    auto w = builder.constant({cnn::DataType::Float32, {1, 1, 2, 2}}, wv, sizeof(wv));
    cnn::Conv2dDescriptor desc;
    desc.strideX = 1;
    desc.strideY = 1;
    desc.dilationX = 1;
    desc.dilationY = 1;
    desc.paddingLeft = 0;
    desc.paddingRight = 0;
    desc.paddingTop = 0;
    desc.paddingBottom = 0;
    desc.groups = 1;
    auto graph = builder.build({{"out", builder.conv2d(x, w, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, true, false});

    float xv[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    // Hand-computed: see conv2d.comp comments for the coordinate math.
    EXPECT_FLOAT_EQ(result[0], 37.0f); // 1*1 + 2*2 + 4*3 + 5*4
    EXPECT_FLOAT_EQ(result[1], 47.0f); // 2*1 + 3*2 + 5*3 + 6*4
    EXPECT_FLOAT_EQ(result[2], 67.0f); // 4*1 + 5*2 + 7*3 + 8*4
    EXPECT_FLOAT_EQ(result[3], 77.0f); // 5*1 + 6*2 + 8*3 + 9*4
}

TEST(GpuGenericOps, MaxPool2d)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 4, 4}});
    cnn::Pool2dDescriptor desc;
    desc.kernelHeight = 2;
    desc.kernelWidth = 2;
    desc.strideX = 2;
    desc.strideY = 2;
    auto graph = builder.build({{"out", builder.maxPool2d(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 4, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, true, false});

    float xv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    EXPECT_FLOAT_EQ(result[0], 6.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
    EXPECT_FLOAT_EQ(result[2], 14.0f);
    EXPECT_FLOAT_EQ(result[3], 16.0f);
}

TEST(GpuGenericOps, AvgPool2d)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 4, 4}});
    cnn::Pool2dDescriptor desc;
    desc.kernelHeight = 2;
    desc.kernelWidth = 2;
    desc.strideX = 2;
    desc.strideY = 2;
    auto graph = builder.build({{"out", builder.avgPool2d(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 4, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, true, false});

    float xv[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[4];
    tout->read(result, sizeof(result));
    EXPECT_FLOAT_EQ(result[0], 3.5f);
    EXPECT_FLOAT_EQ(result[1], 5.5f);
    EXPECT_FLOAT_EQ(result[2], 11.5f);
    EXPECT_FLOAT_EQ(result[3], 13.5f);
}

TEST(GpuGenericOps, ResizeBilinearAlignCorners)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 1, 2, 2}});
    cnn::ResizeDescriptor desc;
    desc.outputHeight = 3;
    desc.outputWidth = 3;
    desc.mode = cnn::ResizeMode::Bilinear;
    desc.alignCorners = true;
    auto graph = builder.build({{"out", builder.resize(x, desc)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 1, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 1, 3, 3}, true, false});

    float xv[4] = {1, 2, 3, 4};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[9];
    tout->read(result, sizeof(result));
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 1.5f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 2.0f);
    EXPECT_FLOAT_EQ(result[4], 2.5f);
    EXPECT_FLOAT_EQ(result[5], 3.0f);
    EXPECT_FLOAT_EQ(result[6], 3.0f);
    EXPECT_FLOAT_EQ(result[7], 3.5f);
    EXPECT_FLOAT_EQ(result[8], 4.0f);
}

TEST(GpuGenericOps, BatchNorm)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 2, 2, 2}});
    float meanV[2] = {2.5f, 6.5f};
    float varV[2] = {1.25f, 1.25f};
    float scaleV[2] = {1.0f, 1.0f};
    float biasV[2] = {0.0f, 0.0f};
    auto mean = builder.constant({cnn::DataType::Float32, {2}}, meanV, sizeof(meanV));
    auto var = builder.constant({cnn::DataType::Float32, {2}}, varV, sizeof(varV));
    auto scale = builder.constant({cnn::DataType::Float32, {2}}, scaleV, sizeof(scaleV));
    auto bias = builder.constant({cnn::DataType::Float32, {2}}, biasV, sizeof(biasV));
    auto graph = builder.build({{"out", builder.batchNorm(x, mean, var, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, true, false});

    float xv[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[8];
    tout->read(result, sizeof(result));
    float invStd = 1.0f / std::sqrt(1.25f + 1e-5f);
    for (int c = 0; c < 2; c++)
    {
        float expected[4] = {-1.5f * invStd, -0.5f * invStd, 0.5f * invStd, 1.5f * invStd};
        for (int i = 0; i < 4; i++)
            EXPECT_NEAR(result[c * 4 + i], expected[i], 1e-4f);
    }
}

TEST(GpuGenericOps, InstanceNorm)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 2, 2, 2}});
    float scaleV[2] = {2.0f, 3.0f};
    float biasV[2] = {10.0f, 100.0f};
    auto scale = builder.constant({cnn::DataType::Float32, {2}}, scaleV, sizeof(scaleV));
    auto bias = builder.constant({cnn::DataType::Float32, {2}}, biasV, sizeof(biasV));
    auto graph = builder.build({{"out", builder.instanceNorm(x, scale, bias, 1e-5f)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 2, 2, 2}, true, false});

    float xv[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[8];
    tout->read(result, sizeof(result));
    float invStd = 1.0f / std::sqrt(1.25f + 1e-5f);
    // Channel 0: scale=2, bias=10.
    EXPECT_NEAR(result[0], -1.5f * invStd * 2.0f + 10.0f, 1e-4f);
    EXPECT_NEAR(result[1], -0.5f * invStd * 2.0f + 10.0f, 1e-4f);
    EXPECT_NEAR(result[2], 0.5f * invStd * 2.0f + 10.0f, 1e-4f);
    EXPECT_NEAR(result[3], 1.5f * invStd * 2.0f + 10.0f, 1e-4f);
    // Channel 1: scale=3, bias=100.
    EXPECT_NEAR(result[4], -1.5f * invStd * 3.0f + 100.0f, 1e-4f);
    EXPECT_NEAR(result[5], -0.5f * invStd * 3.0f + 100.0f, 1e-4f);
    EXPECT_NEAR(result[6], 0.5f * invStd * 3.0f + 100.0f, 1e-4f);
    EXPECT_NEAR(result[7], 1.5f * invStd * 3.0f + 100.0f, 1e-4f);
}

TEST(GpuGenericOps, QuantizeLinear)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {5}});
    auto graph = builder.build({{"out", builder.quantizeLinear(x, 0.5f, 0)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {5}, false, true});
    auto tout = context->createTensor({cnn::DataType::Int8, {5}, true, false});

    float xv[5] = {0.0f, 0.5f, 1.0f, -0.5f, 0.9f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    int8_t result[5];
    tout->read(result, sizeof(result));
    // round(x / 0.5): 0, 1, 2, -1, 2 (0.9/0.5=1.8 -> round=2)
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
    EXPECT_EQ(result[3], -1);
    EXPECT_EQ(result[4], 2);
}

TEST(GpuGenericOps, DequantizeLinear)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Int8, {5}});
    auto graph = builder.build({{"out", builder.dequantizeLinear(x, 0.5f, 0)}});

    auto tx = context->createTensor({cnn::DataType::Int8, {5}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {5}, true, false});

    int8_t xv[5] = {0, 1, 2, -1, 2};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[5];
    tout->read(result, sizeof(result));
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 0.5f);
    EXPECT_FLOAT_EQ(result[2], 1.0f);
    EXPECT_FLOAT_EQ(result[3], -0.5f);
    EXPECT_FLOAT_EQ(result[4], 1.0f);
}

TEST(GpuGenericOps, QuantizeDequantizeRoundTrip)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {5}});
    auto q = builder.quantizeLinear(x, 0.5f, 0);
    auto graph = builder.build({{"out", builder.dequantizeLinear(q, 0.5f, 0)}});

    auto tx = context->createTensor({cnn::DataType::Float32, {5}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {5}, true, false});

    float xv[5] = {0.0f, 0.5f, 1.0f, -0.5f, 0.9f};
    tx->write(xv, sizeof(xv));

    auto fence = context->dispatch(*graph, {{"x", tx}}, {{"out", tout}});
    fence->wait();

    float result[5];
    tout->read(result, sizeof(result));
    // Rounding error only: each value was quantized then dequantized.
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 0.5f);
    EXPECT_FLOAT_EQ(result[2], 1.0f);
    EXPECT_FLOAT_EQ(result[3], -0.5f);
    EXPECT_FLOAT_EQ(result[4], 1.0f); // 0.9 rounds to 2 -> 1.0
}

TEST(GpuGenericOps, AddBroadcast)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {3, 4}});
    auto b = builder.input("b", {cnn::DataType::Float32, {4}});
    auto graph = builder.build({{"out", builder.add(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {3, 4}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {3, 4}, true, false});

    float av[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float bv[4] = {10, 20, 30, 40};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[12];
    tout->read(result, sizeof(result));
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
            EXPECT_FLOAT_EQ(result[r * 4 + c], av[r * 4 + c] + bv[c]);
}

TEST(GpuGenericOps, MulBroadcast)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3, 1}});
    auto b = builder.input("b", {cnn::DataType::Float32, {3, 4}});
    auto graph = builder.build({{"out", builder.mul(a, b)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {2, 3, 1}, false, true});
    auto tb = context->createTensor({cnn::DataType::Float32, {3, 4}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {2, 3, 4}, true, false});

    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    ta->write(av, sizeof(av));
    tb->write(bv, sizeof(bv));

    auto fence = context->dispatch(*graph, {{"a", ta}, {"b", tb}}, {{"out", tout}});
    fence->wait();

    float result[24];
    tout->read(result, sizeof(result));
    for (int n = 0; n < 2; n++)
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
            {
                float aVal = av[n * 3 + r];
                float bVal = bv[r * 4 + c];
                EXPECT_FLOAT_EQ(result[(n * 3 + r) * 4 + c], aVal * bVal);
            }
}

TEST(GpuGenericOps, QuantizedMatmul)
{
    auto context = makeGpuGenericContext();
    cnn::GraphBuilder builder(context);
    // activation: 1x3, weight: 3x2 (int8), scale=0.5, zeroPoint=0.
    auto a = builder.input("a", {cnn::DataType::Float32, {1, 3}});
    int8_t wv[6] = {1, 0, 0, 1, 1, 1}; // dequantized: 0.5*I plus third row all 0.5
    auto w = builder.constant({cnn::DataType::Int8, {3, 2}}, wv, sizeof(wv));
    auto graph = builder.build({{"out", builder.quantizedMatmul(a, w, 0.5f, 0)}});

    auto ta = context->createTensor({cnn::DataType::Float32, {1, 3}, false, true});
    auto tout = context->createTensor({cnn::DataType::Float32, {1, 2}, true, false});

    float av[3] = {2.0f, 3.0f, 4.0f};
    ta->write(av, sizeof(av));

    auto fence = context->dispatch(*graph, {{"a", ta}}, {{"out", tout}});
    fence->wait();

    float result[2];
    tout->read(result, sizeof(result));
    // dequantized weight = [[0.5,0],[0,0.5],[0.5,0.5]]
    // out[0] = 2*0.5 + 3*0 + 4*0.5 = 1 + 2 = 3
    // out[1] = 2*0 + 3*0.5 + 4*0.5 = 1.5 + 2 = 3.5
    EXPECT_FLOAT_EQ(result[0], 3.0f);
    EXPECT_FLOAT_EQ(result[1], 3.5f);
}

TEST(GpuGenericOnnx, ImportConvAddRelu)
{
    auto context = makeGpuGenericContext();
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
    float expected[9] = {107, 109, 111, 115, 117, 119, 123, 125, 127};
    for (int i = 0; i < 9; i++)
        EXPECT_FLOAT_EQ(out[i], expected[i]);
}

#include <gtest/gtest.h>
#include "test_helpers.hpp"

TEST(GraphBuilderValidation, AddShapeMismatchThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 3}});
    EXPECT_THROW(builder.add(a, b), std::runtime_error);
}

TEST(GraphBuilderValidation, SoftmaxAxisOutOfRangeThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 2}});
    EXPECT_THROW(builder.softmax(x, 5), std::runtime_error);
}

TEST(GraphBuilderValidation, MatMulInnerDimMismatchThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {2, 3}});
    auto b = builder.input("b", {cnn::DataType::Float32, {4, 2}});
    EXPECT_THROW(builder.matmul(a, b), std::runtime_error);
}

TEST(GraphBuilderValidation, ReshapeElementCountMismatchThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 3}});
    EXPECT_THROW(builder.reshape(x, {4, 2}), std::runtime_error);
}

TEST(GraphBuilderValidation, TransposePermNotPermutationThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {2, 3}});
    EXPECT_THROW(builder.transpose(x, {0, 0}), std::runtime_error);
}

TEST(GraphBuilderValidation, ConcatDimMismatchThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto a = builder.input("a", {cnn::DataType::Float32, {1, 2}});
    auto b = builder.input("b", {cnn::DataType::Float32, {2, 2}});
    EXPECT_THROW(builder.concat({a, b}, 1), std::runtime_error);
}

TEST(GraphBuilderValidation, SliceOutOfBoundsThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {4}});
    EXPECT_THROW(builder.slice(x, {3}, {2}), std::runtime_error);
}

TEST(GraphBuilderValidation, LayerNormScaleSizeMismatchThrows)
{
    auto context = makeCpuContext();
    cnn::GraphBuilder builder(context);
    auto x = builder.input("x", {cnn::DataType::Float32, {1, 4}});
    auto scale = builder.input("scale", {cnn::DataType::Float32, {3}});
    auto bias = builder.input("bias", {cnn::DataType::Float32, {4}});
    EXPECT_THROW(builder.layerNorm(x, scale, bias, 1e-5f), std::runtime_error);
}

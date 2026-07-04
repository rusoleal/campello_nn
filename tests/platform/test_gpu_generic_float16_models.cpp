#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <campello_nn/onnx_importer.hpp>
#include <campello_nn/graph_builder.hpp>
#include <campello_image/image.hpp>
#include "../universal/test_helpers.hpp"

namespace cimg = systems::leal::campello_image;

namespace
{
    // Decodes an image file and resizes it (via campello_nn's own resize op) to
    // `targetSize`x`targetSize`, producing a BGR NCHW Float16 buffer in [0,255]
    // range — matching YuNet's training preprocessing. The internal resize graph
    // runs in Float16 so the preprocessed tensor can feed a Float16 model directly.
    std::vector<uint16_t> loadAndPreprocessFloat16(std::shared_ptr<cnn::Context> context,
                                                    const std::string &path, int64_t targetSize)
    {
        auto img = cimg::Image::fromFile(path.c_str());
        if (!img)
            throw std::runtime_error("campello_nn test: failed to decode image '" + path + "'");
        int64_t W = img->getWidth(), H = img->getHeight();
        const uint8_t *rgba = (const uint8_t *)img->getData();

        std::vector<float> bgr((size_t)(3 * H * W));
        for (int64_t y = 0; y < H; y++)
        {
            for (int64_t x = 0; x < W; x++)
            {
                const uint8_t *px = rgba + (size_t)(y * W + x) * 4;
                int64_t idx = y * W + x;
                bgr[(size_t)(0 * H * W + idx)] = px[2]; // B
                bgr[(size_t)(1 * H * W + idx)] = px[1]; // G
                bgr[(size_t)(2 * H * W + idx)] = px[0]; // R
            }
        }

        cnn::GraphBuilder builder(context);
        auto x = builder.input("x", {cnn::DataType::Float16, {1, 3, H, W}});
        cnn::ResizeDescriptor desc;
        desc.outputHeight = targetSize;
        desc.outputWidth = targetSize;
        desc.mode = cnn::ResizeMode::Bilinear;
        auto graph = builder.build({{"out", builder.resize(x, desc)}});

        auto tin = context->createTensor({cnn::DataType::Float16, {1, 3, H, W}, false, true});
        auto tout = context->createTensor({cnn::DataType::Float16, {1, 3, targetSize, targetSize}, true, false});

        auto bgrH = toHalf(bgr);
        tin->write(bgrH.data(), bgrH.size() * sizeof(uint16_t));
        auto fence = context->dispatch(*graph, {{"x", tin}}, {{"out", tout}});
        fence->wait();

        std::vector<uint16_t> result((size_t)(3 * targetSize * targetSize));
        tout->read(result.data(), result.size() * sizeof(uint16_t));
        return result;
    }

    // YuNet's exported graph already applies sigmoid to both `cls_*` and `obj_*`
    // outputs. The final per-anchor confidence is their product; we report the max
    // over all anchors across the 8/16/32 feature-pyramid scales. All tensors are
    // Float16; results are decoded to float for comparison.
    float maxFaceConfidence(std::shared_ptr<cnn::Context> context, cnn::OnnxImportResult &model,
                            const std::vector<uint16_t> &inputData)
    {
        auto inTensor = context->createTensor(model.inputs.at("input"));
        inTensor->write(inputData.data(), inputData.size() * sizeof(uint16_t));

        std::unordered_map<std::string, std::shared_ptr<cnn::Tensor>> outputs;
        for (auto &kv : model.outputs)
            outputs[kv.first] = context->createTensor(kv.second);

        auto fence = context->dispatch(*model.graph, {{"input", inTensor}}, outputs);
        fence->wait();

        float maxScore = 0.f;
        for (const char *scale : {"8", "16", "32"})
        {
            std::string clsName = std::string("cls_") + scale;
            std::string objName = std::string("obj_") + scale;
            size_t n = 1;
            for (auto d : model.outputs.at(clsName).shape)
                n *= (size_t)d;
            std::vector<uint16_t> clsH(n), objH(n);
            outputs[clsName]->read(clsH.data(), n * sizeof(uint16_t));
            outputs[objName]->read(objH.data(), n * sizeof(uint16_t));
            auto cls = fromHalf(clsH);
            auto obj = fromHalf(objH);
            for (size_t i = 0; i < n; i++)
                maxScore = std::max(maxScore, cls[i] * obj[i]);
        }
        return maxScore;
    }
}

TEST(GpuGenericFloat16Model, YuNetFaceDetection)
{
    auto context = makeGpuGenericContext();
    std::string fixturesDir = CAMPELLO_NN_TEST_FIXTURES_DIR;
    cnn::OnnxImportOptions options;
    options.targetDataType = cnn::DataType::Float16;
    auto model = cnn::importOnnxFromFile(context, fixturesDir + "/yunet_n_320_320.onnx", options);

    auto faceInput = loadAndPreprocessFloat16(context, fixturesDir + "/images/face.jpg", 320);
    auto noFaceInput = loadAndPreprocessFloat16(context, fixturesDir + "/images/no_face.jpg", 320);

    float faceScore = maxFaceConfidence(context, model, faceInput);
    float noFaceScore = maxFaceConfidence(context, model, noFaceInput);

    EXPECT_GT(faceScore, 0.5f) << "expected a confident detection on a real face photo";
    EXPECT_LT(noFaceScore, 0.05f) << "expected no spurious detection on a face-free image";
    EXPECT_GT(faceScore, noFaceScore * 10.f) << "expected a clear margin between the two";
}

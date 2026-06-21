#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <campello_nn/onnx_importer.hpp>
#include <campello_nn/graph_builder.hpp>
#include <campello_image/image.hpp>
#include "test_helpers.hpp"

namespace cimg = systems::leal::campello_image;

namespace
{
    // Decodes an image file and resizes it (via campello_nn's own resize op, not
    // a separate image-processing path) to `targetSize`x`targetSize`, producing a
    // BGR (not RGB!) NCHW Float32 buffer in [0,255] range — matching YuNet's
    // training preprocessing exactly (libfacedetection.train's
    // yunet_train/tasks/face/transforms.py: mean=(0,0,0), std=(1,1,1),
    // to_rgb=False — i.e. no normalization, OpenCV's native BGR order).
    std::vector<float> loadAndPreprocess(std::shared_ptr<cnn::Context> context, const std::string &path, int64_t targetSize)
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
        auto x = builder.input("x", {cnn::DataType::Float32, {1, 3, H, W}});
        cnn::ResizeDescriptor desc;
        desc.outputHeight = targetSize;
        desc.outputWidth = targetSize;
        desc.mode = cnn::ResizeMode::Bilinear;
        auto graph = builder.build({{"out", builder.resize(x, desc)}});

        auto tin = context->createTensor({cnn::DataType::Float32, {1, 3, H, W}, false, true});
        auto tout = context->createTensor({cnn::DataType::Float32, {1, 3, targetSize, targetSize}, true, false});
        tin->write(bgr.data(), bgr.size() * sizeof(float));
        auto fence = context->dispatch(*graph, {{"x", tin}}, {{"out", tout}});
        fence->wait();

        std::vector<float> result((size_t)(3 * targetSize * targetSize));
        tout->read(result.data(), result.size() * sizeof(float));
        return result;
    }

    // YuNet's exported graph already applies sigmoid to both `cls_*` and `obj_*`
    // outputs (confirmed: the .onnx file's own node list includes Sigmoid nodes
    // feeding them). The final per-anchor confidence is their product — straight
    // from libfacedetection.train's yunet_train/tasks/face/postprocess.py:
    // `scores = max_scores * flatten_objectness`. We report the max over every
    // anchor across all 3 feature-pyramid scales (8/16/32) as "is there a face
    // anywhere in this image".
    float maxFaceConfidence(std::shared_ptr<cnn::Context> context, cnn::OnnxImportResult &model,
                             const std::vector<float> &inputData)
    {
        auto inTensor = context->createTensor(model.inputs.at("input"));
        inTensor->write(inputData.data(), inputData.size() * sizeof(float));

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
            std::vector<float> cls(n), obj(n);
            outputs[clsName]->read(cls.data(), n * sizeof(float));
            outputs[objName]->read(obj.data(), n * sizeof(float));
            for (size_t i = 0; i < n; i++)
                maxScore = std::max(maxScore, cls[i] * obj[i]);
        }
        return maxScore;
    }
}

// End-to-end validation of the whole stack on a real, standard, pretrained
// model: ONNX import (protobuf parsing + op mapping, including Conv-with-fused-
// bias, Transpose, Reshape with -1 inference, MaxPool, and Resize with
// nearest_mode=floor — all added specifically to get this model importing) and
// real image decoding via campello_image, run through the CPU backend.
//
// Measured values on the actual fixtures (see tests/fixtures/images/NOTICE.md):
// face.jpg scores ~0.83, no_face.jpg scores ~0.0005 — a ~1800x margin. Bit-
// identical results were also confirmed on the real MPSGraph/GPU backend
// during development; this test only exercises CPU since it's universal.
TEST(YuNetFaceDetection, DetectsFaceMuchMoreConfidentlyThanNoFace)
{
    auto context = makeCpuContext();
    std::string fixturesDir = CAMPELLO_NN_TEST_FIXTURES_DIR;
    auto model = cnn::importOnnxFromFile(context, fixturesDir + "/yunet_n_320_320.onnx");

    auto faceInput = loadAndPreprocess(context, fixturesDir + "/images/face.jpg", 320);
    auto noFaceInput = loadAndPreprocess(context, fixturesDir + "/images/no_face.jpg", 320);

    float faceScore = maxFaceConfidence(context, model, faceInput);
    float noFaceScore = maxFaceConfidence(context, model, noFaceInput);

    EXPECT_GT(faceScore, 0.5f) << "expected a confident detection on a real face photo";
    EXPECT_LT(noFaceScore, 0.05f) << "expected no spurious detection on a face-free image";
    EXPECT_GT(faceScore, noFaceScore * 10.f) << "expected a clear margin between the two";
}

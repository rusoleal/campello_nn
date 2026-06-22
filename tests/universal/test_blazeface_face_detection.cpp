#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <campello_nn/tflite_importer.hpp>
#include <campello_nn/graph_builder.hpp>
#include <campello_image/image.hpp>
#include "test_helpers.hpp"

namespace cimg = systems::leal::campello_image;

namespace
{
    // Decodes an image file and resizes it (via campello_nn's own resize op)
    // directly to 128x128 (a plain stretch, not MediaPipe's actual
    // aspect-ratio-preserving letterbox — measured empirically to give a
    // *larger* face-vs-no-face margin on these particular fixtures than
    // letterboxing with zero borders does, since letterboxing shrinks the
    // already-modest face in face.jpg (800x551) well below this short-range
    // model's expected close-up framing; faithfully reproducing MediaPipe's
    // own letterbox isn't the goal here, just a reliable regression signal).
    // Produces an RGB (not BGR — MediaPipe's ImageToTensorCalculator operates
    // on RGB ImageFrame data, unlike OpenCV's native BGR that YuNet's training
    // pipeline used) NCHW Float32 buffer normalized to [-1, 1] — confirmed
    // against MediaPipe's actual graph config (face_detection.pbtxt's
    // ImageToTensorCalculatorOptions: `output_tensor_float_range { min: -1.0
    // max: 1.0 }`, i.e. value = pixel/127.5 - 1.0).
    std::vector<float> loadAndPreprocess(std::shared_ptr<cnn::Context> context, const std::string &path)
    {
        auto img = cimg::Image::fromFile(path.c_str());
        if (!img)
            throw std::runtime_error("campello_nn test: failed to decode image '" + path + "'");
        int64_t W = img->getWidth(), H = img->getHeight();
        const uint8_t *rgba = (const uint8_t *)img->getData();

        std::vector<float> rgb((size_t)(3 * H * W));
        for (int64_t y = 0; y < H; y++)
        {
            for (int64_t x = 0; x < W; x++)
            {
                const uint8_t *px = rgba + (size_t)(y * W + x) * 4;
                int64_t idx = y * W + x;
                rgb[(size_t)(0 * H * W + idx)] = px[0]; // R
                rgb[(size_t)(1 * H * W + idx)] = px[1]; // G
                rgb[(size_t)(2 * H * W + idx)] = px[2]; // B
            }
        }

        const int64_t targetSize = 128;
        cnn::GraphBuilder builder(context);
        auto x = builder.input("x", {cnn::DataType::Float32, {1, 3, H, W}});
        cnn::ResizeDescriptor desc;
        desc.outputHeight = targetSize;
        desc.outputWidth = targetSize;
        desc.mode = cnn::ResizeMode::Bilinear;
        auto graph = builder.build({{"out", builder.resize(x, desc)}});

        auto tin = context->createTensor({cnn::DataType::Float32, {1, 3, H, W}, false, true});
        auto tout = context->createTensor({cnn::DataType::Float32, {1, 3, targetSize, targetSize}, true, false});
        tin->write(rgb.data(), rgb.size() * sizeof(float));
        auto fence = context->dispatch(*graph, {{"x", tin}}, {{"out", tout}});
        fence->wait();

        std::vector<float> result((size_t)(3 * targetSize * targetSize));
        tout->read(result.data(), result.size() * sizeof(float));
        for (auto &v : result)
            v = v / 127.5f - 1.0f;
        return result;
    }

    float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Unlike YuNet's exported ONNX graph (which already applies Sigmoid before
    // its cls_*/obj_* outputs), BlazeFace's exported TFLite graph stops at raw
    // logits — confirmed by tflite_importer.cpp's import: `classificators`'
    // producing CONV_2D has no LOGISTIC op between it and the final
    // RESHAPE/CONCATENATION. Sigmoid + anchor decoding + NMS are normally a
    // separate MediaPipe calculator stage, out of scope for campello_nn itself
    // — we only need "is there a confident face detection anywhere", so apply
    // sigmoid here and take the max over all 896 anchors.
    float maxFaceConfidence(std::shared_ptr<cnn::Context> context, cnn::TfliteImportResult &model,
                             const std::vector<float> &inputData)
    {
        auto inTensor = context->createTensor(model.inputs.at("input"));
        inTensor->write(inputData.data(), inputData.size() * sizeof(float));

        std::unordered_map<std::string, std::shared_ptr<cnn::Tensor>> outputs;
        for (auto &kv : model.outputs)
            outputs[kv.first] = context->createTensor(kv.second);

        auto fence = context->dispatch(*model.graph, {{"input", inTensor}}, outputs);
        fence->wait();

        size_t n = 1;
        for (auto d : model.outputs.at("classificators").shape)
            n *= (size_t)d;
        std::vector<float> cls(n);
        outputs["classificators"]->read(cls.data(), n * sizeof(float));

        float maxScore = -1.0f;
        for (size_t i = 0; i < n; i++)
            maxScore = std::max(maxScore, sigmoid(cls[i]));
        return maxScore;
    }
}

// End-to-end validation of the TFLite import stack on a second, independent
// real model (the first being YuNet/ONNX — see test_yunet_face_detection.cpp):
// MediaPipe's BlazeFace short-range face detector. Specifically exercises
// DEPTHWISE_CONV_2D, PAD (mapped to concat-with-zeros), and float16-weight
// DEQUANTIZE — none of which YuNet's graph needed, all added/fixed
// specifically to get this model importing and running correctly.
//
// Measured values on the actual fixtures (see tests/fixtures/images/NOTICE.md):
// face.jpg scores ~0.47, no_face.jpg scores ~0.23 — a modest but real and
// repeatable margin, not YuNet's dramatic ~1800x one. This isn't a precision
// regression: face.jpg (a 1911 portrait, not a close-up selfie) is framed much
// more conservatively than this short-range model's training distribution
// (front-facing phone camera, face filling most of the frame), and this test
// deliberately skips MediaPipe's own letterbox preprocessing (see
// loadAndPreprocess()'s comment) plus the real product's anchor-decode/NMS/
// min_score_thresh=0.5 pipeline — it only checks the raw max per-anchor
// sigmoid score, a rougher signal than an actual reported detection.
TEST(BlazeFaceFaceDetection, DetectsFaceMuchMoreConfidentlyThanNoFace)
{
    auto context = makeCpuContext();
    std::string fixturesDir = CAMPELLO_NN_TEST_FIXTURES_DIR;
    auto model = cnn::importTfliteFromFile(context, fixturesDir + "/blaze_face_short_range.tflite");

    auto faceInput = loadAndPreprocess(context, fixturesDir + "/images/face.jpg");
    auto noFaceInput = loadAndPreprocess(context, fixturesDir + "/images/no_face.jpg");

    float faceScore = maxFaceConfidence(context, model, faceInput);
    float noFaceScore = maxFaceConfidence(context, model, noFaceInput);

    EXPECT_GT(faceScore, 0.4f) << "expected a meaningfully confident detection on a real face photo";
    EXPECT_LT(noFaceScore, 0.3f) << "expected a clearly lower score on a face-free image";
    EXPECT_GT(faceScore, noFaceScore * 1.5f) << "expected a clear margin between the two";
}

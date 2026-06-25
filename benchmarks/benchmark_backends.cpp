// Benchmarks campello_nn backends (Cpu, GpuGeneric, and platform-native Gpu)
// on two workloads:
//   1. Synthetic transformer block: matmul -> add -> gelu -> layerNorm
//   2. Real model: YuNet face detection (ONNX import + image decode/resize)
//
// Build with -DBUILD_BENCHMARKS=ON, then run:
//   ./build/benchmarks/campello_nn_benchmark [batch] [hidden]
//
// The YuNet workload is included when CAMPELLO_NN_TEST_FIXTURES_DIR is defined
// (i.e., when campello_image is available).

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <campello_nn/context.hpp>
#include <campello_nn/graph_builder.hpp>

#ifdef CAMPELLO_NN_TEST_FIXTURES_DIR
#include <campello_image/image.hpp>
#include <campello_nn/onnx_importer.hpp>
#endif

namespace cnn = systems::leal::campello_nn;

namespace
{
    using Clock = std::chrono::steady_clock;
    using DurationMs = std::chrono::duration<double, std::milli>;

    struct Stats
    {
        double min = 0.0;
        double median = 0.0;
        double mean = 0.0;
        double max = 0.0;
    };

    Stats computeStats(std::vector<double> samples)
    {
        if (samples.empty())
            return {};
        std::sort(samples.begin(), samples.end());
        Stats s;
        s.min = samples.front();
        s.max = samples.back();
        s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        if (samples.size() % 2 == 1)
            s.median = samples[samples.size() / 2];
        else
            s.median = (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]) * 0.5;
        return s;
    }

    const char *deviceTypeName(cnn::DeviceType type)
    {
        switch (type)
        {
        case cnn::DeviceType::Cpu:
            return "Cpu";
        case cnn::DeviceType::Gpu:
            return "Gpu (MPSGraph)";
        case cnn::DeviceType::GpuGeneric:
            return "GpuGeneric";
        default:
            return "Unknown";
        }
    }

    struct BackendResult
    {
        cnn::DeviceType type;
        Stats stats;
        double maxAbsDiff = 0.0;
        bool available = false;
    };

    // -------------------------------------------------------------------------
    // Timer helpers
    // -------------------------------------------------------------------------
    Stats timeIterations(int warmUp, int timed, const std::function<void()> &fn)
    {
        for (int i = 0; i < warmUp; ++i)
            fn();

        std::vector<double> times;
        times.reserve(timed);
        for (int i = 0; i < timed; ++i)
        {
            auto start = Clock::now();
            fn();
            auto end = Clock::now();
            times.push_back(DurationMs(end - start).count());
        }
        return computeStats(times);
    }

    // -------------------------------------------------------------------------
    // Workload 1: synthetic transformer block
    // -------------------------------------------------------------------------
    std::vector<float> makeRandomBuffer(size_t count, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> buf(count);
        for (size_t i = 0; i < count; ++i)
            buf[i] = dist(rng);
        return buf;
    }

    struct TransformerGraph
    {
        std::shared_ptr<cnn::Context> context;
        std::shared_ptr<cnn::Graph> graph;
        std::shared_ptr<cnn::Tensor> tx;
        std::shared_ptr<cnn::Tensor> tw;
        std::shared_ptr<cnn::Tensor> tbias;
        std::shared_ptr<cnn::Tensor> tscale;
        std::shared_ptr<cnn::Tensor> tlnBias;
        std::shared_ptr<cnn::Tensor> tout;
    };

    TransformerGraph buildTransformerGraph(std::shared_ptr<cnn::Context> context,
                                           int64_t batch, int64_t hidden)
    {
        TransformerGraph g;
        g.context = context;

        cnn::GraphBuilder builder(context);
        auto x = builder.input("x", {cnn::DataType::Float32, {batch, hidden}});
        auto w = builder.input("w", {cnn::DataType::Float32, {hidden, hidden}});
        auto bias = builder.input("bias", {cnn::DataType::Float32, {1, hidden}});
        auto scale = builder.input("scale", {cnn::DataType::Float32, {hidden}});
        auto lnBias = builder.input("lnBias", {cnn::DataType::Float32, {hidden}});

        auto linear = builder.add(builder.matmul(x, w), bias);
        auto activated = builder.gelu(linear);
        auto out = builder.layerNorm(activated, scale, lnBias, 1e-5f);
        g.graph = builder.build({{"out", out}});

        g.tx = context->createTensor({cnn::DataType::Float32, {batch, hidden}, false, true});
        g.tw = context->createTensor({cnn::DataType::Float32, {hidden, hidden}, false, true});
        g.tbias = context->createTensor({cnn::DataType::Float32, {1, hidden}, false, true});
        g.tscale = context->createTensor({cnn::DataType::Float32, {hidden}, false, true});
        g.tlnBias = context->createTensor({cnn::DataType::Float32, {hidden}, false, true});
        g.tout = context->createTensor({cnn::DataType::Float32, {batch, hidden}, true, false});

        auto xv = makeRandomBuffer(static_cast<size_t>(batch * hidden), 0x12345678);
        auto wv = makeRandomBuffer(static_cast<size_t>(hidden * hidden), 0x23456789);
        auto biasV = makeRandomBuffer(static_cast<size_t>(hidden), 0x3456789a);
        auto scaleV = makeRandomBuffer(static_cast<size_t>(hidden), 0x456789ab);
        auto lnBiasV = makeRandomBuffer(static_cast<size_t>(hidden), 0x56789abc);

        g.tx->write(xv.data(), xv.size() * sizeof(float));
        g.tw->write(wv.data(), wv.size() * sizeof(float));
        g.tbias->write(biasV.data(), biasV.size() * sizeof(float));
        g.tscale->write(scaleV.data(), scaleV.size() * sizeof(float));
        g.tlnBias->write(lnBiasV.data(), lnBiasV.size() * sizeof(float));

        return g;
    }

    void runTransformerBlockOnce(TransformerGraph &g)
    {
        auto fence = g.context->dispatch(
            *g.graph,
            {{"x", g.tx}, {"w", g.tw}, {"bias", g.tbias},
             {"scale", g.tscale}, {"lnBias", g.tlnBias}},
            {{"out", g.tout}});
        fence->wait();
    }

    std::vector<float> readTransformerOutput(TransformerGraph &g)
    {
        std::vector<float> out(static_cast<size_t>(g.tout->shape()[0] *
                                                     g.tout->shape()[1]));
        g.tout->read(out.data(), out.size() * sizeof(float));
        return out;
    }

    void benchmarkTransformerBlock(int64_t batch, int64_t hidden)
    {
        std::printf("\n=== Transformer block (matmul + add + gelu + layerNorm) ===\n");
        std::printf("  Input shape: [%lld, %lld]  Weight shape: [%lld, %lld]\n",
                    batch, hidden, hidden, hidden);
        std::printf("  Warm-up: 3, Timed iterations: 10\n\n");

        // CPU reference: timed run + captured output.
        std::vector<float> referenceOutput;
        Stats cpuStats;
        {
            auto context = cnn::Context::create({cnn::DeviceType::Cpu});
            auto g = buildTransformerGraph(context, batch, hidden);
            cpuStats = timeIterations(3, 10, [&]()
                                      {
                                          runTransformerBlockOnce(g);
                                          referenceOutput = readTransformerOutput(g);
                                      });
        }

        std::vector<BackendResult> results;
        results.push_back({cnn::DeviceType::Cpu, cpuStats, 0.0, true});

        auto tryGpuGeneric = [&]() -> BackendResult
        {
            BackendResult r{cnn::DeviceType::GpuGeneric, {}, 0.0, false};
            try
            {
                auto context = cnn::Context::create({cnn::DeviceType::GpuGeneric});
                auto g = buildTransformerGraph(context, batch, hidden);
                r.stats = timeIterations(3, 10, [&]()
                                         { runTransformerBlockOnce(g); });
                auto out = readTransformerOutput(g);
                r.maxAbsDiff = 0.0;
                for (size_t i = 0; i < out.size(); ++i)
                    r.maxAbsDiff = std::max(r.maxAbsDiff,
                                            static_cast<double>(std::abs(out[i] - referenceOutput[i])));
                r.available = true;
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "GpuGeneric unavailable: %s\n", e.what());
            }
            return r;
        };
        results.push_back(tryGpuGeneric());

#ifdef __APPLE__
        auto tryGpu = [&]() -> BackendResult
        {
            BackendResult r{cnn::DeviceType::Gpu, {}, 0.0, false};
            try
            {
                auto context = cnn::Context::create({cnn::DeviceType::Gpu});
                auto g = buildTransformerGraph(context, batch, hidden);
                r.stats = timeIterations(3, 10, [&]()
                                         { runTransformerBlockOnce(g); });
                auto out = readTransformerOutput(g);
                r.maxAbsDiff = 0.0;
                for (size_t i = 0; i < out.size(); ++i)
                    r.maxAbsDiff = std::max(r.maxAbsDiff,
                                            static_cast<double>(std::abs(out[i] - referenceOutput[i])));
                r.available = true;
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "Gpu unavailable: %s\n", e.what());
            }
            return r;
        }();
        results.push_back(tryGpu);
#endif

        std::printf("%-18s %10s  %10s  %10s  %10s  %12s\n",
                    "Backend", "min", "median", "mean", "max", "maxAbsDiff");
        std::printf("%-18s %10s  %10s  %10s  %10s  %12s\n",
                    "-------", "---", "------", "----", "---", "------------");
        for (const auto &r : results)
        {
            if (!r.available)
            {
                std::printf("%-18s %10s  %10s  %10s  %10s  %12s\n",
                            deviceTypeName(r.type), "N/A", "N/A", "N/A", "N/A", "N/A");
                continue;
            }
            std::printf("%-18s %9.3f ms %9.3f ms %9.3f ms %9.3f ms %11.3e\n",
                        deviceTypeName(r.type),
                        r.stats.min, r.stats.median, r.stats.mean, r.stats.max,
                        r.maxAbsDiff);
        }

        for (const auto &r : results)
        {
            if (!r.available)
                continue;
            double maxVal = 0.0;
            for (float v : referenceOutput)
                maxVal = std::max(maxVal, static_cast<double>(std::abs(v)));
            bool close = r.maxAbsDiff <= 1e-4 || (maxVal > 0.0 && r.maxAbsDiff / maxVal <= 1e-4);
            if (!close)
            {
                std::fprintf(stderr,
                             "ERROR: backend %s deviates from Cpu reference "
                             "(maxAbsDiff=%.3e, maxVal=%.3e)\n",
                             deviceTypeName(r.type), r.maxAbsDiff, maxVal);
            }
        }
    }

#ifdef CAMPELLO_NN_TEST_FIXTURES_DIR
    // -------------------------------------------------------------------------
    // Workload 2: YuNet face detection end-to-end
    // -------------------------------------------------------------------------
    namespace cimg = systems::leal::campello_image;

    std::vector<float> loadAndPreprocessImage(std::shared_ptr<cnn::Context> context,
                                              const std::string &path, int64_t targetSize)
    {
        auto img = cimg::Image::fromFile(path.c_str());
        if (!img)
            throw std::runtime_error("benchmark: failed to decode image '" + path + "'");
        int64_t W = img->getWidth(), H = img->getHeight();
        const uint8_t *rgba = (const uint8_t *)img->getData();

        std::vector<float> bgr(static_cast<size_t>(3 * H * W));
        for (int64_t y = 0; y < H; y++)
        {
            for (int64_t x = 0; x < W; x++)
            {
                const uint8_t *px = rgba + static_cast<size_t>(y * W + x) * 4;
                size_t idx = static_cast<size_t>(y * W + x);
                bgr[(0 * H * W) + idx] = px[2]; // B
                bgr[(1 * H * W) + idx] = px[1]; // G
                bgr[(2 * H * W) + idx] = px[0]; // R
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
        auto tout = context->createTensor({cnn::DataType::Float32,
                                           {1, 3, targetSize, targetSize}, true, false});
        tin->write(bgr.data(), bgr.size() * sizeof(float));
        auto fence = context->dispatch(*graph, {{"x", tin}}, {{"out", tout}});
        fence->wait();

        std::vector<float> result(static_cast<size_t>(3 * targetSize * targetSize));
        tout->read(result.data(), result.size() * sizeof(float));
        return result;
    }

    float maxFaceConfidence(std::shared_ptr<cnn::Context> context,
                            cnn::OnnxImportResult &model,
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
                n *= static_cast<size_t>(d);
            std::vector<float> cls(n), obj(n);
            outputs[clsName]->read(cls.data(), n * sizeof(float));
            outputs[objName]->read(obj.data(), n * sizeof(float));
            for (size_t i = 0; i < n; i++)
                maxScore = std::max(maxScore, cls[i] * obj[i]);
        }
        return maxScore;
    }

    void benchmarkYuNet()
    {
        std::printf("\n=== YuNet face detection (320x320 ONNX) ===\n");
        std::printf("  Warm-up: 3, Timed iterations: 10\n\n");

        std::string fixturesDir = CAMPELLO_NN_TEST_FIXTURES_DIR;

        // Preprocess images once on CPU (not part of the model timing).
        std::vector<float> faceInput;
        std::vector<float> noFaceInput;
        {
            auto cpuContext = cnn::Context::create({cnn::DeviceType::Cpu});
            faceInput = loadAndPreprocessImage(cpuContext, fixturesDir + "/images/face.jpg", 320);
            noFaceInput = loadAndPreprocessImage(cpuContext, fixturesDir + "/images/no_face.jpg", 320);
        }

        // Compute CPU reference confidence scores once.
        float faceRef = 0.f;
        float noFaceRef = 0.f;
        {
            auto cpuContext = cnn::Context::create({cnn::DeviceType::Cpu});
            auto model = cnn::importOnnxFromFile(cpuContext, fixturesDir + "/yunet_n_320_320.onnx");
            faceRef = maxFaceConfidence(cpuContext, model, faceInput);
            noFaceRef = maxFaceConfidence(cpuContext, model, noFaceInput);
            std::printf("  CPU reference confidence: face=%.4f, no_face=%.4f\n\n",
                        faceRef, noFaceRef);
        }

        auto benchmarkBackendYuNet = [&](cnn::DeviceType type,
                                         float faceRef, float noFaceRef) -> BackendResult
        {
            BackendResult r{type, {}, 0.0, false};
            try
            {
                auto context = cnn::Context::create({type});
                auto model = cnn::importOnnxFromFile(context, fixturesDir + "/yunet_n_320_320.onnx");

                r.stats = timeIterations(3, 10, [&]()
                                         {
                                             // Time one face + one no-face inference per iteration.
                                             maxFaceConfidence(context, model, faceInput);
                                             maxFaceConfidence(context, model, noFaceInput);
                                         });

                float faceOut = maxFaceConfidence(context, model, faceInput);
                float noFaceOut = maxFaceConfidence(context, model, noFaceInput);
                r.maxAbsDiff = std::max(static_cast<double>(std::abs(faceOut - faceRef)),
                                        static_cast<double>(std::abs(noFaceOut - noFaceRef)));
                r.available = true;
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "%s unavailable: %s\n", deviceTypeName(type), e.what());
            }
            return r;
        };

        std::vector<BackendResult> results;
        results.push_back(benchmarkBackendYuNet(cnn::DeviceType::Cpu, faceRef, noFaceRef));
        results.push_back(benchmarkBackendYuNet(cnn::DeviceType::GpuGeneric, faceRef, noFaceRef));
#ifdef __APPLE__
        results.push_back(benchmarkBackendYuNet(cnn::DeviceType::Gpu, faceRef, noFaceRef));
#endif

        std::printf("%-18s %10s  %10s  %10s  %10s  %12s\n",
                    "Backend", "min", "median", "mean", "max", "maxScoreDiff");
        std::printf("%-18s %10s  %10s  %10s  %10s  %12s\n",
                    "-------", "---", "------", "----", "---", "------------");
        for (const auto &r : results)
        {
            if (!r.available)
            {
                std::printf("%-18s %10s  %10s  %10s  %10s  %12s\n",
                            deviceTypeName(r.type), "N/A", "N/A", "N/A", "N/A", "N/A");
                continue;
            }
            std::printf("%-18s %9.3f ms %9.3f ms %9.3f ms %9.3f ms %11.3e\n",
                        deviceTypeName(r.type),
                        r.stats.min, r.stats.median, r.stats.mean, r.stats.max,
                        r.maxAbsDiff);
        }
    }
#endif // CAMPELLO_NN_TEST_FIXTURES_DIR
}

int main(int argc, char **argv)
{
    int64_t batch = 1;
    int64_t hidden = 512;
    if (argc > 1)
        batch = std::atoll(argv[1]);
    if (argc > 2)
        hidden = std::atoll(argv[2]);

    if (batch <= 0 || hidden <= 0)
    {
        std::fprintf(stderr, "Usage: %s [batch] [hidden]\n", argv[0]);
        return 1;
    }

    benchmarkTransformerBlock(batch, hidden);

#ifdef CAMPELLO_NN_TEST_FIXTURES_DIR
    benchmarkYuNet();
#else
    std::printf("\nYuNet workload skipped (campello_image not available).\n");
#endif

    return 0;
}

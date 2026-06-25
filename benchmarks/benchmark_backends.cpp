// Benchmarks campello_nn backends (Cpu, GpuGeneric, and platform-native Gpu)
// on a synthetic transformer block: matmul -> add -> gelu -> layerNorm.
//
// Build with -DBUILD_BENCHMARKS=ON, then run:
//   ./build/benchmarks/campello_nn_benchmark [batch] [hidden]

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

    void printStats(const char *label, const Stats &s)
    {
        std::printf("%-18s %8.3f ms %8.3f ms %8.3f ms %8.3f ms\n",
                    label, s.min, s.median, s.mean, s.max);
    }

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
        std::vector<float> outputHost;
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

        g.outputHost.resize(static_cast<size_t>(batch * hidden));
        return g;
    }

    bool runOnce(TransformerGraph &g)
    {
        auto fence = g.context->dispatch(
            *g.graph,
            {{"x", g.tx}, {"w", g.tw}, {"bias", g.tbias},
             {"scale", g.tscale}, {"lnBias", g.tlnBias}},
            {{"out", g.tout}});
        fence->wait();
        g.tout->read(g.outputHost.data(), g.outputHost.size() * sizeof(float));
        return true;
    }

    Stats benchmarkBackend(std::shared_ptr<cnn::Context> context,
                           int64_t batch, int64_t hidden,
                           const std::vector<float> *referenceOutput,
                           double *outAbsMaxDiff)
    {
        TransformerGraph g = buildTransformerGraph(context, batch, hidden);

        // Warm-up.
        for (int i = 0; i < 3; ++i)
            if (!runOnce(g))
                return {};

        const int kTimedIterations = 10;
        std::vector<double> times;
        times.reserve(kTimedIterations);

        for (int i = 0; i < kTimedIterations; ++i)
        {
            auto start = Clock::now();
            if (!runOnce(g))
                return {};
            auto end = Clock::now();
            times.push_back(DurationMs(end - start).count());
        }

        if (referenceOutput && outAbsMaxDiff)
        {
            double maxDiff = 0.0;
            for (size_t i = 0; i < g.outputHost.size(); ++i)
                maxDiff = std::max(maxDiff, static_cast<double>(
                    std::abs(g.outputHost[i] - (*referenceOutput)[i])));
            *outAbsMaxDiff = maxDiff;
        }

        if (referenceOutput == nullptr)
        {
            // This is the reference run; we can't mutate the caller's pointer
            // through the const pointer, but the caller passes nullptr for
            // reference and uses the returned graph's output separately.
        }

        return computeStats(times);
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

    BackendResult tryBenchmarkBackend(cnn::DeviceType type,
                                      int64_t batch, int64_t hidden,
                                      const std::vector<float> &referenceOutput)
    {
        BackendResult r;
        r.type = type;
        try
        {
            auto context = cnn::Context::create({type});
            r.stats = benchmarkBackend(context, batch, hidden,
                                       &referenceOutput, &r.maxAbsDiff);
            r.available = true;
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "Backend %s unavailable: %s\n",
                         deviceTypeName(type), e.what());
        }
        return r;
    }
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

    std::printf("Benchmark: transformer block (matmul + add + gelu + layerNorm)\n");
    std::printf("  Input shape: [%lld, %lld]  Weight shape: [%lld, %lld]\n",
                batch, hidden, hidden, hidden);
    std::printf("  Warm-up: 3, Timed iterations: 10\n\n");

    // Use CPU as the reference output for correctness comparison.
    std::vector<float> referenceOutput;
    Stats cpuStats;
    {
        std::printf("Running reference on Cpu...\n");
        auto cpuContext = cnn::Context::create({cnn::DeviceType::Cpu});
        auto g = buildTransformerGraph(cpuContext, batch, hidden);
        for (int i = 0; i < 3; ++i)
            runOnce(g);
        {
            auto start = Clock::now();
            runOnce(g);
            auto end = Clock::now();
            cpuStats = computeStats({DurationMs(end - start).count()});
        }
        // Run the full timed set on CPU to get stable numbers and populate reference.
        cpuStats = benchmarkBackend(cpuContext, batch, hidden, nullptr, nullptr);
        referenceOutput = std::move(g.outputHost);
    }

    std::vector<BackendResult> results;
    results.push_back({cnn::DeviceType::Cpu, cpuStats, 0.0, true});
    results.push_back(tryBenchmarkBackend(cnn::DeviceType::GpuGeneric, batch, hidden, referenceOutput));

#ifdef __APPLE__
    results.push_back(tryBenchmarkBackend(cnn::DeviceType::Gpu, batch, hidden, referenceOutput));
#endif

    std::printf("\n%-18s %10s  %10s  %10s  %10s  %12s\n",
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

    bool allClose = true;
    for (const auto &r : results)
    {
        if (!r.available)
            continue;
        // Tolerance: gelu/layerNorm are not bit-identical across backends, so
        // use a loose relative tolerance. For fp32 this is ~1e-4 relative.
        double maxVal = 0.0;
        for (float v : referenceOutput)
            maxVal = std::max(maxVal, static_cast<double>(std::abs(v)));
        double relTol = 1e-4;
        double absTol = 1e-4;
        bool close = r.maxAbsDiff <= absTol || (maxVal > 0.0 && r.maxAbsDiff / maxVal <= relTol);
        if (!close)
        {
            std::fprintf(stderr,
                         "ERROR: backend %s deviates from Cpu reference "
                         "(maxAbsDiff=%.3e, maxVal=%.3e)\n",
                         deviceTypeName(r.type), r.maxAbsDiff, maxVal);
            allClose = false;
        }
    }

    return allClose ? 0 : 1;
}

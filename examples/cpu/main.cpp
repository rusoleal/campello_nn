// Minimal campello_nn example: builds a one-op graph (GELU) and runs it on
// the CPU reference backend. No LLM, no accelerator backend — just the
// Context/GraphBuilder/Tensor/Fence lifecycle from NN_ARCHITECTURE.md §3.
#include <cstdio>
#include <campello_nn/context.hpp>
#include <campello_nn/graph_builder.hpp>

using namespace systems::leal::campello_nn;

int main()
{
    auto context = Context::create({DeviceType::Cpu});

    GraphBuilder builder(context);
    auto x = builder.input("x", {DataType::Float32, {1, 4}});
    auto out = builder.gelu(x);
    auto graph = builder.build({{"out", out}});

    auto input = context->createTensor({DataType::Float32, {1, 4}, false, true});
    auto output = context->createTensor({DataType::Float32, {1, 4}, true, false});

    float data[4] = {-2.0f, -0.5f, 0.5f, 2.0f};
    input->write(data, sizeof(data));

    auto fence = context->dispatch(*graph, {{"x", input}}, {{"out", output}});
    fence->wait();

    float result[4];
    output->read(result, sizeof(result));

    std::printf("gelu([-2, -0.5, 0.5, 2]) = [%.4f, %.4f, %.4f, %.4f]\n",
                result[0], result[1], result[2], result[3]);
    return 0;
}

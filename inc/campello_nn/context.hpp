#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <campello_nn/constants/device_type.hpp>
#include <campello_nn/descriptors/context_descriptor.hpp>
#include <campello_nn/descriptors/tensor_descriptor.hpp>
#include <campello_nn/tensor.hpp>
#include <campello_nn/graph.hpp>
#include <campello_nn/fence.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Selects and owns a compute backend — the entry point for campello_nn.
     *
     * A `Context` is the analog of `campello_gpu::Device`: it is the factory for
     * `Tensor`s and the executor of compiled `Graph`s.
     *
     * @code
     * auto context = Context::create({ DeviceType::Cpu });
     * auto input   = context->createTensor({ DataType::Float32, {1, 4}, false, true });
     * input->write(data, sizeof(data));
     *
     * GraphBuilder builder(context);
     * auto x   = builder.input("x", { DataType::Float32, {1, 4} });
     * auto out = builder.gelu(x);
     * auto graph = builder.build({ {"out", out} });
     *
     * auto output = context->createTensor({ DataType::Float32, {1, 4}, true, false });
     * auto fence  = context->dispatch(*graph, {{"x", input}}, {{"out", output}});
     * fence->wait();
     * @endcode
     */
    class Context : public std::enable_shared_from_this<Context>
    {
        friend class GraphBuilder;
        void *native;

        Context(void *pd);

    public:
        ~Context();

        static std::shared_ptr<Context> create(const ContextDescriptor &desc);

        /**
         * @brief The `DeviceType` this context was created with (`ContextDescriptor::deviceType`).
         *
         * Not necessarily the same as which concrete backend ended up handling it —
         * see `Context::create()`'s platform-selection comments (e.g. `Gpu`/`Npu`/
         * `Default` all route to the same MPSGraph backend on Apple platforms) — but
         * it's the one piece of information callers need to know which *graph-level*
         * op set is safe to build: some ops (e.g. `GraphBuilder::gqaMatMul()`) are only
         * implemented for a subset of backends, so a caller building a graph ahead of
         * time (rather than dispatching immediately) needs this to pick a graph shape
         * that the eventual backend can actually run.
         */
        DeviceType deviceType() const;

        std::shared_ptr<Tensor> createTensor(const TensorDescriptor &desc);

        /**
         * @brief Executes a compiled `Graph`, binding tensors to its named inputs/outputs.
         * @return A `Fence` signaled once execution completes.
         */
        std::shared_ptr<Fence> dispatch(
            const Graph &graph,
            const std::unordered_map<std::string, std::shared_ptr<Tensor>> &inputs,
            const std::unordered_map<std::string, std::shared_ptr<Tensor>> &outputs);
    };

} // namespace systems::leal::campello_nn

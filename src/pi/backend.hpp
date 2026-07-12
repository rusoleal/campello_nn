#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <campello_nn/descriptors/tensor_descriptor.hpp>
#include "ir.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief Internal interface every accelerator backend (CPU, MPSGraph, DirectML, ...) implements.
     *
     * `Context`, `GraphBuilder`, `Tensor`, `Graph`, and `Fence` all delegate to a
     * `Backend` instance through opaque `void*` handles — never through public headers.
     */
    class Backend
    {
    public:
        virtual ~Backend() = default;

        virtual void *createTensor(const TensorDescriptor &desc) = 0;
        virtual void destroyTensor(void *native) = 0;
        virtual void writeTensor(void *native, const void *data, size_t size) = 0;
        virtual void readTensor(void *native, void *data, size_t size) = 0;

        // By value, not const&: implementations move `ir` into their compiled
        // graph's stored copy instead of deep-copying it (a real cost for
        // weight-heavy models — every Constant node's raw bytes would otherwise
        // be copied again on top of the copies already made building `ir` in the
        // first place). Callers (GraphBuilder::build()/deserialize()) pass via
        // std::move() for the same reason.
        virtual void *compileGraph(GraphIR ir) = 0;
        virtual void destroyGraph(void *native) = 0;

        // Executes a compiled graph, binding tensor natives by name. Returns a
        // backend-owned fence native.
        virtual void *dispatch(
            void *compiledGraph,
            const std::unordered_map<std::string, void *> &inputs,
            const std::unordered_map<std::string, void *> &outputs) = 0;

        virtual bool waitFence(void *fenceNative, uint64_t timeoutNs) = 0;
        virtual bool isFenceSignaled(void *fenceNative) = 0;
        virtual void destroyFence(void *fenceNative) = 0;
    };

} // namespace systems::leal::campello_nn

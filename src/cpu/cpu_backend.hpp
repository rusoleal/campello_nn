#pragma once

#include "../pi/backend.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief Reference backend: interprets the IR directly on the CPU, Float32 only.
     *
     * No SDK/hardware dependency, so it's the backend exercised by universal tests
     * and the fallback for any `DeviceType` without a dedicated accelerator backend yet.
     */
    class CpuBackend : public Backend
    {
    public:
        void *createTensor(const TensorDescriptor &desc) override;
        void destroyTensor(void *native) override;
        void writeTensor(void *native, const void *data, size_t size) override;
        void readTensor(void *native, void *data, size_t size) override;

        void *compileGraph(const GraphIR &ir) override;
        void destroyGraph(void *native) override;

        void *dispatch(
            void *compiledGraph,
            const std::unordered_map<std::string, void *> &inputs,
            const std::unordered_map<std::string, void *> &outputs) override;

        bool waitFence(void *fenceNative, uint64_t timeoutNs) override;
        bool isFenceSignaled(void *fenceNative) override;
        void destroyFence(void *fenceNative) override;
    };

} // namespace systems::leal::campello_nn

#pragma once

#include "../pi/backend.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief MPSGraph-backed accelerator backend (macOS/iOS).
     *
     * Pure C++ interface — implemented in `mps_backend.mm` since MPSGraph has no
     * Apple-provided C++ binding (unlike Metal/metal-cpp). All Objective-C types
     * are confined to the `Impl` struct defined in the `.mm` file, so this header
     * is safe to include from plain `.cpp` translation units (e.g. `context.cpp`).
     */
    class MpsBackend : public Backend
    {
    public:
        MpsBackend();
        ~MpsBackend() override;

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

    private:
        struct Impl;
        Impl *impl;
    };

} // namespace systems::leal::campello_nn

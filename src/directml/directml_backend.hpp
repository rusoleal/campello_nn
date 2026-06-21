#pragma once

#include "../pi/backend.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief DirectML-backed accelerator backend (Windows).
     *
     * Pure C++ interface — implemented in `directml_backend.cpp`. DirectML's COM
     * API is consumable from plain C++ (`<wrl/client.h>` ComPtr, no language
     * extension needed, unlike MPSGraph), but `<d3d12.h>`/`<DirectML.h>` are kept
     * out of this header (pimpl'd `Impl`) for the same reason the MPSGraph backend
     * keeps Objective-C types out of `mps_backend.hpp` — so `context.cpp` can
     * include this header without pulling in Windows-SDK/DirectML includes.
     */
    class DirectMlBackend : public Backend
    {
    public:
        DirectMlBackend();
        ~DirectMlBackend() override;

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

        // Forward-declared publicly (definition stays in directml_backend.cpp, so
        // no Windows-SDK/DirectML types leak here) because free helper functions
        // in that .cpp's anonymous namespace take `Impl*` — a private nested type
        // would make those parameter types inaccessible outside the class.
        struct Impl;

    private:
        Impl *impl;
    };

} // namespace systems::leal::campello_nn

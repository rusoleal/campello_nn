#pragma once

#include "../pi/backend.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief NNAPI-backed accelerator backend (Android), falling back to XNNPACK.
     *
     * Pure C++ interface — implemented in `android_backend.cpp`. NNAPI's C API
     * (`<android/NeuralNetworks.h>`) is plain C, consumable from ordinary C++ (no
     * language extension needed, same as DirectML's COM API), but kept out of this
     * header via a pimpl'd `Impl` so `context.cpp` doesn't need NDK NNAPI/XNNPACK
     * includes — same reasoning as `mps_backend.hpp`/`directml_backend.hpp`.
     *
     * NNAPI is deprecated as of Android 15 but still present/functional — initially chosen
     * over the newer LiteRT GPU/NPU delegates because NNAPI's model/compilation/execution C
     * API is an IR-walk-and-compile shape matching this repo's other backends, whereas
     * LiteRT delegates attach to a `TfLiteInterpreter` loaded from a TFLite FlatBuffer
     * model (would need a GraphIR→flatbuffer exporter first).
     *
     * **Superseded:** this NNAPI path is being replaced by a self-contained backend built
     * on the sibling `campello_gpu` library's Vulkan support (also "production ready" on
     * Linux, so one backend covers both platforms) instead of depending on NNAPI/LiteRT —
     * see `TODO.md`'s "3c/3d. Linux & Android — `campello_gpu` Vulkan compute backend".
     * This file is kept as history of the path tried first; not yet deleted since
     * `android_backend.cpp` was never written.
     */
    class AndroidBackend : public Backend
    {
    public:
        AndroidBackend();
        ~AndroidBackend() override;

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

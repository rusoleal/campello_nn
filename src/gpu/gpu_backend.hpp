#pragma once

#include "../pi/backend.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief campello_gpu-backed generic accelerator backend (`DeviceType::GpuGeneric`).
     *
     * Pure C++ interface — implemented in `gpu_backend.cpp`. Unlike `mps_backend.hpp`/
     * `directml_backend.hpp` (each compiled on exactly one platform, selected by the
     * top-level `CMakeLists.txt`'s per-platform `.cmake` include), this single source
     * file is compiled on every platform campello_gpu supports — campello_gpu itself
     * picks Metal/Vulkan/DirectX12 internally, so this backend's C++ logic (graph
     * walk, buffer/bind-group management, dispatch) is shared; only the precompiled
     * shader bytes loaded in `gpu_backend.cpp` differ per platform (`#ifdef
     * __APPLE__`/`_WIN32`/else, picking `.metallib`/DirectX-bytecode/`.spv` byte
     * arrays — see `src/gpu/shaders/`).
     *
     * Coverage is documented in `TODO.md` Phase 3c/3d. `DeviceType::GpuGeneric` is an
     * explicitly-selected addition for benchmarking against the platform-native
     * `DeviceType::Gpu` backends (MPSGraph/DirectML), not a replacement for them.
     */
    class GpuBackend : public Backend
    {
    public:
        GpuBackend();
        ~GpuBackend() override;

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

        // Forward-declared publicly, same reasoning as DirectMlBackend::Impl —
        // free helper functions in gpu_backend.cpp's anonymous namespace take
        // `Impl*`, so it can't be a private nested type.
        struct Impl;

    private:
        Impl *impl;
    };

} // namespace systems::leal::campello_nn

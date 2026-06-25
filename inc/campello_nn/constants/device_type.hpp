#pragma once

namespace systems::leal::campello_nn
{

    /**
     * @brief Selects which kind of compute device a `Context` should target.
     */
    enum class DeviceType
    {
        Cpu,
        Gpu,
        Npu,
        Default,

        /**
         * @brief The campello_gpu-based generic GPU backend.
         *
         * Unlike `Gpu` (which routes to the platform-native backend — MPSGraph on
         * Apple, DirectML on Windows — and is unimplemented elsewhere), `GpuGeneric`
         * always routes to the same campello_gpu-based backend on every platform
         * (Metal/Vulkan/DirectX12 under the hood, selected by campello_gpu itself).
         * Exists for explicit benchmarking against the native backends, not as a
         * replacement — `Gpu` keeps its existing platform-native behavior.
         */
        GpuGeneric
    };

}

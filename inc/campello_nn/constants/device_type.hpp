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
        Default
    };

}

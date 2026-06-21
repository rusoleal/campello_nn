#pragma once

#include <campello_nn/constants/device_type.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Configuration used to select and create a `Context`.
     */
    struct ContextDescriptor
    {
        DeviceType deviceType = DeviceType::Default;
    };

}

#pragma once

#include <memory>
#include <campello_nn/constants/device_type.hpp>
#include "backend.hpp"

namespace systems::leal::campello_nn
{

    struct ContextData
    {
        std::unique_ptr<Backend> backend;
        DeviceType deviceType;
    };

} // namespace systems::leal::campello_nn

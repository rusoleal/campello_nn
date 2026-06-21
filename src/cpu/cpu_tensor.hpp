#pragma once

#include <cstdint>
#include <vector>
#include <campello_nn/descriptors/tensor_descriptor.hpp>

namespace systems::leal::campello_nn
{

    struct CpuTensor
    {
        std::vector<uint8_t> bytes;
        TensorDescriptor desc;
    };

} // namespace systems::leal::campello_nn

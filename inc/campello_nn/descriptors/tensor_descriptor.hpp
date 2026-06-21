#pragma once

#include <cstdint>
#include <vector>
#include <campello_nn/constants/data_type.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Describes the shape, element type, and CPU-access mode of a `Tensor`.
     */
    struct TensorDescriptor
    {
        DataType dataType;
        std::vector<int64_t> shape;
        bool readable = false; // CPU-mappable for read()
        bool writable = false; // CPU-mappable for write()
    };

}

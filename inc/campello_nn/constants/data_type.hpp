#pragma once

namespace systems::leal::campello_nn
{

    /**
     * @brief Element type stored in a `Tensor` or produced by a graph `Operand`.
     */
    enum class DataType
    {
        Float32,
        Float16,
        Int32,
        Int8,
        Uint32
    };

}

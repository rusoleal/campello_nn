#pragma once

#include <cstdint>
#include <vector>
#include <campello_nn/constants/data_type.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Materialized value of one IR node during CPU graph execution.
     *
     * Stored as raw bytes (not `vector<float>`) so `Gather`'s `indices` operand
     * can carry Int32 data alongside the Float32 values every other op uses.
     */
    struct CpuValue
    {
        std::vector<uint8_t> bytes;
        std::vector<int64_t> shape;
        DataType dataType = DataType::Float32;

        float *f() { return (float *)bytes.data(); }
        const float *f() const { return (const float *)bytes.data(); }
        int32_t *i32() { return (int32_t *)bytes.data(); }
        const int32_t *i32() const { return (const int32_t *)bytes.data(); }
        uint32_t *u32() { return (uint32_t *)bytes.data(); }
        const uint32_t *u32() const { return (const uint32_t *)bytes.data(); }

        size_t floatCount() const { return bytes.size() / sizeof(float); }
        size_t int32Count() const { return bytes.size() / sizeof(int32_t); }
    };

} // namespace systems::leal::campello_nn

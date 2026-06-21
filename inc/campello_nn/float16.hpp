#pragma once

#include <cstdint>
#include <cstring>

namespace systems::leal::campello_nn
{

    /**
     * @brief Encodes a 32-bit float into IEEE-754 binary16 (half-precision) bits.
     *
     * C++20 has no built-in half-precision type, so `Tensor`s declared with
     * `DataType::Float16` store this 2-byte encoding directly. Use this to produce
     * the bytes passed to `Tensor::write()`.
     */
    inline uint16_t encodeFloat16(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));

        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFFu;

        if (((bits >> 23) & 0xFFu) == 0xFFu)
        {
            // Inf / NaN
            return (uint16_t)(sign | 0x7C00u | (mantissa != 0 ? 0x0200u : 0u));
        }
        if (exponent >= 31)
        {
            // Overflow -> Inf
            return (uint16_t)(sign | 0x7C00u);
        }
        if (exponent <= 0)
        {
            if (exponent < -10)
                return (uint16_t)sign; // Underflow -> zero
            // Subnormal half
            mantissa |= 0x800000u;
            uint32_t shift = (uint32_t)(14 - exponent);
            uint32_t half = mantissa >> shift;
            if ((mantissa >> (shift - 1)) & 1u)
                half++;
            return (uint16_t)(sign | half);
        }
        uint32_t half = mantissa >> 13;
        if (mantissa & 0x1000u)
        {
            half++;
            if (half == 0x400u)
            {
                half = 0;
                exponent++;
                if (exponent >= 31)
                    return (uint16_t)(sign | 0x7C00u);
            }
        }
        return (uint16_t)(sign | ((uint32_t)exponent << 10) | half);
    }

    /**
     * @brief Decodes IEEE-754 binary16 (half-precision) bits into a 32-bit float.
     */
    inline float decodeFloat16(uint16_t h)
    {
        uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        uint32_t exponent = (h >> 10) & 0x1Fu;
        uint32_t mantissa = h & 0x3FFu;
        uint32_t bits;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                // Subnormal half -> normalize into a normal float.
                int32_t e = -1;
                do
                {
                    e++;
                    mantissa <<= 1;
                } while ((mantissa & 0x400u) == 0);
                mantissa &= 0x3FFu;
                uint32_t outExponent = (uint32_t)(127 - 15 - e);
                bits = sign | (outExponent << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 0x1Fu)
        {
            // Inf / NaN
            bits = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            uint32_t outExponent = exponent - 15 + 127;
            bits = sign | (outExponent << 23) | (mantissa << 13);
        }

        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

} // namespace systems::leal::campello_nn

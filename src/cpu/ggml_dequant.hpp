#pragma once

#include <cstdint>
#include <vector>

namespace systems::leal::campello_nn
{

    /**
     * @brief Dequantizes a raw GGML block-quantized tensor into a Float32 buffer.
     *
     * The quantized tensor's original (GGUF) shape is [outFeatures, inFeatures].
     * The returned buffer has shape [inFeatures, outFeatures] in row-major order,
     * i.e. the transpose needed for a standard `x @ W` matmul is applied during
     * dequantization.
     *
     * Supported types mirror campello_llm's GGUF loader: Q4_0, Q4_1, Q5_0, Q5_1,
     * Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K.
     *
     * @param raw          Pointer to the raw quantized bytes.
     * @param rawByteSize  Number of bytes in `raw`.
     * @param ggmlType     GGML quantization type enum value.
     * @param outFeatures  First dimension of the GGUF tensor shape.
     * @param inFeatures   Second dimension of the GGUF tensor shape.
     * @throws std::runtime_error for unsupported types or size mismatches.
     */
    std::vector<float> dequantizeGgmlWeightTransposed(const uint8_t *raw, size_t rawByteSize, int32_t ggmlType,
                                                      int64_t outFeatures, int64_t inFeatures);

} // namespace systems::leal::campello_nn

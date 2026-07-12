#include "ggml_dequant.hpp"

#include <campello_nn/float16.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace systems::leal::campello_nn
{

    namespace
    {
        float decodeFloat16Scale(std::uint16_t bits)
        {
            return decodeFloat16(bits);
        }

        float decodeFloat32Scale(const std::uint8_t *ptr)
        {
            float d;
            std::memcpy(&d, ptr, sizeof(d));
            return d;
        }

        // Helpers to write a dequantized value into the transposed output buffer.
        // The GGUF tensor is row-major [outFeatures, inFeatures]; the output is
        // row-major [inFeatures, outFeatures].
        inline size_t transposedIndex(int64_t outFeature, int64_t inFeature, int64_t outFeatures)
        {
            return static_cast<size_t>(inFeature * outFeatures + outFeature);
        }

        void checkTotalElements(int64_t outFeatures, int64_t inFeatures, size_t expectedBytes, size_t actualBytes)
        {
            if (actualBytes != expectedBytes)
                throw std::runtime_error("campello_nn: GGML quantized weight byte size does not match shape");
            (void)outFeatures;
            (void)inFeatures;
        }

        // Q4_0: 32 elements per block. Bytes: d(2), qs[16].
        std::vector<float> dequantizeQ4_0(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 32;
            constexpr size_t kBlockBytes = 2 + 16;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q4_0 tensor element count is not a multiple of 32");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, raw, sizeof(scaleBits));
                raw += sizeof(scaleBits);
                float d = decodeFloat16Scale(scaleBits);
                for (size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t q = raw[j];
                    int64_t o = static_cast<int64_t>((outIdx + j) / inFeatures);
                    int64_t i = static_cast<int64_t>((outIdx + j) % inFeatures);
                    out[transposedIndex(o, i, outFeatures)] = (static_cast<float>(static_cast<std::int8_t>(q & 0x0F) - 8)) * d;
                    o = static_cast<int64_t>((outIdx + j + 16) / inFeatures);
                    i = static_cast<int64_t>((outIdx + j + 16) % inFeatures);
                    out[transposedIndex(o, i, outFeatures)] = (static_cast<float>(static_cast<std::int8_t>(q >> 4) - 8)) * d;
                }
                raw += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q4_1: 32 elements per block. Bytes: d(2), m(2), qs[16].
        std::vector<float> dequantizeQ4_1(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 32;
            constexpr size_t kBlockBytes = 2 + 2 + 16;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q4_1 tensor element count is not a multiple of 32");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits, minBits;
                std::memcpy(&scaleBits, raw, sizeof(scaleBits));
                std::memcpy(&minBits, raw + 2, sizeof(minBits));
                raw += 4;
                float d = decodeFloat16Scale(scaleBits);
                float m = decodeFloat16Scale(minBits);
                for (size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t q = raw[j];
                    int64_t o = static_cast<int64_t>((outIdx + j) / inFeatures);
                    int64_t i = static_cast<int64_t>((outIdx + j) % inFeatures);
                    out[transposedIndex(o, i, outFeatures)] = static_cast<float>(q & 0x0F) * d + m;
                    o = static_cast<int64_t>((outIdx + j + 16) / inFeatures);
                    i = static_cast<int64_t>((outIdx + j + 16) % inFeatures);
                    out[transposedIndex(o, i, outFeatures)] = static_cast<float>(q >> 4) * d + m;
                }
                raw += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q5_0: 32 elements per block. Bytes: d(2), qh(4), qs(16).
        std::vector<float> dequantizeQ5_0(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 32;
            constexpr size_t kBlockBytes = 2 + 4 + 16;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q5_0 tensor element count is not a multiple of 32");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, raw, sizeof(scaleBits));
                raw += 2;
                float d = decodeFloat16Scale(scaleBits);
                std::uint32_t qh;
                std::memcpy(&qh, raw, sizeof(qh));
                raw += 4;
                for (size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
                    std::uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
                    std::int32_t x0 = static_cast<std::int32_t>((raw[j] & 0x0F) | xh0) - 16;
                    std::int32_t x1 = static_cast<std::int32_t>((raw[j] >> 4) | xh1) - 16;
                    int64_t o0 = static_cast<int64_t>((outIdx + j) / inFeatures);
                    int64_t i0 = static_cast<int64_t>((outIdx + j) % inFeatures);
                    int64_t o1 = static_cast<int64_t>((outIdx + j + 16) / inFeatures);
                    int64_t i1 = static_cast<int64_t>((outIdx + j + 16) % inFeatures);
                    out[transposedIndex(o0, i0, outFeatures)] = static_cast<float>(x0) * d;
                    out[transposedIndex(o1, i1, outFeatures)] = static_cast<float>(x1) * d;
                }
                raw += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q5_1: 32 elements per block. Bytes: d(2), m(2), qh(4), qs(16).
        std::vector<float> dequantizeQ5_1(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 32;
            constexpr size_t kBlockBytes = 2 + 2 + 4 + 16;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q5_1 tensor element count is not a multiple of 32");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits, minBits;
                std::memcpy(&scaleBits, raw, sizeof(scaleBits));
                std::memcpy(&minBits, raw + 2, sizeof(minBits));
                raw += 4;
                float d = decodeFloat16Scale(scaleBits);
                float m = decodeFloat16Scale(minBits);
                std::uint32_t qh;
                std::memcpy(&qh, raw, sizeof(qh));
                raw += 4;
                for (size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
                    std::uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
                    std::int32_t x0 = static_cast<std::int32_t>((raw[j] & 0x0F) | xh0);
                    std::int32_t x1 = static_cast<std::int32_t>((raw[j] >> 4) | xh1);
                    int64_t o0 = static_cast<int64_t>((outIdx + j) / inFeatures);
                    int64_t i0 = static_cast<int64_t>((outIdx + j) % inFeatures);
                    int64_t o1 = static_cast<int64_t>((outIdx + j + 16) / inFeatures);
                    int64_t i1 = static_cast<int64_t>((outIdx + j + 16) % inFeatures);
                    out[transposedIndex(o0, i0, outFeatures)] = static_cast<float>(x0) * d + m;
                    out[transposedIndex(o1, i1, outFeatures)] = static_cast<float>(x1) * d + m;
                }
                raw += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q8_0: 32 elements per block. Bytes: d(2), qs[32].
        std::vector<float> dequantizeQ8_0(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 32;
            constexpr size_t kBlockBytes = 2 + 32;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q8_0 tensor element count is not a multiple of 32");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, raw, sizeof(scaleBits));
                raw += sizeof(scaleBits);
                float d = decodeFloat16Scale(scaleBits);
                for (size_t j = 0; j < kBlockElements; ++j)
                {
                    int64_t o = static_cast<int64_t>((outIdx + j) / inFeatures);
                    int64_t i = static_cast<int64_t>((outIdx + j) % inFeatures);
                    out[transposedIndex(o, i, outFeatures)] = static_cast<float>(static_cast<std::int8_t>(raw[j])) * d;
                }
                raw += kBlockElements;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q8_1: 32 elements per block. Bytes: d(2), s(2), qs[32].
        std::vector<float> dequantizeQ8_1(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 32;
            constexpr size_t kBlockBytes = 2 + 2 + 32;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q8_1 tensor element count is not a multiple of 32");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, raw, sizeof(scaleBits));
                raw += 4; // skip d and s
                float d = decodeFloat16Scale(scaleBits);
                for (size_t j = 0; j < kBlockElements; ++j)
                {
                    int64_t o = static_cast<int64_t>((outIdx + j) / inFeatures);
                    int64_t i = static_cast<int64_t>((outIdx + j) % inFeatures);
                    out[transposedIndex(o, i, outFeatures)] = static_cast<float>(static_cast<std::int8_t>(raw[j])) * d;
                }
                raw += kBlockElements;
                outIdx += kBlockElements;
            }
            return out;
        }

        inline void getScaleMinK4(int j, const std::uint8_t *q, std::uint8_t *d, std::uint8_t *m)
        {
            if (j < 4)
            {
                *d = q[j] & 63;
                *m = q[j + 4] & 63;
            }
            else
            {
                *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
                *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
            }
        }

        // Q2_K: 256 elements per super-block. Bytes: scales[16], qs[64], d(2), dmin(2).
        std::vector<float> dequantizeQ2_K(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 256;
            constexpr size_t kBlockBytes = 16 + 64 + 2 + 2;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q2_K tensor element count is not a multiple of 256");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                const std::uint8_t *scales = raw;
                raw += 16;
                const std::uint8_t *qs = raw;
                raw += 64;
                std::uint16_t dBits, minBits;
                std::memcpy(&dBits, raw, sizeof(dBits));
                std::memcpy(&minBits, raw + 2, sizeof(minBits));
                raw += 4;
                float d = decodeFloat16Scale(dBits);
                float min = decodeFloat16Scale(minBits);

                int is = 0;
                for (int n = 0; n < static_cast<int>(kBlockElements); n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        std::uint8_t sc = scales[is++];
                        float dl = d * static_cast<float>(sc & 0x0F);
                        float ml = min * static_cast<float>(sc >> 4);
                        for (int l = 0; l < 16; ++l)
                        {
                            int64_t o = static_cast<int64_t>((outIdx) / inFeatures);
                            int64_t i = static_cast<int64_t>((outIdx) % inFeatures);
                            out[transposedIndex(o, i, outFeatures)] = dl * static_cast<float>((qs[l] >> shift) & 3) - ml;
                            ++outIdx;
                        }
                        sc = scales[is++];
                        dl = d * static_cast<float>(sc & 0x0F);
                        ml = min * static_cast<float>(sc >> 4);
                        for (int l = 0; l < 16; ++l)
                        {
                            int64_t o = static_cast<int64_t>((outIdx) / inFeatures);
                            int64_t i = static_cast<int64_t>((outIdx) % inFeatures);
                            out[transposedIndex(o, i, outFeatures)] = dl * static_cast<float>((qs[l + 16] >> shift) & 3) - ml;
                            ++outIdx;
                        }
                        shift += 2;
                    }
                    qs += 32;
                }
            }
            return out;
        }

        // Q3_K: 256 elements per super-block. Bytes: hmask[32], qs[64], scales[12], d(2).
        std::vector<float> dequantizeQ3_K(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 256;
            constexpr size_t kBlockBytes = 32 + 64 + 12 + 2;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q3_K tensor element count is not a multiple of 256");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                const std::uint8_t *hmask = raw;
                raw += 32;
                const std::uint8_t *qs = raw;
                raw += 64;
                const std::uint8_t *scalesIn = raw;
                raw += 12;
                std::uint16_t dBits;
                std::memcpy(&dBits, raw, sizeof(dBits));
                raw += 2;
                float dAll = decodeFloat16Scale(dBits);

                std::uint32_t aux[4];
                std::memcpy(aux, scalesIn, 12);
                std::uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
                aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
                aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
                const auto *scales = reinterpret_cast<const std::int8_t *>(aux);

                std::uint8_t m = 1;
                int is = 0;
                for (int n = 0; n < static_cast<int>(kBlockElements); n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        float dl = dAll * static_cast<float>(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l)
                        {
                            int q = static_cast<int>((qs[l] >> shift) & 3);
                            q -= (hmask[l] & m) ? 0 : 4;
                            int64_t o = static_cast<int64_t>(outIdx / inFeatures);
                            int64_t i = static_cast<int64_t>(outIdx % inFeatures);
                            out[transposedIndex(o, i, outFeatures)] = dl * static_cast<float>(q);
                            ++outIdx;
                        }
                        dl = dAll * static_cast<float>(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l)
                        {
                            int q = static_cast<int>((qs[l + 16] >> shift) & 3);
                            q -= (hmask[l + 16] & m) ? 0 : 4;
                            int64_t o = static_cast<int64_t>(outIdx / inFeatures);
                            int64_t i = static_cast<int64_t>(outIdx % inFeatures);
                            out[transposedIndex(o, i, outFeatures)] = dl * static_cast<float>(q);
                            ++outIdx;
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    qs += 32;
                }
            }
            return out;
        }

        // Q4_K: 256 elements per super-block. Bytes: d(2), dmin(2), scales[12], qs[128].
        std::vector<float> dequantizeQ4_K(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 256;
            constexpr size_t kBlockBytes = 2 + 2 + 12 + 128;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q4_K tensor element count is not a multiple of 256");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t dBits, minBits;
                std::memcpy(&dBits, raw, sizeof(dBits));
                std::memcpy(&minBits, raw + 2, sizeof(minBits));
                raw += 4;
                float d = decodeFloat16Scale(dBits);
                float min = decodeFloat16Scale(minBits);
                const std::uint8_t *scales = raw;
                raw += 12;
                const std::uint8_t *q = raw;
                raw += 128;

                int is = 0;
                std::uint8_t sc, m;
                for (size_t j = 0; j < kBlockElements; j += 64)
                {
                    getScaleMinK4(is + 0, scales, &sc, &m);
                    float d1 = d * static_cast<float>(sc);
                    float m1 = min * static_cast<float>(m);
                    getScaleMinK4(is + 1, scales, &sc, &m);
                    float d2 = d * static_cast<float>(sc);
                    float m2 = min * static_cast<float>(m);
                    for (int l = 0; l < 32; ++l)
                    {
                        int64_t o = static_cast<int64_t>(outIdx / inFeatures);
                        int64_t i = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o, i, outFeatures)] = d1 * static_cast<float>(q[l] & 0x0F) - m1;
                        ++outIdx;
                    }
                    for (int l = 0; l < 32; ++l)
                    {
                        int64_t o = static_cast<int64_t>(outIdx / inFeatures);
                        int64_t i = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o, i, outFeatures)] = d2 * static_cast<float>(q[l] >> 4) - m2;
                        ++outIdx;
                    }
                    q += 32;
                    is += 2;
                }
            }
            return out;
        }

        // Q5_K: 256 elements per super-block. Bytes: d(2), dmin(2), scales[12], qh[32], qs[128].
        std::vector<float> dequantizeQ5_K(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 256;
            constexpr size_t kBlockBytes = 2 + 2 + 12 + 32 + 128;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q5_K tensor element count is not a multiple of 256");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t dBits, minBits;
                std::memcpy(&dBits, raw, sizeof(dBits));
                std::memcpy(&minBits, raw + 2, sizeof(minBits));
                raw += 4;
                float d = decodeFloat16Scale(dBits);
                float min = decodeFloat16Scale(minBits);
                const std::uint8_t *scales = raw;
                raw += 12;
                const std::uint8_t *qh = raw;
                raw += 32;
                const std::uint8_t *ql = raw;
                raw += 128;

                int is = 0;
                std::uint8_t sc, m;
                std::uint8_t u1 = 1, u2 = 2;
                for (size_t j = 0; j < kBlockElements; j += 64)
                {
                    getScaleMinK4(is + 0, scales, &sc, &m);
                    float d1 = d * static_cast<float>(sc);
                    float m1 = min * static_cast<float>(m);
                    getScaleMinK4(is + 1, scales, &sc, &m);
                    float d2 = d * static_cast<float>(sc);
                    float m2 = min * static_cast<float>(m);
                    for (int l = 0; l < 32; ++l)
                    {
                        std::uint8_t v = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                        int64_t o = static_cast<int64_t>(outIdx / inFeatures);
                        int64_t i = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o, i, outFeatures)] = d1 * static_cast<float>(v) - m1;
                        ++outIdx;
                    }
                    for (int l = 0; l < 32; ++l)
                    {
                        std::uint8_t v = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                        int64_t o = static_cast<int64_t>(outIdx / inFeatures);
                        int64_t i = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o, i, outFeatures)] = d2 * static_cast<float>(v) - m2;
                        ++outIdx;
                    }
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            return out;
        }

        // Q6_K: 256 elements per super-block. Bytes: ql[128], qh[64], scales[16], d(2).
        std::vector<float> dequantizeQ6_K(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 256;
            constexpr size_t kBlockBytes = 128 + 64 + 16 + 2;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q6_K tensor element count is not a multiple of 256");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                const std::uint8_t *ql = raw;
                raw += 128;
                const std::uint8_t *qh = raw;
                raw += 64;
                const auto *scales = reinterpret_cast<const std::int8_t *>(raw);
                raw += 16;
                std::uint16_t dBits;
                std::memcpy(&dBits, raw, sizeof(dBits));
                raw += 2;
                float d = decodeFloat16Scale(dBits);

                for (size_t n = 0; n < kBlockElements; n += 128)
                {
                    for (int l = 0; l < 32; ++l)
                    {
                        int is = l / 16;
                        std::int8_t q1 = static_cast<std::int8_t>((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        std::int8_t q2 = static_cast<std::int8_t>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        std::int8_t q3 = static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        std::int8_t q4 = static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        int64_t o1 = static_cast<int64_t>(outIdx / inFeatures);
                        int64_t i1 = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o1, i1, outFeatures)] = d * static_cast<float>(scales[is + 0]) * static_cast<float>(q1);
                        ++outIdx;
                        o1 = static_cast<int64_t>(outIdx / inFeatures);
                        i1 = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o1, i1, outFeatures)] = d * static_cast<float>(scales[is + 2]) * static_cast<float>(q2);
                        ++outIdx;
                        o1 = static_cast<int64_t>(outIdx / inFeatures);
                        i1 = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o1, i1, outFeatures)] = d * static_cast<float>(scales[is + 4]) * static_cast<float>(q3);
                        ++outIdx;
                        o1 = static_cast<int64_t>(outIdx / inFeatures);
                        i1 = static_cast<int64_t>(outIdx % inFeatures);
                        out[transposedIndex(o1, i1, outFeatures)] = d * static_cast<float>(scales[is + 6]) * static_cast<float>(q4);
                        ++outIdx;
                    }
                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
            return out;
        }

        // Q8_K: 256 elements per super-block. Bytes: d(4), qs[256], bsums[16]*2.
        std::vector<float> dequantizeQ8_K(const uint8_t *raw, size_t rawByteSize, int64_t outFeatures, int64_t inFeatures)
        {
            constexpr size_t kBlockElements = 256;
            constexpr size_t kBlockBytes = 4 + 256 + 32;
            int64_t totalElements = outFeatures * inFeatures;
            if (totalElements % static_cast<int64_t>(kBlockElements) != 0)
                throw std::runtime_error("campello_nn: Q8_K tensor element count is not a multiple of 256");
            size_t nBlocks = static_cast<size_t>(totalElements / static_cast<int64_t>(kBlockElements));
            checkTotalElements(outFeatures, inFeatures, nBlocks * kBlockBytes, rawByteSize);

            std::vector<float> out(static_cast<size_t>(totalElements));
            size_t outIdx = 0;
            for (size_t b = 0; b < nBlocks; ++b)
            {
                float d = decodeFloat32Scale(raw);
                raw += 4;
                const auto *qs = reinterpret_cast<const std::int8_t *>(raw);
                raw += 256;
                raw += 32; // skip bsums
                for (size_t j = 0; j < kBlockElements; ++j)
                {
                    int64_t o = static_cast<int64_t>((outIdx + j) / inFeatures);
                    int64_t i = static_cast<int64_t>((outIdx + j) % inFeatures);
                    out[transposedIndex(o, i, outFeatures)] = d * static_cast<float>(qs[j]);
                }
                outIdx += kBlockElements;
            }
            return out;
        }
    } // namespace

    std::vector<float> dequantizeGgmlWeightTransposed(const uint8_t *raw, size_t rawByteSize, int32_t ggmlType,
                                                      int64_t outFeatures, int64_t inFeatures)
    {
        if (outFeatures <= 0 || inFeatures <= 0)
            throw std::runtime_error("campello_nn: GGML quantized weight shape must be positive");
        if (raw == nullptr)
            throw std::runtime_error("campello_nn: GGML quantized weight data is null");

        switch (ggmlType)
        {
        case 2:
            return dequantizeQ4_0(raw, rawByteSize, outFeatures, inFeatures);
        case 3:
            return dequantizeQ4_1(raw, rawByteSize, outFeatures, inFeatures);
        case 6:
            return dequantizeQ5_0(raw, rawByteSize, outFeatures, inFeatures);
        case 7:
            return dequantizeQ5_1(raw, rawByteSize, outFeatures, inFeatures);
        case 8:
            return dequantizeQ8_0(raw, rawByteSize, outFeatures, inFeatures);
        case 9:
            return dequantizeQ8_1(raw, rawByteSize, outFeatures, inFeatures);
        case 10:
            return dequantizeQ2_K(raw, rawByteSize, outFeatures, inFeatures);
        case 11:
            return dequantizeQ3_K(raw, rawByteSize, outFeatures, inFeatures);
        case 12:
            return dequantizeQ4_K(raw, rawByteSize, outFeatures, inFeatures);
        case 13:
            return dequantizeQ5_K(raw, rawByteSize, outFeatures, inFeatures);
        case 14:
            return dequantizeQ6_K(raw, rawByteSize, outFeatures, inFeatures);
        case 15:
            return dequantizeQ8_K(raw, rawByteSize, outFeatures, inFeatures);
        default:
            throw std::runtime_error("campello_nn: unsupported GGML quantization type " + std::to_string(ggmlType));
        }
    }

} // namespace systems::leal::campello_nn

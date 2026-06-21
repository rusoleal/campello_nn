#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

namespace systems::leal::campello_nn
{

    inline int64_t numElements(const std::vector<int64_t> &shape)
    {
        return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
    }

    inline std::vector<int64_t> rowMajorStrides(const std::vector<int64_t> &shape)
    {
        std::vector<int64_t> strides(shape.size());
        int64_t acc = 1;
        for (int64_t i = (int64_t)shape.size() - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    inline std::vector<int64_t> unravelIndex(int64_t flat, const std::vector<int64_t> &shape)
    {
        std::vector<int64_t> idx(shape.size());
        for (int64_t d = (int64_t)shape.size() - 1; d >= 0; d--)
        {
            idx[d] = flat % shape[d];
            flat /= shape[d];
        }
        return idx;
    }

    inline int64_t ravelIndex(const std::vector<int64_t> &idx, const std::vector<int64_t> &strides)
    {
        int64_t flat = 0;
        for (size_t d = 0; d < idx.size(); d++)
            flat += idx[d] * strides[d];
        return flat;
    }

} // namespace systems::leal::campello_nn

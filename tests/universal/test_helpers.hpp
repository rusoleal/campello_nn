#pragma once

#include <vector>
#include <campello_nn/context.hpp>
#include <campello_nn/graph_builder.hpp>
#include <campello_nn/float16.hpp>

namespace cnn = systems::leal::campello_nn;

inline std::shared_ptr<cnn::Context> makeCpuContext()
{
    return cnn::Context::create({cnn::DeviceType::Cpu});
}

inline std::shared_ptr<cnn::Context> makeGpuContext()
{
    return cnn::Context::create({cnn::DeviceType::Gpu});
}

inline std::vector<uint16_t> toHalf(const std::vector<float> &v)
{
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); i++)
        h[i] = cnn::encodeFloat16(v[i]);
    return h;
}

inline std::vector<float> fromHalf(const std::vector<uint16_t> &h)
{
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); i++)
        v[i] = cnn::decodeFloat16(h[i]);
    return v;
}

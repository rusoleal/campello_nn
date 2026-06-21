#include <cstring>
#include <stdexcept>
#include <campello_nn/float16.hpp>
#include "cpu_backend.hpp"
#include "cpu_tensor.hpp"
#include "cpu_graph.hpp"
#include "cpu_fence.hpp"
#include "cpu_value.hpp"
#include "ops.hpp"
#include "strides.hpp"

using namespace systems::leal::campello_nn;

namespace
{
    size_t elementByteSize(DataType dt)
    {
        switch (dt)
        {
        case DataType::Float32:
        case DataType::Int32:
        case DataType::Uint32:
            return 4;
        case DataType::Float16:
            return 2;
        case DataType::Int8:
            return 1;
        }
        throw std::runtime_error("campello_nn: unknown DataType");
    }

    // The CPU backend computes every op in Float32 internally (the reference/
    // correctness path, not a performance path) — Float16 tensors are decoded to
    // Float32 right when an Input/Constant node is populated, and re-encoded only
    // when writing the final result into a Float16-declared output Tensor. Every
    // kernel in ops.cpp is therefore unaware Float16 exists at all.
    void decodeIntoValue(const std::vector<uint8_t> &srcBytes, DataType declaredType, CpuValue &dst)
    {
        if (declaredType == DataType::Float16)
        {
            size_t count = srcBytes.size() / sizeof(uint16_t);
            dst.bytes.resize(count * sizeof(float));
            const uint16_t *h = (const uint16_t *)srcBytes.data();
            float *f = (float *)dst.bytes.data();
            for (size_t k = 0; k < count; k++)
                f[k] = decodeFloat16(h[k]);
            dst.dataType = DataType::Float32;
        }
        else
        {
            dst.bytes = srcBytes;
            dst.dataType = declaredType;
        }
    }

    void encodeFromValue(const CpuValue &src, DataType declaredType, std::vector<uint8_t> &dstBytes)
    {
        if (declaredType == DataType::Float16)
        {
            size_t count = src.bytes.size() / sizeof(float);
            dstBytes.resize(count * sizeof(uint16_t));
            const float *f = (const float *)src.bytes.data();
            uint16_t *h = (uint16_t *)dstBytes.data();
            for (size_t k = 0; k < count; k++)
                h[k] = encodeFloat16(f[k]);
        }
        else
        {
            dstBytes = src.bytes;
        }
    }
}

void *CpuBackend::createTensor(const TensorDescriptor &desc)
{
    auto t = new CpuTensor();
    t->desc = desc;
    t->bytes.resize(elementByteSize(desc.dataType) * numElements(desc.shape));
    return t;
}

void CpuBackend::destroyTensor(void *native)
{
    delete (CpuTensor *)native;
}

void CpuBackend::writeTensor(void *native, const void *data, size_t size)
{
    auto t = (CpuTensor *)native;
    if (size > t->bytes.size())
        throw std::runtime_error("campello_nn: write() exceeds tensor capacity");
    std::memcpy(t->bytes.data(), data, size);
}

void CpuBackend::readTensor(void *native, void *data, size_t size)
{
    auto t = (CpuTensor *)native;
    if (size > t->bytes.size())
        throw std::runtime_error("campello_nn: read() exceeds tensor capacity");
    std::memcpy(data, t->bytes.data(), size);
}

void *CpuBackend::compileGraph(const GraphIR &ir)
{
    auto g = new CpuGraph();
    g->ir = ir;
    return g;
}

void CpuBackend::destroyGraph(void *native)
{
    delete (CpuGraph *)native;
}

void *CpuBackend::dispatch(
    void *compiledGraph,
    const std::unordered_map<std::string, void *> &inputs,
    const std::unordered_map<std::string, void *> &outputs)
{
    auto g = (CpuGraph *)compiledGraph;
    size_t n = g->ir.nodes.size();
    std::vector<CpuValue> values(n);

    for (size_t i = 0; i < n; i++)
    {
        const Node &node = g->ir.nodes[i];
        if (node.kind == OpKind::Input)
        {
            auto it = inputs.find(node.name);
            if (it == inputs.end())
                throw std::runtime_error("campello_nn: dispatch() missing input tensor '" + node.name + "'");
            auto t = (CpuTensor *)it->second;
            decodeIntoValue(t->bytes, node.dataType, values[i]);
            values[i].shape = node.shape;
        }
        else if (node.kind == OpKind::Constant)
        {
            decodeIntoValue(node.constantBytes, node.dataType, values[i]);
            values[i].shape = node.shape;
        }
        else
        {
            evalNode(node, i, values);
        }
    }

    for (auto &[name, nodeIdx] : g->ir.outputs)
    {
        auto it = outputs.find(name);
        if (it == outputs.end())
            continue;
        auto t = (CpuTensor *)it->second;
        encodeFromValue(values[nodeIdx], g->ir.nodes[nodeIdx].dataType, t->bytes);
    }

    return new CpuFence{true};
}

bool CpuBackend::waitFence(void *fenceNative, uint64_t)
{
    return ((CpuFence *)fenceNative)->signaled;
}

bool CpuBackend::isFenceSignaled(void *fenceNative)
{
    return ((CpuFence *)fenceNative)->signaled;
}

void CpuBackend::destroyFence(void *fenceNative)
{
    delete (CpuFence *)fenceNative;
}

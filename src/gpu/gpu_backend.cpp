#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <campello_gpu/device.hpp>

#include "gpu_backend.hpp"
#include "../pi/ir.hpp"

#if defined(__APPLE__)
#include "shaders/relu_metallib.hpp"
#include "shaders/add_metallib.hpp"
#include "shaders/matmul_metallib.hpp"
#elif !defined(_WIN32)
#include "shaders/relu_spv.hpp"
#include "shaders/add_spv.hpp"
#include "shaders/matmul_spv.hpp"
#endif
// _WIN32 (DirectX12): no precompiled bytecode shipped yet — src/gpu/shaders/
// *.hlsl are written against real D3D12/HLSL semantics but unverified (no
// Windows toolchain available where they were written). GpuBackend throws a
// clear message on that platform instead of silently failing — see
// loadShaderModule() below.

using namespace systems::leal::campello_nn;
namespace cgpu = systems::leal::campello_gpu;

namespace
{
    // campello_gpu's BufferUsage/ShaderStage are plain `enum class` types with no
    // operator| overload anywhere in the library (confirmed by searching its
    // headers and sources directly — its own internal code combines flags via
    // `(int)usage & (int)BufferUsage::X` bit tests, so callers are expected to
    // do the same). This is a real gap in the dependency, not an oversight here.
    cgpu::BufferUsage combineUsage(cgpu::BufferUsage a, cgpu::BufferUsage b)
    {
        return static_cast<cgpu::BufferUsage>(static_cast<int>(a) | static_cast<int>(b));
    }

    // No mapRead/mapWrite here — nothing in this backend ever calls a mapping
    // API directly, and campello_gpu's Buffer::download() now (as of the
    // local 0.13.3 fix, see TODO.md) correctly encodes a `synchronizeResource:`
    // blit before reading MTLResourceStorageModeManaged buffers (the default
    // storage mode without those flags), so there's no longer a reason to
    // force MTLResourceStorageModeShared here — Managed mode is the more
    // efficient default on discrete-GPU Metal hardware, and Vulkan's
    // Device::createBuffer() doesn't distinguish the two anyway.
    cgpu::BufferUsage tensorBufferUsage()
    {
        return combineUsage(combineUsage(cgpu::BufferUsage::storage, cgpu::BufferUsage::copySrc),
                             cgpu::BufferUsage::copyDst);
    }

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
        throw std::runtime_error("campello_nn: GpuBackend: unknown DataType");
    }

    int64_t numElements(const std::vector<int64_t> &shape)
    {
        int64_t n = 1;
        for (auto d : shape)
            n *= d;
        return n;
    }

    struct GpuTensor
    {
        std::shared_ptr<cgpu::Buffer> buffer;
        uint64_t byteSize;
    };

    // dispatch() blocks on a real campello_gpu::Fence before returning (see
    // there), so by the time a GpuFence exists the work is already done —
    // same "always pre-signaled" shape as CpuFence/MpsBackend's/
    // DirectMlBackend's fences, all of which are synchronous-dispatch-then-
    // pre-signaled too.
    struct GpuFence
    {
        bool signaled = true;
    };

    // One per OpKind implemented this round — built lazily on first use, reused
    // for every node of that kind in every graph (same "build pipeline once"
    // precedent as the MPSGraph/DirectML backends).
    struct OpResources
    {
        std::shared_ptr<cgpu::BindGroupLayout> bindGroupLayout;
        std::shared_ptr<cgpu::PipelineLayout> pipelineLayout;
        std::shared_ptr<cgpu::ComputePipeline> pipeline;
    };

    // Uniform params layouts — must match the corresponding shader's `Params`
    // struct exactly (see src/gpu/shaders/*.{comp,metal,hlsl}). Padded to 16
    // bytes, a safe alignment for both GLSL std140/Metal/HLSL cbuffer rules.
    struct ParamsElementwise
    {
        uint32_t count, pad0, pad1, pad2;
    };
    struct ParamsMatMul
    {
        uint32_t m, k, n, pad0;
    };

    struct ShaderBytes
    {
        const unsigned char *data;
        unsigned long size;
        const char *entryPoint;
    };

    // Picks the precompiled shader for `kind` matching the platform campello_gpu
    // selected at compile time for this TU (Metal on Apple, SPIR-V everywhere
    // else except DirectX12 — see the #if block above). Entry point names match
    // what's actually declared in each shader source: GLSL's compute entry is
    // always named "main" (a GLSL requirement); the Metal/HLSL sources here are
    // both named "computeMain" by choice.
    ShaderBytes shaderBytesFor(OpKind kind)
    {
#if defined(__APPLE__)
        switch (kind)
        {
        case OpKind::Relu:
            return {relu_metallib_bytes, relu_metallib_bytes_len, "computeMain"};
        case OpKind::Add:
            return {add_metallib_bytes, add_metallib_bytes_len, "computeMain"};
        case OpKind::MatMul:
            return {matmul_metallib_bytes, matmul_metallib_bytes_len, "computeMain"};
        default:
            throw std::runtime_error("campello_nn: GpuBackend: unsupported OpKind");
        }
#elif defined(_WIN32)
        (void)kind;
        throw std::runtime_error(
            "campello_nn: GpuBackend: no precompiled DirectX12 shader bytecode shipped yet "
            "(src/gpu/shaders/*.hlsl are written but unverified — see TODO.md)");
#else
        switch (kind)
        {
        case OpKind::Relu:
            return {relu_spv_bytes, relu_spv_bytes_len, "main"};
        case OpKind::Add:
            return {add_spv_bytes, add_spv_bytes_len, "main"};
        case OpKind::MatMul:
            return {matmul_spv_bytes, matmul_spv_bytes_len, "main"};
        default:
            throw std::runtime_error("campello_nn: GpuBackend: unsupported OpKind");
        }
#endif
    }

    // Every op's binding layout: `numInputs` read-only storage buffers, then one
    // read/write storage buffer (the output), then one uniform buffer (params) —
    // same shape for every op in this round, so one helper covers Relu/Add/MatMul
    // rather than special-casing each op's layout construction.
    std::shared_ptr<cgpu::BindGroupLayout> buildBindGroupLayout(cgpu::Device &device, uint32_t numInputs)
    {
        cgpu::BindGroupLayoutDescriptor desc;
        for (uint32_t i = 0; i < numInputs; i++)
        {
            cgpu::EntryObject e{};
            e.binding = i;
            e.visibility = cgpu::ShaderStage::compute;
            e.type = cgpu::EntryObjectType::buffer;
            e.data.buffer.hasDinamicOffaset = false;
            e.data.buffer.minBindingSize = 0;
            e.data.buffer.type = cgpu::EntryObjectBufferType::readOnlyStorage;
            desc.entries.push_back(e);
        }
        {
            cgpu::EntryObject out{};
            out.binding = numInputs;
            out.visibility = cgpu::ShaderStage::compute;
            out.type = cgpu::EntryObjectType::buffer;
            out.data.buffer.hasDinamicOffaset = false;
            out.data.buffer.minBindingSize = 0;
            out.data.buffer.type = cgpu::EntryObjectBufferType::storage;
            desc.entries.push_back(out);
        }
        {
            cgpu::EntryObject params{};
            params.binding = numInputs + 1;
            params.visibility = cgpu::ShaderStage::compute;
            params.type = cgpu::EntryObjectType::buffer;
            params.data.buffer.hasDinamicOffaset = false;
            params.data.buffer.minBindingSize = 0;
            params.data.buffer.type = cgpu::EntryObjectBufferType::uniform;
            desc.entries.push_back(params);
        }
        auto layout = device.createBindGroupLayout(desc);
        if (!layout)
            throw std::runtime_error("campello_nn: GpuBackend: createBindGroupLayout failed");
        return layout;
    }

    uint32_t numInputsFor(OpKind kind)
    {
        switch (kind)
        {
        case OpKind::Relu:
            return 1;
        case OpKind::Add:
        case OpKind::MatMul:
            return 2;
        default:
            throw std::runtime_error("campello_nn: GpuBackend: unsupported OpKind");
        }
    }

    // One compiled node's resources. `output`/`paramsBuffer` are allocated once
    // at compileGraph() time (both depend only on the static IR shape, not on
    // dispatch-time input values). `bindGroup` is deliberately NOT cached here —
    // campello_gpu's BindGroup is immutable once created (WebGPU-shaped: to
    // rebind a different buffer you create a new BindGroup, unlike D3D12's
    // rebindable descriptor tables that DirectMlBackend's resolveBuffer()
    // pattern relies on), and a node's actual input buffers may trace back to an
    // Input node whose real Tensor differs per dispatch() call. So `dispatch()`
    // rebuilds every real op's BindGroup fresh, every call — simpler and
    // correct, at the cost of optimizing away rebuilds for inputs that
    // provably never change between calls (e.g. pure-Constant subgraphs) —
    // deliberately not done in this vertical-slice round, see TODO.md.
    struct CompiledNode
    {
        OpKind kind;
        std::shared_ptr<cgpu::Buffer> output;       // null for Input
        std::shared_ptr<cgpu::Buffer> paramsBuffer;  // null for Input/Constant
        uint64_t dispatchX = 0, dispatchY = 1, dispatchZ = 1;
    };

    struct CompiledGraph
    {
        GraphIR ir;
        std::vector<CompiledNode> nodes;
    };
}

struct GpuBackend::Impl
{
    std::shared_ptr<cgpu::Device> device;
    std::unordered_map<OpKind, OpResources> opResources;

    OpResources &resourcesFor(OpKind kind)
    {
        auto it = opResources.find(kind);
        if (it != opResources.end())
            return it->second;

        OpResources res;
        uint32_t numInputs = numInputsFor(kind);
        res.bindGroupLayout = buildBindGroupLayout(*device, numInputs);

        cgpu::PipelineLayoutDescriptor playoutDesc;
        playoutDesc.bindGroupLayouts = {res.bindGroupLayout};
        res.pipelineLayout = device->createPipelineLayout(playoutDesc);
        if (!res.pipelineLayout)
            throw std::runtime_error("campello_nn: GpuBackend: createPipelineLayout failed");

        ShaderBytes sb = shaderBytesFor(kind);
        auto module = device->createShaderModule(sb.data, sb.size);
        if (!module)
            throw std::runtime_error("campello_nn: GpuBackend: createShaderModule failed");

        cgpu::ComputePipelineDescriptor cpDesc;
        cpDesc.compute.module = module;
        cpDesc.compute.entryPoint = sb.entryPoint;
        cpDesc.layout = res.pipelineLayout;
        res.pipeline = device->createComputePipeline(cpDesc);
        if (!res.pipeline)
            throw std::runtime_error("campello_nn: GpuBackend: createComputePipeline failed");

        auto [it2, ok] = opResources.emplace(kind, std::move(res));
        (void)ok;
        return it2->second;
    }
};

GpuBackend::GpuBackend()
{
    impl = new Impl();
    impl->device = cgpu::Device::createDefaultDevice(nullptr);
    if (!impl->device)
        throw std::runtime_error("campello_nn: GpuBackend: no campello_gpu device available");
}

GpuBackend::~GpuBackend()
{
    delete impl;
}

void *GpuBackend::createTensor(const TensorDescriptor &desc)
{
    uint64_t size = (uint64_t)elementByteSize(desc.dataType) * (uint64_t)numElements(desc.shape);
    auto buf = impl->device->createBuffer(size, tensorBufferUsage());
    if (!buf)
        throw std::runtime_error("campello_nn: GpuBackend: createBuffer failed");
    return new GpuTensor{buf, size};
}

void GpuBackend::destroyTensor(void *native)
{
    delete (GpuTensor *)native;
}

void GpuBackend::writeTensor(void *native, const void *data, size_t size)
{
    auto t = (GpuTensor *)native;
    if (!t->buffer->upload(0, size, const_cast<void *>(data)))
        throw std::runtime_error("campello_nn: GpuBackend: Buffer::upload failed");
}

void GpuBackend::readTensor(void *native, void *data, size_t size)
{
    auto t = (GpuTensor *)native;
    if (!t->buffer->download(0, size, data))
        throw std::runtime_error("campello_nn: GpuBackend: Buffer::download failed");
}

void *GpuBackend::compileGraph(const GraphIR &ir)
{
    auto compiled = new CompiledGraph();
    compiled->ir = ir;
    size_t n = ir.nodes.size();
    compiled->nodes.resize(n);

    for (size_t i = 0; i < n; i++)
    {
        const Node &node = ir.nodes[i];
        CompiledNode &cn = compiled->nodes[i];
        cn.kind = node.kind;

        switch (node.kind)
        {
        case OpKind::Input:
            continue; // resolved at dispatch time from the caller's `inputs` map

        case OpKind::Constant:
        {
            uint64_t size = (uint64_t)elementByteSize(node.dataType) * (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(size, tensorBufferUsage(),
                                                    (void *)node.constantBytes.data());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (constant) failed");
            continue;
        }

        case OpKind::Relu:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: relu() only supports Float32 in this round");
            impl->resourcesFor(OpKind::Relu);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (relu output) failed");
            ParamsElementwise p{(uint32_t)count, 0, 0, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (relu params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::Add:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: add() only supports Float32 in this round");
            const std::vector<int64_t> &aShape = ir.nodes[node.inputs[0]].shape;
            const std::vector<int64_t> &bShape = ir.nodes[node.inputs[1]].shape;
            if (aShape != node.shape || bShape != node.shape)
                throw std::runtime_error(
                    "campello_nn: GpuBackend: add() with broadcasting is not implemented in this round "
                    "(exact-shape add only)");
            impl->resourcesFor(OpKind::Add);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (add output) failed");
            ParamsElementwise p{(uint32_t)count, 0, 0, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (add params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::MatMul:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: matmul() only supports Float32 in this round");
            const std::vector<int64_t> &aShape = ir.nodes[node.inputs[0]].shape;
            const std::vector<int64_t> &bShape = ir.nodes[node.inputs[1]].shape;
            if (aShape.size() != 2 || bShape.size() != 2)
                throw std::runtime_error(
                    "campello_nn: GpuBackend: batched matmul is not implemented in this round "
                    "(rank-2 unbatched matmul only)");
            impl->resourcesFor(OpKind::MatMul);
            uint64_t m = (uint64_t)aShape[0], k = (uint64_t)aShape[1], outN = (uint64_t)bShape[1];
            cn.output = impl->device->createBuffer(m * outN * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (matmul output) failed");
            ParamsMatMul p{(uint32_t)m, (uint32_t)k, (uint32_t)outN, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (matmul params) failed");
            cn.dispatchX = outN;
            cn.dispatchY = m;
            continue;
        }

        default:
            throw std::runtime_error(
                std::string("campello_nn: GpuBackend: OpKind '") + toString(node.kind) +
                "' is not implemented yet (this round's vertical slice covers Input/Constant/Relu/Add/MatMul only)");
        }
    }

    return compiled;
}

void GpuBackend::destroyGraph(void *native)
{
    delete (CompiledGraph *)native;
}

void *GpuBackend::dispatch(
    void *compiledGraph,
    const std::unordered_map<std::string, void *> &inputs,
    const std::unordered_map<std::string, void *> &outputs)
{
    auto g = (CompiledGraph *)compiledGraph;
    size_t n = g->ir.nodes.size();
    std::vector<std::shared_ptr<cgpu::Buffer>> resolved(n);

    auto encoder = impl->device->createCommandEncoder();
    auto pass = encoder->beginComputePass();

    for (size_t i = 0; i < n; i++)
    {
        const Node &node = g->ir.nodes[i];
        CompiledNode &cn = g->nodes[i];

        if (node.kind == OpKind::Input)
        {
            auto it = inputs.find(node.name);
            if (it == inputs.end())
                throw std::runtime_error("campello_nn: GpuBackend: dispatch() missing input tensor '" + node.name + "'");
            resolved[i] = ((GpuTensor *)it->second)->buffer;
            continue;
        }
        if (node.kind == OpKind::Constant)
        {
            resolved[i] = cn.output;
            continue;
        }

        OpResources &res = impl->resourcesFor(cn.kind);

        cgpu::BindGroupDescriptor bgDesc;
        bgDesc.layout = res.bindGroupLayout;
        uint32_t binding = 0;
        for (size_t inputIdx : node.inputs)
        {
            auto &buf = resolved[inputIdx];
            bgDesc.entries.push_back({binding++, cgpu::BufferBinding{buf, 0, buf->getLength()}});
        }
        bgDesc.entries.push_back({binding++, cgpu::BufferBinding{cn.output, 0, cn.output->getLength()}});
        bgDesc.entries.push_back({binding++, cgpu::BufferBinding{cn.paramsBuffer, 0, cn.paramsBuffer->getLength()}});

        auto bindGroup = impl->device->createBindGroup(bgDesc);
        if (!bindGroup)
            throw std::runtime_error("campello_nn: GpuBackend: createBindGroup failed");

        pass->setPipeline(res.pipeline);
        pass->setBindGroup(0, bindGroup, {}, 0, 0);
        pass->dispatchWorkgroups(cn.dispatchX, cn.dispatchY, cn.dispatchZ);

        resolved[i] = cn.output;
    }

    pass->end();

    for (auto &[name, nodeIdx] : g->ir.outputs)
    {
        auto it = outputs.find(name);
        if (it == outputs.end())
            continue;
        auto dst = (GpuTensor *)it->second;
        encoder->copyBufferToBuffer(resolved[nodeIdx], 0, dst->buffer, 0, dst->byteSize);
    }

    auto cmdBuffer = encoder->finish();
    // Blocks here (synchronous dispatch, same convention the CPU/MPSGraph/
    // DirectML backends already use) via a real campello_gpu::Fence. Needs
    // the local 0.13.3 fix (see TODO.md) to work correctly: a freshly created
    // Fence previously defaulted to already-signaled, so wait() returned
    // immediately without ever waiting for this submission — Device::submit()
    // now resets it to unsignaled right before commit.
    auto fence = impl->device->createFence();
    impl->device->submit(cmdBuffer, fence);
    fence->wait();

    return new GpuFence();
}

bool GpuBackend::waitFence(void *fenceNative, uint64_t)
{
    return ((GpuFence *)fenceNative)->signaled;
}

bool GpuBackend::isFenceSignaled(void *fenceNative)
{
    return ((GpuFence *)fenceNative)->signaled;
}

void GpuBackend::destroyFence(void *fenceNative)
{
    delete (GpuFence *)fenceNative;
}

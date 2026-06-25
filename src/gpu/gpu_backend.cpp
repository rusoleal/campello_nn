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
#include "shaders/mul_metallib.hpp"
#include "shaders/sigmoid_metallib.hpp"
#include "shaders/gelu_metallib.hpp"
#include "shaders/layernorm_metallib.hpp"
#include "shaders/rmsnorm_metallib.hpp"
#include "shaders/transpose_metallib.hpp"
#include "shaders/slice_metallib.hpp"
#include "shaders/concat_metallib.hpp"
#include "shaders/gather_metallib.hpp"
#include "shaders/gemm_metallib.hpp"
#include "shaders/softmax_metallib.hpp"
#include "shaders/conv2d_metallib.hpp"
#include "shaders/pool2d_metallib.hpp"
#include "shaders/resize_metallib.hpp"
#include "shaders/batchnorm_metallib.hpp"
#include "shaders/instancenorm_metallib.hpp"
#include "shaders/quantize_linear_metallib.hpp"
#include "shaders/dequantize_linear_metallib.hpp"
#include "shaders/broadcast_binary_metallib.hpp"
#elif !defined(_WIN32)
#include "shaders/relu_spv.hpp"
#include "shaders/add_spv.hpp"
#include "shaders/matmul_spv.hpp"
#include "shaders/mul_spv.hpp"
#include "shaders/sigmoid_spv.hpp"
#include "shaders/gelu_spv.hpp"
#include "shaders/layernorm_spv.hpp"
#include "shaders/rmsnorm_spv.hpp"
#include "shaders/transpose_spv.hpp"
#include "shaders/slice_spv.hpp"
#include "shaders/concat_spv.hpp"
#include "shaders/gather_spv.hpp"
#include "shaders/gemm_spv.hpp"
#include "shaders/softmax_spv.hpp"
#include "shaders/conv2d_spv.hpp"
#include "shaders/pool2d_spv.hpp"
#include "shaders/resize_spv.hpp"
#include "shaders/batchnorm_spv.hpp"
#include "shaders/instancenorm_spv.hpp"
#include "shaders/quantize_linear_spv.hpp"
#include "shaders/dequantize_linear_spv.hpp"
#include "shaders/broadcast_binary_spv.hpp"
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

    // Row-major strides of `shape` (stride[rank-1] = 1, stride[d] =
    // stride[d+1] * shape[d+1]) — same convention src/cpu/ops.cpp's
    // rowMajorStrides() uses, recomputed here since this TU has no access to
    // that internal CPU-backend helper.
    std::vector<int64_t> rowMajorStrides(const std::vector<int64_t> &shape)
    {
        std::vector<int64_t> strides(shape.size());
        int64_t acc = 1;
        for (size_t d = shape.size(); d-- > 0;)
        {
            strides[d] = acc;
            acc *= shape[d];
        }
        return strides;
    }

    // Transpose/Slice's generic per-dim remap is capped at this rank — both
    // ops' Params structs carry MAX_RANK flat scalar fields (divisor0..7,
    // etc., see transpose.comp's comment for why they're not real arrays).
    // Every real model this library has imported (ONNX/TFLite) stays well
    // under this; throw rather than guess past it, same precedent as every
    // other unsupported-variant case in this codebase.
    constexpr size_t kMaxRank = 8;

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
        uint32_t m, k, n, batchCount;
    };
    // For LayerNorm/RmsNorm's row-per-workgroup dispatch (see those shaders'
    // comments). Mixed uint32/float is fine here — a flat sequence of 4-byte
    // scalars packs identically under GLSL std140, HLSL cbuffer, and MSL
    // constant-struct rules at this size (no vec2/vec3 alignment surprises).
    struct ParamsNorm
    {
        uint32_t lastDim;
        float eps;
        uint32_t pad1, pad2;
    };
    // Transpose's generic arbitrary-rank permute (see transpose.comp's
    // comment for the divisor/gatherStride math). Flat named fields, not
    // real arrays — GLSL std140/HLSL cbuffers pad array elements to 16
    // bytes inside a uniform block, Metal's `constant` structs don't, so a
    // real array would need a different byte layout per backend. Plain
    // scalar fields avoid the question entirely (same reasoning as
    // ParamsNorm just scaled up to kMaxRank fields).
    struct ParamsTranspose
    {
        uint32_t rank, count;
        uint32_t divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
        uint32_t gather0, gather1, gather2, gather3, gather4, gather5, gather6, gather7;
    };
    // Slice's generic per-dim decode — same shape as ParamsTranspose plus a
    // fixed `baseOffset` (see slice.comp's comment).
    struct ParamsSlice
    {
        uint32_t rank, count, baseOffset, pad0;
        uint32_t divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
        uint32_t mult0, mult1, mult2, mult3, mult4, mult5, mult6, mult7;
    };
    // Concat's per-input-dispatch outer/axis/inner split (see concat.comp's
    // comment) — one of these per concat input, not per node.
    struct ParamsConcat
    {
        uint32_t count, axisSizeIn, axisSizeOut, innerSize, axisOffset, pad0, pad1, pad2;
    };
    // Gather's outer/axisSize/innerSize split, matching evalGather's own
    // decomposition in src/cpu/ops.cpp (see gather.comp's comment).
    struct ParamsGather
    {
        uint32_t outerSize, axisSize, innerSize, numIndices;
    };
    // Gemm's alpha/beta/C-broadcast extension of plain matmul (see
    // gemm.comp's comment). Mixed uint32/float, same precedent as
    // ParamsNorm.
    struct ParamsGemm
    {
        uint32_t m, k, n, cElems;
        float alpha, beta;
        uint32_t pad0, pad1;
    };
    // Softmax's row-per-workgroup dispatch with an arbitrary (not
    // necessarily last) reduction axis — same flat-named-field/kMaxRank
    // treatment as ParamsTranspose/ParamsSlice (see softmax.comp's
    // comment), plus the axis's own size/stride and the outer dispatch
    // dimension's total.
    struct ParamsSoftmax
    {
        uint32_t outerRank, axisSize, axisStride, outerTotal;
        uint32_t divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
        uint32_t origStride0, origStride1, origStride2, origStride3, origStride4, origStride5, origStride6, origStride7;
    };
    // Conv2d: NCHW input, OIHW weights, explicit padding/stride/dilation/groups.
    // Matches conv2d.comp's Params block field-for-field.
    struct ParamsConv
    {
        uint32_t N, O, C, H, W, Cg, KH, KW, outH, outW;
        uint32_t strideX, strideY, dilationX, dilationY, paddingLeft, paddingTop;
        uint32_t inPerGroup, outPerGroup;
    };
    // Shared by MaxPool2d and AvgPool2d; `isMax` selects the behavior.
    // Matches pool2d.comp's Params block field-for-field.
    struct ParamsPool
    {
        uint32_t H, W, outH, outW;
        uint32_t kernelHeight, kernelWidth, strideX, strideY, paddingLeft, paddingTop;
        uint32_t isMax;
    };
    // Resize: nearest/bilinear with centerResult/alignCorners/nearestRoundsDown.
    // Matches resize.comp's Params block field-for-field.
    struct ParamsResize
    {
        uint32_t H, W, outH, outW;
        uint32_t mode, centerResult, alignCorners, nearestRoundsDown;
    };
    // BatchNorm: given mean/variance/scale/bias, per-channel affine over NCHW.
    // Matches batchnorm.comp's Params block field-for-field.
    struct ParamsBatchNorm
    {
        uint32_t count;
        uint32_t C;
        uint32_t spatial;
        float eps;
    };
    // InstanceNorm: computes mean/variance per-(N,C) spatial plane.
    // Matches instancenorm.comp's Params block field-for-field.
    struct ParamsInstanceNorm
    {
        uint32_t spatial;
        uint32_t C;
        float eps;
        uint32_t pad0;
    };
    // Shared params layout for QuantizeLinear (Float32 → Int8) and
    // DequantizeLinear (Int8 → Float32). Mixed uint/float/int packs to 16 bytes
    // under GLSL std140, HLSL cbuffer, and MSL constant-struct rules.
    struct ParamsQuantizeDequantize
    {
        uint32_t count;
        float scale;
        int32_t zeroPoint;
        uint32_t pad0;
    };

    // 4 + 8*3 = 28 uint32_t uniform block.
    struct ParamsBroadcast
    {
        uint32_t count;
        uint32_t rank;
        uint32_t mode; // 0 = add, 1 = mul
        uint32_t pad0;
        uint32_t shape[8];
        uint32_t strideA[8];
        uint32_t strideB[8];
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
        case OpKind::Mul:
            return {mul_metallib_bytes, mul_metallib_bytes_len, "computeMain"};
        case OpKind::Sigmoid:
            return {sigmoid_metallib_bytes, sigmoid_metallib_bytes_len, "computeMain"};
        case OpKind::Gelu:
            return {gelu_metallib_bytes, gelu_metallib_bytes_len, "computeMain"};
        case OpKind::LayerNorm:
            return {layernorm_metallib_bytes, layernorm_metallib_bytes_len, "computeMain"};
        case OpKind::RmsNorm:
            return {rmsnorm_metallib_bytes, rmsnorm_metallib_bytes_len, "computeMain"};
        case OpKind::Transpose:
            return {transpose_metallib_bytes, transpose_metallib_bytes_len, "computeMain"};
        case OpKind::Slice:
            return {slice_metallib_bytes, slice_metallib_bytes_len, "computeMain"};
        case OpKind::Concat:
            return {concat_metallib_bytes, concat_metallib_bytes_len, "computeMain"};
        case OpKind::Gather:
            return {gather_metallib_bytes, gather_metallib_bytes_len, "computeMain"};
        case OpKind::Gemm:
            return {gemm_metallib_bytes, gemm_metallib_bytes_len, "computeMain"};
        case OpKind::Softmax:
            return {softmax_metallib_bytes, softmax_metallib_bytes_len, "computeMain"};
        case OpKind::Conv2d:
            return {conv2d_metallib_bytes, conv2d_metallib_bytes_len, "computeMain"};
        case OpKind::MaxPool2d:
        case OpKind::AvgPool2d:
            return {pool2d_metallib_bytes, pool2d_metallib_bytes_len, "computeMain"};
        case OpKind::Resize:
            return {resize_metallib_bytes, resize_metallib_bytes_len, "computeMain"};
        case OpKind::BatchNorm:
            return {batchnorm_metallib_bytes, batchnorm_metallib_bytes_len, "computeMain"};
        case OpKind::InstanceNorm:
            return {instancenorm_metallib_bytes, instancenorm_metallib_bytes_len, "computeMain"};
        case OpKind::QuantizeLinear:
            return {quantize_linear_metallib_bytes, quantize_linear_metallib_bytes_len, "computeMain"};
        case OpKind::DequantizeLinear:
            return {dequantize_linear_metallib_bytes, dequantize_linear_metallib_bytes_len, "computeMain"};
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
        case OpKind::Mul:
            return {mul_spv_bytes, mul_spv_bytes_len, "main"};
        case OpKind::Sigmoid:
            return {sigmoid_spv_bytes, sigmoid_spv_bytes_len, "main"};
        case OpKind::Gelu:
            return {gelu_spv_bytes, gelu_spv_bytes_len, "main"};
        case OpKind::LayerNorm:
            return {layernorm_spv_bytes, layernorm_spv_bytes_len, "main"};
        case OpKind::RmsNorm:
            return {rmsnorm_spv_bytes, rmsnorm_spv_bytes_len, "main"};
        case OpKind::Transpose:
            return {transpose_spv_bytes, transpose_spv_bytes_len, "main"};
        case OpKind::Slice:
            return {slice_spv_bytes, slice_spv_bytes_len, "main"};
        case OpKind::Concat:
            return {concat_spv_bytes, concat_spv_bytes_len, "main"};
        case OpKind::Gather:
            return {gather_spv_bytes, gather_spv_bytes_len, "main"};
        case OpKind::Gemm:
            return {gemm_spv_bytes, gemm_spv_bytes_len, "main"};
        case OpKind::Softmax:
            return {softmax_spv_bytes, softmax_spv_bytes_len, "main"};
        case OpKind::Conv2d:
            return {conv2d_spv_bytes, conv2d_spv_bytes_len, "main"};
        case OpKind::MaxPool2d:
        case OpKind::AvgPool2d:
            return {pool2d_spv_bytes, pool2d_spv_bytes_len, "main"};
        case OpKind::Resize:
            return {resize_spv_bytes, resize_spv_bytes_len, "main"};
        case OpKind::BatchNorm:
            return {batchnorm_spv_bytes, batchnorm_spv_bytes_len, "main"};
        case OpKind::InstanceNorm:
            return {instancenorm_spv_bytes, instancenorm_spv_bytes_len, "main"};
        case OpKind::QuantizeLinear:
            return {quantize_linear_spv_bytes, quantize_linear_spv_bytes_len, "main"};
        case OpKind::DequantizeLinear:
            return {dequantize_linear_spv_bytes, dequantize_linear_spv_bytes_len, "main"};
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
        case OpKind::Sigmoid:
        case OpKind::Gelu:
        case OpKind::Transpose:
        case OpKind::Slice:
        case OpKind::Softmax:
        case OpKind::Concat: // each concatPieces dispatch binds exactly one input — see dispatch()
        case OpKind::MaxPool2d:
        case OpKind::AvgPool2d:
        case OpKind::Resize:
        case OpKind::QuantizeLinear:
        case OpKind::DequantizeLinear:
            return 1;
        case OpKind::Add:
        case OpKind::Mul:
        case OpKind::MatMul:
        case OpKind::RmsNorm: // x, scale
        case OpKind::Gather:  // data, indices
            return 2;
        case OpKind::LayerNorm: // x, scale, bias
        case OpKind::Gemm:      // a, b, c
            return 3;
        case OpKind::Conv2d: // input, weights
            return 2;
        case OpKind::BatchNorm: // x, mean, variance, scale, bias
            return 5;
        case OpKind::InstanceNorm: // x, scale, bias
            return 3;
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
    // Concat has a variable number of real inputs per node (unlike every
    // other op, which has a fixed arity for a given OpKind), so it can't
    // reuse the single-BindGroupLayout-per-OpKind/single-dispatch model the
    // rest of this backend relies on. Instead it's compiled as N independent
    // "copy one input into its axis-offset slice of the (shared) output"
    // pieces — see concat.comp — each with its own params buffer (axisOffset
    // and that input's own axis size differ per piece) but reusing the same
    // 1-input OpResources (Concat's numInputsFor() == 1).
    struct ConcatPiece
    {
        size_t inputIdx;
        std::shared_ptr<cgpu::Buffer> paramsBuffer;
        uint64_t dispatchX;
    };

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
        bool usesBroadcastBinary = false; // true for Add/Mul that need broadcasting
        std::shared_ptr<cgpu::Buffer> output;       // null for Input
        std::shared_ptr<cgpu::Buffer> paramsBuffer;  // null for Input/Constant/Concat
        uint64_t dispatchX = 0, dispatchY = 1, dispatchZ = 1;
        std::vector<ConcatPiece> concatPieces; // only populated for OpKind::Concat
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
    std::optional<OpResources> broadcastBinaryResources;

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

    OpResources &broadcastBinaryResourcesFor()
    {
        if (broadcastBinaryResources)
            return *broadcastBinaryResources;

        OpResources res;
        uint32_t numInputs = 2;
        res.bindGroupLayout = buildBindGroupLayout(*device, numInputs);

        cgpu::PipelineLayoutDescriptor playoutDesc;
        playoutDesc.bindGroupLayouts = {res.bindGroupLayout};
        res.pipelineLayout = device->createPipelineLayout(playoutDesc);
        if (!res.pipelineLayout)
            throw std::runtime_error("campello_nn: GpuBackend: createPipelineLayout failed (broadcast)");

#if defined(__APPLE__)
        ShaderBytes sb{broadcast_binary_metallib_bytes, broadcast_binary_metallib_bytes_len, "computeMain"};
#else
        ShaderBytes sb{broadcast_binary_spv_bytes, broadcast_binary_spv_bytes_len, "main"};
#endif
        auto module = device->createShaderModule(sb.data, sb.size);
        if (!module)
            throw std::runtime_error("campello_nn: GpuBackend: createShaderModule failed (broadcast)");

        cgpu::ComputePipelineDescriptor cpDesc;
        cpDesc.compute.module = module;
        cpDesc.compute.entryPoint = sb.entryPoint;
        cpDesc.layout = res.pipelineLayout;
        res.pipeline = device->createComputePipeline(cpDesc);
        if (!res.pipeline)
            throw std::runtime_error("campello_nn: GpuBackend: createComputePipeline failed (broadcast)");

        broadcastBinaryResources = std::move(res);
        return *broadcastBinaryResources;
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
            // campello_gpu's createBuffer(size=0) returns nullptr on at least the
            // Metal backend (newBuffer(0, ...) returns nil). Some imported models
            // (e.g. YuNet) contain zero-element initializers that are dead in the
            // graph; allocate a 1-byte placeholder so bindings stay valid.
            uint64_t allocSize = size == 0 ? 1 : size;
            // The data-ful createBuffer() always calls upload(); passing a null
            // pointer with a non-zero size would crash inside memcpy. For zero-
            // byte constants use the no-initial-data overload instead.
            cn.output = size == 0
                ? impl->device->createBuffer(allocSize, tensorBufferUsage())
                : impl->device->createBuffer(allocSize, tensorBufferUsage(), (void *)node.constantBytes.data());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (constant) failed");
            continue;
        }

        case OpKind::Reshape:
            // Zero-cost alias, same precedent as the DirectML backend's
            // resolveBuffer(): no shader, no buffer of its own — dispatch()
            // resolves it by aliasing its source node's buffer directly.
            continue;

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
        case OpKind::Mul:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error(
                    std::string("campello_nn: GpuBackend: ") +
                    (node.kind == OpKind::Add ? "add()" : "mul()") +
                    " only supports Float32 in this round");

            const std::vector<int64_t> &aShape = ir.nodes[node.inputs[0]].shape;
            const std::vector<int64_t> &bShape = ir.nodes[node.inputs[1]].shape;
            bool broadcast = (aShape != node.shape || bShape != node.shape);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error(
                    std::string("campello_nn: GpuBackend: createBuffer (") +
                    (node.kind == OpKind::Add ? "add" : "mul") + " output) failed");

            if (broadcast)
            {
                cn.usesBroadcastBinary = true;
                impl->broadcastBinaryResourcesFor();

                size_t rank = node.shape.size();
                if (rank > 8)
                    throw std::runtime_error(
                        "campello_nn: GpuBackend: broadcast binary supports at most rank-8 tensors");

                auto makeStrides = [](const std::vector<int64_t> &shape)
                {
                    std::vector<int64_t> strides(shape.size(), 1);
                    for (int i = (int)shape.size() - 2; i >= 0; --i)
                        strides[i] = strides[i + 1] * shape[i + 1];
                    return strides;
                };

                std::vector<int64_t> aStrides = makeStrides(aShape);
                std::vector<int64_t> bStrides = makeStrides(bShape);

                ParamsBroadcast p{};
                p.count = (uint32_t)count;
                p.rank = (uint32_t)rank;
                p.mode = (node.kind == OpKind::Add) ? 0u : 1u;
                p.pad0 = 0;

                for (size_t i = 0; i < 8; i++)
                {
                    p.shape[i] = 1;
                    p.strideA[i] = 0;
                    p.strideB[i] = 0;
                }
                for (size_t i = 0; i < rank; i++)
                {
                    p.shape[i] = (uint32_t)node.shape[i];
                    size_t aDim = i + aShape.size() - rank;
                    size_t bDim = i + bShape.size() - rank;
                    if (aShape.size() > 0 && aDim < aShape.size())
                    {
                        if (aShape[aDim] == node.shape[i])
                            p.strideA[i] = (uint32_t)aStrides[aDim];
                        // else broadcast (dim == 1) -> stride stays 0
                    }
                    if (bShape.size() > 0 && bDim < bShape.size())
                    {
                        if (bShape[bDim] == node.shape[i])
                            p.strideB[i] = (uint32_t)bStrides[bDim];
                        // else broadcast (dim == 1) -> stride stays 0
                    }
                }

                cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
                if (!cn.paramsBuffer)
                    throw std::runtime_error(
                        std::string("campello_nn: GpuBackend: createBuffer (") +
                        (node.kind == OpKind::Add ? "add" : "mul") + " broadcast params) failed");
                cn.dispatchX = count;
            }
            else
            {
                impl->resourcesFor(node.kind);
                ParamsElementwise p{(uint32_t)count, 0, 0, 0};
                cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
                if (!cn.paramsBuffer)
                    throw std::runtime_error(
                        std::string("campello_nn: GpuBackend: createBuffer (") +
                        (node.kind == OpKind::Add ? "add" : "mul") + " params) failed");
                cn.dispatchX = count;
            }
            continue;
        }

        case OpKind::MatMul:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: matmul() only supports Float32 in this round");
            const std::vector<int64_t> &aShape = ir.nodes[node.inputs[0]].shape;
            const std::vector<int64_t> &bShape = ir.nodes[node.inputs[1]].shape;
            if (aShape.size() < 2 || bShape.size() < 2)
                throw std::runtime_error("campello_nn: GpuBackend: matmul() operands must have rank >= 2");
            if (aShape.size() != bShape.size())
                throw std::runtime_error(
                    "campello_nn: GpuBackend: matmul() operands must have the same rank "
                    "(no implicit broadcasting yet)");
            size_t rank = aShape.size();
            for (size_t i = 0; i < rank - 2; i++)
                if (aShape[i] != bShape[i])
                    throw std::runtime_error(
                        "campello_nn: GpuBackend: matmul() batch dimensions must match");

            impl->resourcesFor(OpKind::MatMul);
            uint64_t m = (uint64_t)aShape[rank - 2];
            uint64_t k = (uint64_t)aShape[rank - 1];
            uint64_t outN = (uint64_t)bShape[rank - 1];
            uint64_t batchCount = 1;
            for (size_t i = 0; i < rank - 2; i++)
                batchCount *= (uint64_t)aShape[i];

            uint64_t outElems = batchCount * m * outN;
            cn.output = impl->device->createBuffer(outElems * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (matmul output) failed");
            constexpr uint32_t kMatMulTileWidth = 8;
            ParamsMatMul p{(uint32_t)m, (uint32_t)k, (uint32_t)outN, (uint32_t)batchCount};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (matmul params) failed");
            cn.dispatchX = (outN + kMatMulTileWidth - 1) / kMatMulTileWidth;
            cn.dispatchY = m;
            cn.dispatchZ = batchCount;
            continue;
        }



        case OpKind::Sigmoid:
        case OpKind::Gelu:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error(
                    std::string("campello_nn: GpuBackend: ") + toString(node.kind) +
                    "() only supports Float32 in this round");
            impl->resourcesFor(node.kind);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (output) failed");
            ParamsElementwise p{(uint32_t)count, 0, 0, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::LayerNorm:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: layerNorm() only supports Float32 in this round");
            impl->resourcesFor(OpKind::LayerNorm);
            int64_t lastDim = node.shape.back();
            int64_t outerTotal = numElements(node.shape) / lastDim;
            cn.output = impl->device->createBuffer((uint64_t)numElements(node.shape) * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (layerNorm output) failed");
            ParamsNorm p{(uint32_t)lastDim, node.floatAttr0, 0, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (layerNorm params) failed");
            cn.dispatchX = (uint64_t)outerTotal;
            continue;
        }

        case OpKind::RmsNorm:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: rmsNorm() only supports Float32 in this round");
            impl->resourcesFor(OpKind::RmsNorm);
            int64_t lastDim = node.shape.back();
            int64_t outerTotal = numElements(node.shape) / lastDim;
            cn.output = impl->device->createBuffer((uint64_t)numElements(node.shape) * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (rmsNorm output) failed");
            ParamsNorm p{(uint32_t)lastDim, node.floatAttr0, 0, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (rmsNorm params) failed");
            cn.dispatchX = (uint64_t)outerTotal;
            continue;
        }

        case OpKind::Transpose:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: transpose() only supports Float32 in this round");
            size_t rank = node.shape.size();
            if (rank > kMaxRank)
                throw std::runtime_error("campello_nn: GpuBackend: transpose() rank > 8 is not implemented in this round");
            const std::vector<int64_t> &inShape = ir.nodes[node.inputs[0]].shape;
            std::vector<int64_t> inStrides = rowMajorStrides(inShape);
            std::vector<int64_t> outStrides = rowMajorStrides(node.shape);
            const auto &perm = node.intAttr0;
            impl->resourcesFor(OpKind::Transpose);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (transpose output) failed");
            uint32_t divisor[kMaxRank] = {0}, gather[kMaxRank] = {0};
            for (size_t d = 0; d < rank; d++)
            {
                divisor[d] = (uint32_t)outStrides[d];
                gather[d] = (uint32_t)inStrides[perm[d]];
            }
            ParamsTranspose p{(uint32_t)rank, (uint32_t)count,
                               divisor[0], divisor[1], divisor[2], divisor[3], divisor[4], divisor[5], divisor[6], divisor[7],
                               gather[0], gather[1], gather[2], gather[3], gather[4], gather[5], gather[6], gather[7]};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (transpose params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::Slice:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: slice() only supports Float32 in this round");
            size_t rank = node.shape.size();
            if (rank > kMaxRank)
                throw std::runtime_error("campello_nn: GpuBackend: slice() rank > 8 is not implemented in this round");
            const std::vector<int64_t> &inShape = ir.nodes[node.inputs[0]].shape;
            std::vector<int64_t> inStrides = rowMajorStrides(inShape);
            std::vector<int64_t> outStrides = rowMajorStrides(node.shape);
            const auto &starts = node.intAttr0;
            int64_t baseOffset = 0;
            for (size_t d = 0; d < rank; d++)
                baseOffset += starts[d] * inStrides[d];
            impl->resourcesFor(OpKind::Slice);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (slice output) failed");
            uint32_t divisor[kMaxRank] = {0}, mult[kMaxRank] = {0};
            for (size_t d = 0; d < rank; d++)
            {
                divisor[d] = (uint32_t)outStrides[d];
                mult[d] = (uint32_t)inStrides[d];
            }
            ParamsSlice p{(uint32_t)rank, (uint32_t)count, (uint32_t)baseOffset, 0,
                          divisor[0], divisor[1], divisor[2], divisor[3], divisor[4], divisor[5], divisor[6], divisor[7],
                          mult[0], mult[1], mult[2], mult[3], mult[4], mult[5], mult[6], mult[7]};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (slice params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::Concat:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: concat() only supports Float32 in this round");
            // node.axis is already resolved to a non-negative index by
            // GraphBuilder::concat() (graph_builder.cpp), so no further
            // normalization is needed here.
            int64_t axis = node.axis;
            int64_t axisSizeOut = node.shape[axis];
            int64_t innerSize = 1;
            for (size_t d = axis + 1; d < node.shape.size(); d++)
                innerSize *= node.shape[d];
            impl->resourcesFor(OpKind::Concat);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (concat output) failed");
            int64_t axisOffset = 0;
            for (size_t inputIdx : node.inputs)
            {
                const std::vector<int64_t> &pieceShape = ir.nodes[inputIdx].shape;
                int64_t axisSizeIn = pieceShape[axis];
                uint64_t pieceCount = (uint64_t)numElements(pieceShape);
                ParamsConcat p{(uint32_t)pieceCount, (uint32_t)axisSizeIn, (uint32_t)axisSizeOut,
                               (uint32_t)innerSize, (uint32_t)axisOffset, 0, 0, 0};
                auto paramsBuf = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
                if (!paramsBuf)
                    throw std::runtime_error("campello_nn: GpuBackend: createBuffer (concat piece params) failed");
                cn.concatPieces.push_back({inputIdx, paramsBuf, pieceCount});
                axisOffset += axisSizeIn;
            }
            continue;
        }

        case OpKind::Gather:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: gather() only supports Float32 data in this round");
            const std::vector<int64_t> &dataShape = ir.nodes[node.inputs[0]].shape;
            const std::vector<int64_t> &indicesShape = ir.nodes[node.inputs[1]].shape;
            // node.axis is already resolved to a non-negative index by
            // GraphBuilder::gather() (graph_builder.cpp).
            int64_t axis = node.axis;
            int64_t outerSize = 1;
            for (int64_t d = 0; d < axis; d++)
                outerSize *= dataShape[d];
            int64_t axisSize = dataShape[axis];
            int64_t innerSize = 1;
            for (size_t d = axis + 1; d < dataShape.size(); d++)
                innerSize *= dataShape[d];
            int64_t numIndices = numElements(indicesShape);
            impl->resourcesFor(OpKind::Gather);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (gather output) failed");
            ParamsGather p{(uint32_t)outerSize, (uint32_t)axisSize, (uint32_t)innerSize, (uint32_t)numIndices};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (gather params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::Gemm:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: gemm() only supports Float32 in this round");
            // GraphBuilder::gemm() (graph_builder.cpp) already requires a/b
            // to be rank-2 with no transA/transB — nothing further to
            // validate here.
            const std::vector<int64_t> &aShape = ir.nodes[node.inputs[0]].shape;
            const std::vector<int64_t> &bShape = ir.nodes[node.inputs[1]].shape;
            const std::vector<int64_t> &cShape = ir.nodes[node.inputs[2]].shape;
            uint64_t m = (uint64_t)aShape[0], k = (uint64_t)aShape[1], outN = (uint64_t)bShape[1];
            uint64_t cElems = (uint64_t)numElements(cShape);
            impl->resourcesFor(OpKind::Gemm);
            cn.output = impl->device->createBuffer(m * outN * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (gemm output) failed");
            ParamsGemm p{(uint32_t)m, (uint32_t)k, (uint32_t)outN, (uint32_t)cElems, node.floatAttr0, node.floatAttr1, 0, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (gemm params) failed");
            cn.dispatchX = outN;
            cn.dispatchY = m;
            continue;
        }

        case OpKind::Softmax:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: softmax() only supports Float32 in this round");
            size_t rank = node.shape.size();
            if (rank > kMaxRank + 1) // outer (rank-1 dims) must itself fit kMaxRank
                throw std::runtime_error("campello_nn: GpuBackend: softmax() rank too high for this round");
            std::vector<int64_t> strides = rowMajorStrides(node.shape);
            // node.axis is already resolved to a non-negative index by
            // GraphBuilder::softmax() (graph_builder.cpp).
            int64_t axis = node.axis;
            uint64_t axisSize = (uint64_t)node.shape[axis];
            uint64_t axisStride = (uint64_t)strides[axis];
            std::vector<int64_t> outerShape;
            std::vector<int64_t> outerOrigStride;
            for (size_t d = 0; d < rank; d++)
            {
                if ((int64_t)d == axis)
                    continue;
                outerShape.push_back(node.shape[d]);
                outerOrigStride.push_back(strides[d]);
            }
            std::vector<int64_t> outerDivisor = rowMajorStrides(outerShape);
            uint64_t outerTotal = (uint64_t)numElements(outerShape);
            impl->resourcesFor(OpKind::Softmax);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (softmax output) failed");
            uint32_t divisor[kMaxRank] = {0}, origStride[kMaxRank] = {0};
            for (size_t d = 0; d < outerShape.size(); d++)
            {
                divisor[d] = (uint32_t)outerDivisor[d];
                origStride[d] = (uint32_t)outerOrigStride[d];
            }
            ParamsSoftmax p{(uint32_t)outerShape.size(), (uint32_t)axisSize, (uint32_t)axisStride, (uint32_t)outerTotal,
                            divisor[0], divisor[1], divisor[2], divisor[3], divisor[4], divisor[5], divisor[6], divisor[7],
                            origStride[0], origStride[1], origStride[2], origStride[3],
                            origStride[4], origStride[5], origStride[6], origStride[7]};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(p), cgpu::BufferUsage::uniform, &p);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (softmax params) failed");
            cn.dispatchX = outerTotal;
            continue;
        }

        case OpKind::Conv2d:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: conv2d() only supports Float32 in this round");
            const std::vector<int64_t> &xShape = ir.nodes[node.inputs[0]].shape;
            const std::vector<int64_t> &wShape = ir.nodes[node.inputs[1]].shape;
            if (xShape.size() != 4 || wShape.size() != 4 || node.shape.size() != 4)
                throw std::runtime_error("campello_nn: GpuBackend: conv2d() expects rank-4 tensors");
            const Conv2dDescriptor &p = node.convParams;
            uint32_t N = (uint32_t)xShape[0];
            uint32_t C = (uint32_t)xShape[1];
            uint32_t H = (uint32_t)xShape[2];
            uint32_t W = (uint32_t)xShape[3];
            uint32_t O = (uint32_t)wShape[0];
            uint32_t Cg = (uint32_t)wShape[1];
            uint32_t KH = (uint32_t)wShape[2];
            uint32_t KW = (uint32_t)wShape[3];
            uint32_t outH = (uint32_t)node.shape[2];
            uint32_t outW = (uint32_t)node.shape[3];
            uint32_t groups = (uint32_t)p.groups;
            uint32_t inPerGroup = C / groups;
            uint32_t outPerGroup = O / groups;
            impl->resourcesFor(OpKind::Conv2d);
            uint64_t outCount = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(outCount * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (conv2d output) failed");
            ParamsConv params{N, O, C, H, W, Cg, KH, KW, outH, outW,
                              (uint32_t)p.strideX, (uint32_t)p.strideY,
                              (uint32_t)p.dilationX, (uint32_t)p.dilationY,
                              (uint32_t)p.paddingLeft, (uint32_t)p.paddingTop,
                              inPerGroup, outPerGroup};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(params), cgpu::BufferUsage::uniform, &params);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (conv2d params) failed");
            constexpr uint32_t kConvTileWidth = 8;
            uint64_t totalOutputs = (uint64_t)N * O * outH * outW;
            cn.dispatchX = (totalOutputs + kConvTileWidth - 1) / kConvTileWidth;
            cn.dispatchY = 1;
            cn.dispatchZ = 1;
            continue;
        }

        case OpKind::MaxPool2d:
        case OpKind::AvgPool2d:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: pool2d() only supports Float32 in this round");
            const std::vector<int64_t> &xShape = ir.nodes[node.inputs[0]].shape;
            if (xShape.size() != 4 || node.shape.size() != 4)
                throw std::runtime_error("campello_nn: GpuBackend: pool2d() expects rank-4 tensors");
            const Pool2dDescriptor &p = node.poolParams;
            uint32_t H = (uint32_t)xShape[2];
            uint32_t W = (uint32_t)xShape[3];
            uint32_t outH = (uint32_t)node.shape[2];
            uint32_t outW = (uint32_t)node.shape[3];
            uint32_t N = (uint32_t)node.shape[0];
            uint32_t C = (uint32_t)node.shape[1];
            OpKind poolKind = node.kind;
            impl->resourcesFor(poolKind);
            uint64_t outCount = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(outCount * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (pool2d output) failed");
            ParamsPool params{H, W, outH, outW,
                              (uint32_t)p.kernelHeight, (uint32_t)p.kernelWidth,
                              (uint32_t)p.strideX, (uint32_t)p.strideY,
                              (uint32_t)p.paddingLeft, (uint32_t)p.paddingTop,
                              poolKind == OpKind::MaxPool2d ? 1u : 0u};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(params), cgpu::BufferUsage::uniform, &params);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (pool2d params) failed");
            cn.dispatchX = outW;
            cn.dispatchY = outH;
            cn.dispatchZ = (uint64_t)N * C;
            continue;
        }

        case OpKind::Resize:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: resize() only supports Float32 in this round");
            const std::vector<int64_t> &xShape = ir.nodes[node.inputs[0]].shape;
            if (xShape.size() != 4 || node.shape.size() != 4)
                throw std::runtime_error("campello_nn: GpuBackend: resize() expects rank-4 tensors");
            const ResizeDescriptor &p = node.resizeParams;
            uint32_t H = (uint32_t)xShape[2];
            uint32_t W = (uint32_t)xShape[3];
            uint32_t outH = (uint32_t)node.shape[2];
            uint32_t outW = (uint32_t)node.shape[3];
            uint32_t N = (uint32_t)node.shape[0];
            uint32_t C = (uint32_t)node.shape[1];
            impl->resourcesFor(OpKind::Resize);
            uint64_t outCount = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(outCount * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (resize output) failed");
            ParamsResize params{H, W, outH, outW,
                                p.mode == ResizeMode::Nearest ? 0u : 1u,
                                p.centerResult ? 1u : 0u,
                                p.alignCorners ? 1u : 0u,
                                p.nearestRoundsDown ? 1u : 0u};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(params), cgpu::BufferUsage::uniform, &params);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (resize params) failed");
            cn.dispatchX = outW;
            cn.dispatchY = outH;
            cn.dispatchZ = (uint64_t)N * C;
            continue;
        }

        case OpKind::BatchNorm:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: batchNorm() only supports Float32 in this round");
            if (node.shape.size() != 4)
                throw std::runtime_error("campello_nn: GpuBackend: batchNorm() expects rank-4 tensors");
            uint32_t N = (uint32_t)node.shape[0];
            uint32_t C = (uint32_t)node.shape[1];
            uint32_t H = (uint32_t)node.shape[2];
            uint32_t W = (uint32_t)node.shape[3];
            uint32_t spatial = H * W;
            uint64_t count = (uint64_t)numElements(node.shape);
            impl->resourcesFor(OpKind::BatchNorm);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (batchNorm output) failed");
            ParamsBatchNorm params{(uint32_t)count, C, spatial, node.floatAttr0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(params), cgpu::BufferUsage::uniform, &params);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (batchNorm params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::InstanceNorm:
        {
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: instanceNorm() only supports Float32 in this round");
            if (node.shape.size() != 4)
                throw std::runtime_error("campello_nn: GpuBackend: instanceNorm() expects rank-4 tensors");
            uint32_t N = (uint32_t)node.shape[0];
            uint32_t C = (uint32_t)node.shape[1];
            uint32_t H = (uint32_t)node.shape[2];
            uint32_t W = (uint32_t)node.shape[3];
            uint32_t spatial = H * W;
            impl->resourcesFor(OpKind::InstanceNorm);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (instanceNorm output) failed");
            ParamsInstanceNorm params{spatial, C, node.floatAttr0, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(params), cgpu::BufferUsage::uniform, &params);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (instanceNorm params) failed");
            cn.dispatchX = (uint64_t)N * C;
            continue;
        }

        case OpKind::QuantizeLinear:
        {
            // Input is Float32; output is Int8. The IR node's dataType is Int8.
            if (node.dataType != DataType::Int8)
                throw std::runtime_error("campello_nn: GpuBackend: quantizeLinear() output must be Int8");
            const Node &inputNode = ir.nodes[node.inputs[0]];
            if (inputNode.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: quantizeLinear() input must be Float32");
            impl->resourcesFor(OpKind::QuantizeLinear);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * elementByteSize(DataType::Int8), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (quantizeLinear output) failed");
            ParamsQuantizeDequantize params{(uint32_t)count, node.floatAttr0, (int32_t)node.floatAttr1, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(params), cgpu::BufferUsage::uniform, &params);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (quantizeLinear params) failed");
            cn.dispatchX = count;
            continue;
        }

        case OpKind::DequantizeLinear:
        {
            // Input is Int8; output is Float32. The IR node's dataType is Float32.
            if (node.dataType != DataType::Float32)
                throw std::runtime_error("campello_nn: GpuBackend: dequantizeLinear() output must be Float32");
            const Node &inputNode = ir.nodes[node.inputs[0]];
            if (inputNode.dataType != DataType::Int8)
                throw std::runtime_error("campello_nn: GpuBackend: dequantizeLinear() input must be Int8");
            impl->resourcesFor(OpKind::DequantizeLinear);
            uint64_t count = (uint64_t)numElements(node.shape);
            cn.output = impl->device->createBuffer(count * sizeof(float), tensorBufferUsage());
            if (!cn.output)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (dequantizeLinear output) failed");
            ParamsQuantizeDequantize params{(uint32_t)count, node.floatAttr0, (int32_t)node.floatAttr1, 0};
            cn.paramsBuffer = impl->device->createBuffer(sizeof(params), cgpu::BufferUsage::uniform, &params);
            if (!cn.paramsBuffer)
                throw std::runtime_error("campello_nn: GpuBackend: createBuffer (dequantizeLinear params) failed");
            cn.dispatchX = count;
            continue;
        }

        default:
            throw std::runtime_error(
                std::string("campello_nn: GpuBackend: OpKind '") + toString(node.kind) +
                "' is not implemented yet (this round's coverage: Input/Constant/Relu/Add/Mul/Sigmoid/"
                "Gelu/LayerNorm/RmsNorm/MatMul/Reshape/Transpose/Slice/Concat/Gather/Gemm/Softmax/"
                "Conv2d/MaxPool2d/AvgPool2d/Resize/BatchNorm/InstanceNorm — "
                "see TODO.md)");
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
        if (node.kind == OpKind::Reshape)
        {
            // node.inputs[0] is always an earlier index (the IR is built in
            // dependency order), so it's already been resolved by this point
            // in the loop — no recursion needed (unlike DirectML's
            // resolveBuffer(), which resolves lazily and so does need to
            // walk Reshape-of-Reshape chains itself).
            resolved[i] = resolved[node.inputs[0]];
            continue;
        }
        if (node.kind == OpKind::Concat)
        {
            // Variable arity, so unlike every other op this is N separate
            // dispatches against the same OpResources (Concat's
            // numInputsFor() == 1) rather than one dispatch with N bound
            // inputs — see CompiledNode::concatPieces's comment.
            OpResources &res = impl->resourcesFor(OpKind::Concat);
            pass->setPipeline(res.pipeline);
            for (auto &piece : cn.concatPieces)
            {
                auto &buf = resolved[piece.inputIdx];
                cgpu::BindGroupDescriptor bgDesc;
                bgDesc.layout = res.bindGroupLayout;
                bgDesc.entries.push_back({0, cgpu::BufferBinding{buf, 0, buf->getLength()}});
                bgDesc.entries.push_back({1, cgpu::BufferBinding{cn.output, 0, cn.output->getLength()}});
                bgDesc.entries.push_back(
                    {2, cgpu::BufferBinding{piece.paramsBuffer, 0, piece.paramsBuffer->getLength()}});
                auto bindGroup = impl->device->createBindGroup(bgDesc);
                if (!bindGroup)
                    throw std::runtime_error("campello_nn: GpuBackend: createBindGroup (concat piece) failed");
                pass->setBindGroup(0, bindGroup, {}, 0, 0);
                pass->dispatchWorkgroups(piece.dispatchX, 1, 1);
            }
            resolved[i] = cn.output;
            continue;
        }

        OpResources &res = cn.usesBroadcastBinary
                              ? impl->broadcastBinaryResourcesFor()
                              : impl->resourcesFor(cn.kind);

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

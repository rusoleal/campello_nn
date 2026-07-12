#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "mps_backend.hpp"
#include "ggml_quant_metal.hpp"
#include "../pi/ir.hpp"

using namespace systems::leal::campello_nn;

namespace
{
    // Mirrors the switch in ggml_quant_metal.hpp's dequantizedWeight(): any type not
    // listed here has no Metal dequant kernel and must fail loudly rather than
    // silently dispatching a kernel that returns 0.0f for every weight element.
    void requireSupportedGgmlType(uint32_t ggmlType)
    {
        switch (ggmlType)
        {
        case 2: case 3: case 6: case 7: case 8: case 9:
        case 10: case 11: case 12: case 13: case 14: case 15:
            return;
        default:
            throw std::runtime_error("campello_nn: MPS backend has no Metal dequant kernel for GGML type " +
                                      std::to_string(ggmlType));
        }
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
        throw std::runtime_error("campello_nn: unknown DataType");
    }

    int64_t numElements(const std::vector<int64_t> &shape)
    {
        int64_t n = 1;
        for (auto d : shape)
            n *= d;
        return n;
    }

    MPSDataType mpsDataType(DataType dt)
    {
        switch (dt)
        {
        case DataType::Float32: return MPSDataTypeFloat32;
        case DataType::Float16: return MPSDataTypeFloat16;
        case DataType::Int32: return MPSDataTypeInt32;
        case DataType::Uint32: return MPSDataTypeUInt32;
        case DataType::Int8: return MPSDataTypeInt8;
        }
        throw std::runtime_error("campello_nn: unknown DataType");
    }

    MPSShape *shapeFor(const std::vector<int64_t> &shape)
    {
        NSMutableArray<NSNumber *> *arr = [NSMutableArray arrayWithCapacity:shape.size()];
        for (auto d : shape)
            [arr addObject:@(d)];
        return arr;
    }

    NSString *nameFor(const std::string &s)
    {
        return s.empty() ? nil : [NSString stringWithUTF8String:s.c_str()];
    }

    MPSGraphTensor *reshapeTrailing(MPSGraph *graph, MPSGraphTensor *t, size_t rank, int64_t lastDim)
    {
        std::vector<int64_t> shape(rank, 1);
        shape[rank - 1] = lastDim;
        return [graph reshapeTensor:t withShape:shapeFor(shape) name:nil];
    }

    MPSGraphTensor *reshapeGemmC(MPSGraph *graph, MPSGraphTensor *c, int64_t cElems, int64_t N)
    {
        std::vector<int64_t> shape = cElems == 1 ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1, N};
        if (cElems != 1 && cElems != N)
            return c;
        return [graph reshapeTensor:c withShape:shapeFor(shape) name:nil];
    }

    MPSGraphTensor *reshapeChannel(MPSGraph *graph, MPSGraphTensor *t, int64_t C)
    {
        return [graph reshapeTensor:t withShape:shapeFor({1, C, 1, 1}) name:nil];
    }

    MPSGraphTensor *buildMpsNode(MPSGraph *graph, const Node &node, const std::vector<MPSGraphTensor *> &tensors)
    {
        switch (node.kind)
        {
        case OpKind::Input:
            return [graph placeholderWithShape:shapeFor(node.shape)
                                        dataType:mpsDataType(node.dataType)
                                            name:nameFor(node.name)];
        case OpKind::Constant:
        {
            NSData *data = [NSData dataWithBytes:node.constantBytes.data() length:node.constantBytes.size()];
            return [graph constantWithData:data shape:shapeFor(node.shape) dataType:mpsDataType(node.dataType)];
        }
        case OpKind::Add:
            return [graph additionWithPrimaryTensor:tensors[node.inputs[0]]
                                      secondaryTensor:tensors[node.inputs[1]]
                                                 name:nil];
        case OpKind::Mul:
            return [graph multiplicationWithPrimaryTensor:tensors[node.inputs[0]]
                                            secondaryTensor:tensors[node.inputs[1]]
                                                       name:nil];
        case OpKind::Gelu:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            MPSDataType dt = mpsDataType(node.dataType);
            MPSGraphTensor *invSqrt2 = [graph constantWithScalar:0.70710678118654752 dataType:dt];
            MPSGraphTensor *half = [graph constantWithScalar:0.5 dataType:dt];
            MPSGraphTensor *one = [graph constantWithScalar:1.0 dataType:dt];
            MPSGraphTensor *erfTerm = [graph erfWithTensor:[graph multiplicationWithPrimaryTensor:x
                                                                                  secondaryTensor:invSqrt2
                                                                                             name:nil]
                                                      name:nil];
            MPSGraphTensor *onePlusErf = [graph additionWithPrimaryTensor:erfTerm secondaryTensor:one name:nil];
            MPSGraphTensor *halfX = [graph multiplicationWithPrimaryTensor:x secondaryTensor:half name:nil];
            return [graph multiplicationWithPrimaryTensor:halfX secondaryTensor:onePlusErf name:nil];
        }
        case OpKind::Relu:
            return [graph reLUWithTensor:tensors[node.inputs[0]] name:nil];
        case OpKind::Sigmoid:
            return [graph sigmoidWithTensor:tensors[node.inputs[0]] name:nil];
        case OpKind::Softmax:
            return [graph softMaxWithTensor:tensors[node.inputs[0]] axis:(NSInteger)node.axis name:nil];
        case OpKind::LayerNorm:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            NSInteger lastAxis = (NSInteger)node.shape.size() - 1;
            NSArray<NSNumber *> *axes = @[ @(lastAxis) ];
            MPSGraphTensor *mean = [graph meanOfTensor:x axes:axes name:nil];
            MPSGraphTensor *variance = [graph varianceOfTensor:x meanTensor:mean axes:axes name:nil];
            MPSGraphTensor *gamma = reshapeTrailing(graph, tensors[node.inputs[1]], node.shape.size(), node.shape.back());
            MPSGraphTensor *beta = reshapeTrailing(graph, tensors[node.inputs[2]], node.shape.size(), node.shape.back());
            return [graph normalizationWithTensor:x
                                       meanTensor:mean
                                   varianceTensor:variance
                                      gammaTensor:gamma
                                       betaTensor:beta
                                          epsilon:node.floatAttr0
                                             name:nil];
        }
        case OpKind::RmsNorm:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            MPSDataType dt = mpsDataType(node.dataType);
            NSInteger lastAxis = (NSInteger)node.shape.size() - 1;
            NSArray<NSNumber *> *axes = @[ @(lastAxis) ];
            MPSGraphTensor *square = [graph squareWithTensor:x name:nil];
            MPSGraphTensor *meanSquare = [graph meanOfTensor:square axes:axes name:nil];
            MPSGraphTensor *epsConst = [graph constantWithScalar:(double)node.floatAttr0 dataType:dt];
            MPSGraphTensor *meanSquarePlusEps = [graph additionWithPrimaryTensor:meanSquare
                                                                  secondaryTensor:epsConst
                                                                             name:nil];
            MPSGraphTensor *invRms = [graph reciprocalSquareRootWithTensor:meanSquarePlusEps name:nil];
            MPSGraphTensor *gamma = reshapeTrailing(graph, tensors[node.inputs[1]], node.shape.size(), node.shape.back());
            MPSGraphTensor *normalized = [graph multiplicationWithPrimaryTensor:x secondaryTensor:invRms name:nil];
            return [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:gamma name:nil];
        }
        case OpKind::BatchNorm:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            int64_t C = node.shape[1];
            MPSGraphTensor *mean = reshapeChannel(graph, tensors[node.inputs[1]], C);
            MPSGraphTensor *variance = reshapeChannel(graph, tensors[node.inputs[2]], C);
            MPSGraphTensor *gamma = reshapeChannel(graph, tensors[node.inputs[3]], C);
            MPSGraphTensor *beta = reshapeChannel(graph, tensors[node.inputs[4]], C);
            return [graph normalizationWithTensor:x
                                       meanTensor:mean
                                   varianceTensor:variance
                                      gammaTensor:gamma
                                       betaTensor:beta
                                          epsilon:node.floatAttr0
                                             name:nil];
        }
        case OpKind::InstanceNorm:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            int64_t C = node.shape[1];
            NSArray<NSNumber *> *axes = @[ @2, @3 ];
            MPSGraphTensor *mean = [graph meanOfTensor:x axes:axes name:nil];
            MPSGraphTensor *variance = [graph varianceOfTensor:x meanTensor:mean axes:axes name:nil];
            MPSGraphTensor *gamma = reshapeChannel(graph, tensors[node.inputs[1]], C);
            MPSGraphTensor *beta = reshapeChannel(graph, tensors[node.inputs[2]], C);
            return [graph normalizationWithTensor:x
                                       meanTensor:mean
                                   varianceTensor:variance
                                      gammaTensor:gamma
                                       betaTensor:beta
                                          epsilon:node.floatAttr0
                                             name:nil];
        }
        case OpKind::MatMul:
            return [graph matrixMultiplicationWithPrimaryTensor:tensors[node.inputs[0]]
                                                 secondaryTensor:tensors[node.inputs[1]]
                                                            name:nil];
        case OpKind::Gemm:
        {
            int64_t N = node.shape[1];
            MPSGraphTensor *mm = [graph matrixMultiplicationWithPrimaryTensor:tensors[node.inputs[0]]
                                                               secondaryTensor:tensors[node.inputs[1]]
                                                                          name:nil];
            MPSGraphTensor *alpha = [graph constantWithScalar:node.floatAttr0 dataType:mpsDataType(node.dataType)];
            MPSGraphTensor *beta = [graph constantWithScalar:node.floatAttr1 dataType:mpsDataType(node.dataType)];
            MPSGraphTensor *scaledMm = [graph multiplicationWithPrimaryTensor:mm secondaryTensor:alpha name:nil];
            MPSGraphTensor *c = tensors[node.inputs[2]];
            int64_t cCount = 1;
            for (NSNumber *d in c.shape)
                cCount *= d.longLongValue;
            MPSGraphTensor *cReshaped = reshapeGemmC(graph, c, cCount, N);
            MPSGraphTensor *scaledC = [graph multiplicationWithPrimaryTensor:cReshaped secondaryTensor:beta name:nil];
            return [graph additionWithPrimaryTensor:scaledMm secondaryTensor:scaledC name:nil];
        }
        case OpKind::Reshape:
            return [graph reshapeTensor:tensors[node.inputs[0]] withShape:shapeFor(node.shape) name:nil];
        case OpKind::Transpose:
        {
            NSMutableArray<NSNumber *> *perm = [NSMutableArray arrayWithCapacity:node.intAttr0.size()];
            for (auto p : node.intAttr0)
                [perm addObject:@(p)];
            return [graph transposeTensor:tensors[node.inputs[0]] permutation:perm name:nil];
        }
        case OpKind::Concat:
        {
            NSMutableArray<MPSGraphTensor *> *inputs = [NSMutableArray arrayWithCapacity:node.inputs.size()];
            for (auto idx : node.inputs)
                [inputs addObject:tensors[idx]];
            return [graph concatTensors:inputs dimension:(NSInteger)node.axis name:nil];
        }
        case OpKind::Slice:
        {
            NSMutableArray<NSNumber *> *starts = [NSMutableArray arrayWithCapacity:node.intAttr0.size()];
            NSMutableArray<NSNumber *> *ends = [NSMutableArray arrayWithCapacity:node.intAttr0.size()];
            NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:node.intAttr0.size()];
            for (size_t d = 0; d < node.intAttr0.size(); d++)
            {
                [starts addObject:@(node.intAttr0[d])];
                [ends addObject:@(node.intAttr0[d] + node.intAttr1[d])];
                [strides addObject:@1];
            }
            return [graph sliceTensor:tensors[node.inputs[0]] starts:starts ends:ends strides:strides name:nil];
        }
        case OpKind::Gather:
            return [graph gatherWithUpdatesTensor:tensors[node.inputs[0]]
                                     indicesTensor:tensors[node.inputs[1]]
                                              axis:(NSUInteger)node.axis
                                   batchDimensions:0
                                              name:nil];
        case OpKind::QuantizeLinear:
            return [graph quantizeTensor:tensors[node.inputs[0]]
                                   scale:(double)node.floatAttr0
                               zeroPoint:(double)node.floatAttr1
                                dataType:MPSDataTypeInt8
                                    name:nil];
        case OpKind::DequantizeLinear:
            return [graph dequantizeTensor:tensors[node.inputs[0]]
                                     scale:(double)node.floatAttr0
                                 zeroPoint:(double)node.floatAttr1
                                  dataType:MPSDataTypeFloat32
                                      name:nil];
        case OpKind::Conv2d:
        {
            const Conv2dDescriptor &p = node.convParams;
            MPSGraphConvolution2DOpDescriptor *convDesc =
                [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)p.strideX
                                                                  strideInY:(NSUInteger)p.strideY
                                                            dilationRateInX:(NSUInteger)p.dilationX
                                                            dilationRateInY:(NSUInteger)p.dilationY
                                                                     groups:(NSUInteger)p.groups
                                                                paddingLeft:(NSUInteger)p.paddingLeft
                                                               paddingRight:(NSUInteger)p.paddingRight
                                                                 paddingTop:(NSUInteger)p.paddingTop
                                                              paddingBottom:(NSUInteger)p.paddingBottom
                                                               paddingStyle:MPSGraphPaddingStyleExplicit
                                                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
            return [graph convolution2DWithSourceTensor:tensors[node.inputs[0]]
                                           weightsTensor:tensors[node.inputs[1]]
                                              descriptor:convDesc
                                                    name:nil];
        }
        case OpKind::MaxPool2d:
        case OpKind::AvgPool2d:
        {
            const Pool2dDescriptor &p = node.poolParams;
            MPSGraphPooling2DOpDescriptor *poolDesc =
                [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)p.kernelWidth
                                                             kernelHeight:(NSUInteger)p.kernelHeight
                                                                strideInX:(NSUInteger)p.strideX
                                                                strideInY:(NSUInteger)p.strideY
                                                          dilationRateInX:1
                                                          dilationRateInY:1
                                                              paddingLeft:(NSUInteger)p.paddingLeft
                                                             paddingRight:(NSUInteger)p.paddingRight
                                                               paddingTop:(NSUInteger)p.paddingTop
                                                            paddingBottom:(NSUInteger)p.paddingBottom
                                                             paddingStyle:MPSGraphPaddingStyleExplicit
                                                               dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
            if (node.kind == OpKind::MaxPool2d)
                return [graph maxPooling2DWithSourceTensor:tensors[node.inputs[0]] descriptor:poolDesc name:nil];
            else
                return [graph avgPooling2DWithSourceTensor:tensors[node.inputs[0]] descriptor:poolDesc name:nil];
        }
        case OpKind::Resize:
        {
            const ResizeDescriptor &p = node.resizeParams;
            if (p.mode == ResizeMode::Nearest && p.nearestRoundsDown)
            {
                int32_t sizeVals[2] = {(int32_t)p.outputHeight, (int32_t)p.outputWidth};
                NSData *sizeData = [NSData dataWithBytes:sizeVals length:sizeof(sizeVals)];
                MPSGraphTensor *sizeTensor = [graph constantWithData:sizeData
                                                                 shape:shapeFor({2})
                                                              dataType:MPSDataTypeInt32];
                return [graph resizeNearestWithTensor:tensors[node.inputs[0]]
                                             sizeTensor:sizeTensor
                                    nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
                                           centerResult:p.centerResult
                                           alignCorners:p.alignCorners
                                                 layout:MPSGraphTensorNamedDataLayoutNCHW
                                                   name:nil];
            }
            else
            {
                MPSGraphResizeMode mode = p.mode == ResizeMode::Nearest ? MPSGraphResizeNearest : MPSGraphResizeBilinear;
                MPSShape *size = shapeFor({p.outputHeight, p.outputWidth});
                return [graph resizeTensor:tensors[node.inputs[0]]
                                      size:size
                                      mode:mode
                              centerResult:p.centerResult
                              alignCorners:p.alignCorners
                                    layout:MPSGraphTensorNamedDataLayoutNCHW
                                      name:nil];
            }
        }
        case OpKind::GgmlQuantizedMatmul:
            return nil;
        case OpKind::GqaMatMul:
            // Only implemented on the Cpu and GpuGeneric backends (see
            // GraphBuilder::gqaMatMul()'s doc comment) — callers must check
            // Context::deviceType() before building a graph that uses it.
            throw std::runtime_error("campello_nn: MPS backend does not implement GqaMatMul (Cpu/GpuGeneric only)");
        }
        return nil;
    }
}

struct MpsTensor
{
    id<MTLBuffer> buffer;
    TensorDescriptor desc;
};

struct MpsGraphSegment
{
    MPSGraph *graph;
    std::vector<std::pair<size_t, std::string>> inputs;  // (sourceNodeIdx, placeholderName)
    std::vector<std::pair<size_t, std::string>> outputs; // (sourceNodeIdx, targetTensorName)
    std::unordered_map<std::string, MPSGraphTensor *> inputTensors;
    std::unordered_map<std::string, MPSGraphTensor *> outputTensors;
};

struct MpsCustomSegment
{
    // One or more consecutive GgmlQuantizedMatmul nodes with no plain (MPSGraph)
    // op between them in the IR -- e.g. q/k/v or gate/up projections, which are
    // architecturally independent (all read the same upstream activation, write
    // to different outputs) and so can share one command buffer/encoder instead
    // of each paying its own commit+waitUntilCompleted round trip. See
    // compileGraph()'s segment-building loop for how these get grouped, and
    // dispatch()'s custom-segment branch for how they're encoded together.
    std::vector<size_t> nodeIndices;
};

struct MpsSegment
{
    bool isCustom;
    MpsGraphSegment graphSegment;
    MpsCustomSegment customSegment;
};

// A buffer plus a byte offset into it. Most resolved buffers (inputs, computed/
// cross-segment values, individually-allocated constants) have offset 0 -- this
// only becomes nonzero for a constant sharing space in the single consolidated
// buffer built by uploadConstantBuffers() below.
struct MpsBufferRef
{
    id<MTLBuffer> buffer;
    NSUInteger offset = 0;
};

struct MpsCompiledGraph
{
    GraphIR ir;
    std::vector<MpsSegment> segments;

    // Device-resident storage for every Constant node's bytes, uploaded once in
    // compileGraph() and reused for the compiled graph's whole lifetime -- see
    // uploadConstantBuffers() below for why this replaced a per-dispatch()-call
    // allocate+memcpy of the same bytes (e.g. once per generated token for a
    // quantized weight feeding a GgmlQuantizedMatmul segment). Most entries here
    // share a handful of large consolidated buffers (sliced by offset) rather
    // than each getting its own MTLBuffer -- see uploadConstantBuffers()'s doc
    // comment for why a small set of constants (ones an MPSGraph operation might
    // read directly) still get an individual, unshared buffer.
    std::unordered_map<size_t, MpsBufferRef> constantBuffers;

    // Reusable pool of transient (cross-segment/intermediate-activation) buffers,
    // keyed by byte size, that dispatch() draws from instead of always calling
    // newBufferWithLength: -- see allocateBuffer() and destroyFence() below.
    // Every dispatch() call against this graph touches the same set of
    // intermediate-activation shapes (only the values differ token-to-token, not
    // the shapes), so after a brief warm-up this pool covers the whole graph and
    // stops growing, instead of Metal's own driver-level memory pool
    // accumulating a distinct new region for every fresh allocation once per
    // token indefinitely.
    std::unordered_map<std::size_t, std::vector<id<MTLBuffer>>> bufferPool;
};

struct MpsFence
{
    bool signaled = true;
    std::vector<id<MTLBuffer>> ownedBuffers;

    // Graph whose bufferPool ownedBuffers should be returned to once this fence
    // is destroyed (see destroyFence()); nullptr if this fence never allocated
    // any transient buffers (e.g. every value it touched was a caller-provided
    // input/output or a constant). By the time a fence exists to return, the
    // Metal work using ownedBuffers has already completed synchronously --
    // dispatch() only returns after runWithMTLCommandQueue:/waitUntilCompleted
    // has finished -- so it's always safe to recycle them here.
    MpsCompiledGraph *owningGraph = nullptr;
};

struct MpsBackend::Impl
{
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> quantLibrary;
    id<MTLComputePipelineState> quantPipeline;
};

MpsBackend::MpsBackend()
{
    impl = new Impl();
    impl->device = MTLCreateSystemDefaultDevice();
    if (!impl->device)
        throw std::runtime_error("campello_nn: MTLCreateSystemDefaultDevice() returned nil");
    impl->queue = [impl->device newCommandQueue];

    NSError *error = nil;
    impl->quantLibrary = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:ggmlQuantizedMatmulMetalSource()]
                                                    options:nil
                                                      error:&error];
    if (!impl->quantLibrary)
    {
        NSString *msg = error ? [error localizedDescription] : @"unknown";
        throw std::runtime_error("campello_nn: failed to compile GGML Metal library: " +
                                 std::string([msg UTF8String]));
    }
    id<MTLFunction> fn = [impl->quantLibrary newFunctionWithName:@"ggmlQuantizedMatmul"];
    if (!fn)
        throw std::runtime_error("campello_nn: Metal kernel 'ggmlQuantizedMatmul' not found");
    impl->quantPipeline = [impl->device newComputePipelineStateWithFunction:fn error:&error];
    if (!impl->quantPipeline)
    {
        NSString *msg = error ? [error localizedDescription] : @"unknown";
        throw std::runtime_error("campello_nn: failed to create GGML Metal pipeline: " +
                                 std::string([msg UTF8String]));
    }
}

MpsBackend::~MpsBackend()
{
    delete impl;
}

void *MpsBackend::createTensor(const TensorDescriptor &desc)
{
    size_t byteSize = elementByteSize(desc.dataType) * numElements(desc.shape);
    id<MTLBuffer> buffer = [impl->device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
    auto t = new MpsTensor{buffer, desc};
    return t;
}

void MpsBackend::destroyTensor(void *native)
{
    delete (MpsTensor *)native;
}

void MpsBackend::writeTensor(void *native, const void *data, size_t size)
{
    auto t = (MpsTensor *)native;
    if (size > t->buffer.length)
        throw std::runtime_error("campello_nn: write() exceeds tensor capacity");
    memcpy(t->buffer.contents, data, size);
}

void MpsBackend::readTensor(void *native, void *data, size_t size)
{
    auto t = (MpsTensor *)native;
    if (size > t->buffer.length)
        throw std::runtime_error("campello_nn: read() exceeds tensor capacity");
    memcpy(data, t->buffer.contents, size);
}

namespace
{
    // Uploads every Constant node's bytes to a persistent MTLBuffer, once, and then
    // drops the node's own host-side copy -- called at the end of compileGraph(),
    // after any MPSGraph segment has already embedded whatever it needs from
    // node.constantBytes via constantWithData: (which makes its own internal copy,
    // unrelated to this one). Before this, any Constant referenced by a custom
    // GgmlQuantizedMatmul segment (every quantized weight) or by a cross-segment
    // MPSGraph placeholder got re-allocated and re-memcpy'd from node.constantBytes
    // on *every* dispatch() call (see the old getInputBuffer() Constant branch) --
    // for an 8B-parameter model that's ~4.6GB re-copied per generated token, on top
    // of node.constantBytes itself being a second permanent copy of every weight
    // byte for the compiled graph's whole lifetime.
    //
    // `consumedByRealOp` (nullptr for the no-custom-node graph, where nothing ever
    // reads constantBuffers at all -- see compileGraph()) marks constants an
    // MPSGraph *operation* might read directly (norm gammas etc. -- see the
    // comment where it's computed in compileGraph()). Everything else is *only*
    // ever consumed via getInputBuffer()'s raw-MTLBuffer path (i.e. by a
    // GgmlQuantizedMatmul custom kernel), so those are packed into as few large
    // buffers as possible (sliced by offset) instead of one MTLBuffer allocation
    // each -- 292 separate allocations for an 8B-parameter model, ~9,211
    // driver-level "graphics" memory regions measured for llama3.1_8b before this
    // change, comparable to llama.cpp/Ollama's ~47 for the same model. The
    // remaining, possibly-MPSGraph-fed constants keep an individual buffer each:
    // MPSGraphTensorData's initializer has no byte-offset parameter, so a
    // shared/sliced buffer can't be handed to it.
    //
    // Packing into ONE single buffer (rather than a handful) was tried and
    // crashed: MTLDevice has a maxBufferLength ceiling per single allocation, and
    // an ~8B-parameter model's ~4.6GB of quantized weights exceeded it on the
    // hardware this was tested on -- newBufferWithLength: silently returned nil,
    // and the following memcpy into nil.contents (0x0) segfaulted. Chunking by
    // that device-reported ceiling (with headroom) keeps the same "few large
    // buffers" win without assuming unlimited single-buffer size.
    // Operates on `ir` directly (the compileGraph() parameter, not
    // compiled->ir) so it can run *before* compiled->ir is ever populated --
    // see compileGraph()'s comment on why populating compiled->ir early (then
    // clearing it here) would leave `ir` and compiled->ir simultaneously
    // holding full uncleared copies of every weight's bytes for this whole
    // function's duration.
    void uploadConstantBuffers(id<MTLDevice> device, GraphIR &ir, MpsCompiledGraph *compiled,
                               const std::vector<bool> *consumedByRealOp)
    {
        constexpr std::size_t kAlignment = 256; // conservative, cache-line-friendly offset alignment
        auto alignUp = [](std::size_t v, std::size_t a) { return (v + a - 1) / a * a; };

        NSUInteger maxBufferLength = device.maxBufferLength;
        if (maxBufferLength == 0)
            maxBufferLength = 256ull * 1024 * 1024; // conservative fallback if unreported
        std::size_t chunkCap = static_cast<std::size_t>(maxBufferLength) * 3 / 4; // headroom below the hard ceiling

        struct PendingConstant
        {
            size_t nodeIdx;
            std::size_t size;
        };
        std::vector<PendingConstant> pending;
        for (size_t i = 0; i < ir.nodes.size(); ++i)
        {
            const Node &node = ir.nodes[i];
            if (node.kind != OpKind::Constant || node.constantBytes.empty())
                continue;
            bool mustBeIndividual = consumedByRealOp == nullptr || (*consumedByRealOp)[i];
            if (mustBeIndividual)
                continue;
            pending.push_back({i, node.constantBytes.size()});
        }

        // Greedily pack constants into chunks no larger than chunkCap each. A
        // single constant bigger than chunkCap (not expected for any real weight
        // tensor, but not assumed away) gets a solo buffer sized to itself.
        size_t idx = 0;
        while (idx < pending.size())
        {
            std::vector<std::pair<size_t, std::size_t>> chunkOffsets; // (nodeIdx, offset)
            std::size_t chunkSize = 0;
            while (idx < pending.size())
            {
                std::size_t candidateEnd = alignUp(chunkSize + pending[idx].size, kAlignment);
                if (!chunkOffsets.empty() && candidateEnd > chunkCap)
                    break;
                chunkOffsets.emplace_back(pending[idx].nodeIdx, chunkSize);
                chunkSize = candidateEnd;
                ++idx;
            }

            id<MTLBuffer> buffer = [device newBufferWithLength:chunkSize options:MTLResourceStorageModeShared];
            if (buffer == nil)
            {
                throw std::runtime_error("campello_nn: MPS backend failed to allocate a " +
                                         std::to_string(chunkSize) + "-byte constant buffer chunk");
            }
            for (auto &[nodeIdx, offset] : chunkOffsets)
            {
                Node &node = ir.nodes[nodeIdx];
                memcpy(static_cast<uint8_t *>(buffer.contents) + offset, node.constantBytes.data(),
                       node.constantBytes.size());
                compiled->constantBuffers[nodeIdx] = {buffer, static_cast<NSUInteger>(offset)};
                node.constantBytes.clear();
                node.constantBytes.shrink_to_fit();
            }
        }

        for (size_t i = 0; i < ir.nodes.size(); ++i)
        {
            Node &node = ir.nodes[i];
            if (node.kind != OpKind::Constant || node.constantBytes.empty())
                continue;
            if (compiled->constantBuffers.count(i))
                continue; // already placed in a consolidated chunk above
            id<MTLBuffer> buffer = [device newBufferWithLength:node.constantBytes.size()
                                                        options:MTLResourceStorageModeShared];
            memcpy(buffer.contents, node.constantBytes.data(), node.constantBytes.size());
            compiled->constantBuffers[i] = {buffer, 0};
            node.constantBytes.clear();
            node.constantBytes.shrink_to_fit();
        }
    }
} // namespace

void *MpsBackend::compileGraph(GraphIR ir)
{
    auto compiled = new MpsCompiledGraph{};
    // compiled->ir is populated by moving `ir` at each return point below,
    // not copied up front -- see uploadConstantBuffers()'s comment for why an
    // early copy would leave `ir` and compiled->ir simultaneously holding
    // full uncleared copies of every weight's bytes for no reason.

    bool hasCustom = false;
    for (const Node &node : ir.nodes)
        if (node.kind == OpKind::GgmlQuantizedMatmul)
            hasCustom = true;

    if (!hasCustom)
    {
        MpsSegment segment;
        segment.isCustom = false;
        segment.graphSegment.graph = [[MPSGraph alloc] init];
        std::vector<MPSGraphTensor *> tensors(ir.nodes.size());
        for (size_t i = 0; i < ir.nodes.size(); i++)
        {
            const Node &node = ir.nodes[i];
            tensors[i] = buildMpsNode(segment.graphSegment.graph, node, tensors);
            if (node.kind == OpKind::Input)
            {
                std::string name = node.name;
                segment.graphSegment.inputs.push_back({i, name});
                segment.graphSegment.inputTensors[name] = tensors[i];
            }
        }
        for (auto &[name, nodeIdx] : ir.outputs)
            segment.graphSegment.outputs.push_back({nodeIdx, name});
        for (auto &[nodeIdx, name] : segment.graphSegment.outputs)
            segment.graphSegment.outputTensors[name] = tensors[nodeIdx];
        compiled->segments.push_back(std::move(segment));
        // No custom (GgmlQuantizedMatmul) segments and no cross-segment mechanism
        // (there's only the one segment) -- every Constant was already embedded
        // inline via constantWithData: above (which makes its own internal copy),
        // and nothing in dispatch() would ever call getInputBuffer() for a
        // Constant in this path. Uploading persistent buffers nobody reads would
        // just be wasted allocation -- and node.constantBytes itself is now a
        // second, permanently-retained copy of every weight for no purpose, so
        // clear it before moving `ir` into compiled->ir below.
        for (Node &node : ir.nodes)
        {
            if (node.kind == OpKind::Constant)
            {
                node.constantBytes.clear();
                node.constantBytes.shrink_to_fit();
            }
        }
        compiled->ir = std::move(ir);
        return compiled;
    }

    std::vector<bool> isCustomNode(ir.nodes.size(), false);
    for (size_t i = 0; i < ir.nodes.size(); i++)
        isCustomNode[i] = ir.nodes[i].kind == OpKind::GgmlQuantizedMatmul;

    // Constants a real MPSGraph operation never reads (e.g. a GGML-quantized
    // weight only consumed by the custom GgmlQuantizedMatmul kernel via raw
    // bytes) don't need a constantWithData: embedding at all -- getInputBuffer()
    // resolves any Constant node straight from constantBuffers (see
    // uploadConstantBuffers()) regardless of which segment declares it or who
    // reads it, so embedding one nobody's MPSGraph ops touch would just be a
    // second, permanently-retained copy of those bytes for no purpose. A
    // Constant only needs embedding if some non-Input/non-Constant/non-custom
    // node (i.e. something that goes through buildMpsNode(), which dereferences
    // tensors[inputIdx] directly) actually consumes it.
    std::vector<bool> consumedByRealOp(ir.nodes.size(), false);
    for (size_t j = 0; j < ir.nodes.size(); ++j)
    {
        OpKind jk = ir.nodes[j].kind;
        if (jk == OpKind::Input || jk == OpKind::Constant || jk == OpKind::GgmlQuantizedMatmul)
            continue;
        for (size_t inputIdx : ir.nodes[j].inputs)
            consumedByRealOp[inputIdx] = true;
    }

    std::vector<bool> isOutputNode(ir.nodes.size(), false);
    for (auto &[name, nodeIdx] : ir.outputs)
        isOutputNode[nodeIdx] = true;
    for (size_t i = 0; i < ir.nodes.size(); i++)
    {
        const Node &node = ir.nodes[i];
        for (size_t inputIdx : node.inputs)
        {
            if (ir.nodes[inputIdx].kind == OpKind::Constant)
            {
                // Never needs cross-segment MPSGraph readback -- see above.
                continue;
            }
            bool producerCustom = isCustomNode[inputIdx];
            bool consumerCustom = isCustomNode[i];
            if (producerCustom || consumerCustom)
            {
                isOutputNode[inputIdx] = true;
            }
            else
            {
                bool split = false;
                for (size_t k = inputIdx + 1; k < i; ++k)
                {
                    if (isCustomNode[k])
                    {
                        split = true;
                        break;
                    }
                }
                if (split)
                    isOutputNode[inputIdx] = true;
            }
        }
    }

    size_t i = 0;
    while (i < ir.nodes.size())
    {
        if (isCustomNode[i])
        {
            size_t runStart = i;
            while (i < ir.nodes.size() && isCustomNode[i])
                ++i;

            MpsSegment segment;
            segment.isCustom = true;
            segment.customSegment.nodeIndices.reserve(i - runStart);
            for (size_t k = runStart; k < i; ++k)
                segment.customSegment.nodeIndices.push_back(k);
            compiled->segments.push_back(std::move(segment));
        }
        else
        {
            size_t segStart = i;
            while (i < ir.nodes.size() && !isCustomNode[i])
                ++i;
            size_t segEnd = i;

            MpsSegment segment;
            segment.isCustom = false;
            segment.graphSegment.graph = [[MPSGraph alloc] init];
            std::vector<MPSGraphTensor *> tensors(ir.nodes.size(), nullptr);

            // First pass: add placeholders for any input that lives outside this
            // segment (e.g. the output of a preceding custom GgmlQuantizedMatmul
            // segment, or a Constant produced in an earlier segment). Without this,
            // buildMpsNode would dereference a nullptr for tensors[inputIdx].
            for (size_t k = segStart; k < segEnd; ++k)
            {
                const Node &node = ir.nodes[k];
                for (size_t inputIdx : node.inputs)
                {
                    if (inputIdx >= segStart && inputIdx < segEnd)
                    {
                        // Same-segment input: handled in the second pass below.
                        continue;
                    }
                    if (tensors[inputIdx] != nullptr)
                    {
                        // Already created a placeholder for this external input.
                        continue;
                    }
                    const Node &inputNode = ir.nodes[inputIdx];
                    std::string inputName = "node_" + std::to_string(inputIdx);
                    tensors[inputIdx] = [segment.graphSegment.graph placeholderWithShape:shapeFor(inputNode.shape)
                                                                                dataType:mpsDataType(inputNode.dataType)
                                                                                    name:nameFor(inputName)];
                    segment.graphSegment.inputs.push_back({inputIdx, inputName});
                    segment.graphSegment.inputTensors[inputName] = tensors[inputIdx];
                }
            }

            // Second pass: build Input/Constant/operation nodes inside the segment.
            for (size_t k = segStart; k < segEnd; ++k)
            {
                const Node &node = ir.nodes[k];
                if (node.kind == OpKind::Input)
                {
                    tensors[k] = [segment.graphSegment.graph placeholderWithShape:shapeFor(node.shape)
                                                                          dataType:mpsDataType(node.dataType)
                                                                              name:nameFor(node.name)];
                    std::string name = node.name;
                    segment.graphSegment.inputs.push_back({k, name});
                    segment.graphSegment.inputTensors[name] = tensors[k];
                }
                else if (node.kind == OpKind::Constant)
                {
                    // Skip the embedding entirely when nothing in this (or any)
                    // segment's MPSGraph actually reads it as an operand -- see the
                    // consumedByRealOp comment above. tensors[k] stays nullptr,
                    // which is safe: isOutputNode[k] is never true for a Constant
                    // (see above), so it's never added to segment.graphSegment.outputs
                    // or otherwise dereferenced from here on.
                    if (consumedByRealOp[k])
                    {
                        NSData *data = [NSData dataWithBytes:node.constantBytes.data() length:node.constantBytes.size()];
                        tensors[k] = [segment.graphSegment.graph constantWithData:data
                                                                            shape:shapeFor(node.shape)
                                                                         dataType:mpsDataType(node.dataType)];
                    }
                }
                else
                {
                    tensors[k] = buildMpsNode(segment.graphSegment.graph, node, tensors);
                }
            }

            for (size_t k = segStart; k < segEnd; ++k)
            {
                if (isOutputNode[k])
                    segment.graphSegment.outputs.push_back({k, "node_" + std::to_string(k)});
            }
            for (auto &[nodeIdx, name] : segment.graphSegment.outputs)
                segment.graphSegment.outputTensors[name] = tensors[nodeIdx];

            compiled->segments.push_back(std::move(segment));
        }
    }

    uploadConstantBuffers(impl->device, ir, compiled, &consumedByRealOp);
    compiled->ir = std::move(ir);
    return compiled;
}

void MpsBackend::destroyGraph(void *native)
{
    delete (MpsCompiledGraph *)native;
}

void *MpsBackend::dispatch(
    void *compiledGraph,
    const std::unordered_map<std::string, void *> &inputs,
    const std::unordered_map<std::string, void *> &outputs)
{
    auto g = (MpsCompiledGraph *)compiledGraph;
    const GraphIR &ir = g->ir;

    std::unordered_map<size_t, MpsBufferRef> nodeBuffers;
    std::unordered_set<size_t> outputNodeIndices;
    for (auto &[name, nodeIdx] : ir.outputs)
    {
        (void)name;
        outputNodeIndices.insert(nodeIdx);
    }

    auto fence = new MpsFence{true, {}, g};

    auto allocateBuffer = [&](const Node &node) -> id<MTLBuffer>
    {
        size_t byteSize = elementByteSize(node.dataType) * numElements(node.shape);
        std::vector<id<MTLBuffer>> &pooled = g->bufferPool[byteSize];
        id<MTLBuffer> buffer;
        if (!pooled.empty())
        {
            buffer = pooled.back();
            pooled.pop_back();
        }
        else
        {
            buffer = [impl->device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
        }
        fence->ownedBuffers.push_back(buffer);
        return buffer;
    };

    auto getInputBuffer = [&](size_t nodeIdx) -> MpsBufferRef
    {
        auto it = nodeBuffers.find(nodeIdx);
        if (it != nodeBuffers.end())
            return it->second;

        const Node &node = ir.nodes[nodeIdx];
        if (node.kind == OpKind::Input)
        {
            auto inIt = inputs.find(node.name);
            if (inIt == inputs.end())
                throw std::runtime_error("campello_nn: MPS backend missing input tensor '" + node.name + "'");
            auto t = (MpsTensor *)inIt->second;
            MpsBufferRef ref{t->buffer, 0};
            nodeBuffers[nodeIdx] = ref;
            return ref;
        }
        else if (node.kind == OpKind::Constant)
        {
            // Persistent buffer uploaded once in compileGraph(); node.constantBytes
            // was cleared right after, so it can't be re-copied from here anymore
            // (and doesn't need to be -- this buffer is reused across every dispatch).
            // May be a slice of a larger consolidated buffer -- see
            // uploadConstantBuffers()'s doc comment.
            auto bufIt = g->constantBuffers.find(nodeIdx);
            if (bufIt == g->constantBuffers.end())
                throw std::runtime_error("campello_nn: MPS backend missing uploaded buffer for constant node " +
                                         std::to_string(nodeIdx));
            nodeBuffers[nodeIdx] = bufIt->second;
            return bufIt->second;
        }
        else
        {
            throw std::runtime_error("campello_nn: MPS backend requested buffer for non-input/constant node " +
                                     std::to_string(nodeIdx));
        }
    };

    auto getOrCreateOutputBuffer = [&](size_t nodeIdx) -> id<MTLBuffer>
    {
        auto it = nodeBuffers.find(nodeIdx);
        if (it != nodeBuffers.end())
            return it->second.buffer;

        const Node &node = ir.nodes[nodeIdx];
        if (outputNodeIndices.count(nodeIdx))
        {
            // Find the output name for this node index.
            std::string outName;
            for (auto &[name, idx] : ir.outputs)
            {
                if (idx == nodeIdx)
                {
                    outName = name;
                    break;
                }
            }
            auto outIt = outputs.find(outName);
            if (outIt != outputs.end())
            {
                auto t = (MpsTensor *)outIt->second;
                nodeBuffers[nodeIdx] = MpsBufferRef{t->buffer, 0};
                return t->buffer;
            }
        }
        id<MTLBuffer> buffer = allocateBuffer(node);
        nodeBuffers[nodeIdx] = MpsBufferRef{buffer, 0};
        return buffer;
    };

    for (const MpsSegment &segment : g->segments)
    {
        if (segment.isCustom)
        {
            // All nodes in this segment are consecutive-in-IR GgmlQuantizedMatmul
            // ops with no plain op between them (see compileGraph()) -- e.g. q/k/v
            // or gate/up, which are architecturally independent (same upstream
            // input, different outputs). Encoding them into one command
            // buffer/encoder with a single commit+waitUntilCompleted instead of one
            // round trip each is what turns N synchronous CPU-GPU stalls per layer
            // into 1.
            id<MTLCommandBuffer> cmdBuffer = [impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
            [encoder setComputePipelineState:impl->quantPipeline];

            for (size_t nodeIdx : segment.customSegment.nodeIndices)
            {
                const Node &node = ir.nodes[nodeIdx];
                if (node.kind != OpKind::GgmlQuantizedMatmul)
                    throw std::runtime_error("campello_nn: MPS backend custom segment is not GgmlQuantizedMatmul");
                if (node.intAttr0.size() != 2)
                    throw std::runtime_error("campello_nn: MPS backend GgmlQuantizedMatmul weightShape missing");

                int64_t inFeatures = node.intAttr0[0];
                int64_t outFeatures = node.intAttr0[1];
                int64_t M = node.shape[node.shape.size() - 2];
                int64_t batchCount = 1;
                for (size_t d = 0; d + 2 < node.shape.size(); ++d)
                    batchCount *= node.shape[d];
                uint32_t rows = static_cast<uint32_t>(batchCount * M);
                uint32_t K = static_cast<uint32_t>(inFeatures);
                uint32_t N = static_cast<uint32_t>(outFeatures);
                uint32_t ggmlType = static_cast<uint32_t>(node.axis);
                requireSupportedGgmlType(ggmlType);

                MpsBufferRef activationBuffer = getInputBuffer(node.inputs[0]);
                MpsBufferRef weightBuffer = getInputBuffer(node.inputs[1]);
                id<MTLBuffer> outputBuffer = getOrCreateOutputBuffer(nodeIdx);

                [encoder setBuffer:activationBuffer.buffer offset:activationBuffer.offset atIndex:0];
                [encoder setBuffer:weightBuffer.buffer offset:weightBuffer.offset atIndex:1];
                [encoder setBuffer:outputBuffer offset:0 atIndex:2];
                uint32_t dims[4] = {rows, K, N, ggmlType};
                [encoder setBytes:dims length:sizeof(dims) atIndex:3];

                MTLSize gridSize = MTLSizeMake(rows, N, 1);
                NSUInteger w = impl->quantPipeline.maxTotalThreadsPerThreadgroup;
                if (w == 0)
                    w = 1;
                NSUInteger tgx = std::min((NSUInteger)16, w);
                NSUInteger tgy = std::min((NSUInteger)16, w / tgx);
                if (tgy == 0)
                    tgy = 1;
                MTLSize threadgroupSize = MTLSizeMake(tgx, tgy, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            }

            [encoder endEncoding];
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }
        else
        {
            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = [NSMutableDictionary dictionary];
            for (auto &[nodeIdx, name] : segment.graphSegment.inputs)
            {
                id<MTLBuffer> buffer;
                const Node &node = ir.nodes[nodeIdx];
                if (node.kind == OpKind::Input || node.kind == OpKind::Constant)
                {
                    MpsBufferRef ref = getInputBuffer(nodeIdx);
                    if (ref.offset != 0)
                    {
                        // Can't happen in practice -- only constants excluded from
                        // consumedByRealOp share the consolidated/offset buffer (see
                        // uploadConstantBuffers()), and those are never fed to an
                        // MPSGraph operation. Guard against it anyway rather than
                        // silently binding the wrong bytes if that invariant is ever
                        // violated: MPSGraphTensorData has no byte-offset initializer.
                        throw std::runtime_error("campello_nn: MPS backend cannot feed an offset constant "
                                                 "buffer directly into MPSGraph (node " +
                                                 std::to_string(nodeIdx) + ")");
                    }
                    buffer = ref.buffer;
                }
                else
                {
                    // External (cross-segment) input: the buffer was produced by a
                    // previous segment and should already be in nodeBuffers.
                    auto it = nodeBuffers.find(nodeIdx);
                    if (it == nodeBuffers.end())
                    {
                        throw std::runtime_error("campello_nn: MPS backend missing cross-segment buffer for node " +
                                                  std::to_string(nodeIdx));
                    }
                    buffer = it->second.buffer;
                }
                MPSGraphTensorData *td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                                  shape:shapeFor(node.shape)
                                                                               dataType:mpsDataType(node.dataType)];
                feeds[segment.graphSegment.inputTensors.at(name)] = td;
            }

            NSMutableArray<MPSGraphTensor *> *targetTensors = [NSMutableArray array];
            std::vector<size_t> outputNodeIdxs;
            std::vector<id<MTLBuffer>> outputBuffers;
            for (auto &[nodeIdx, name] : segment.graphSegment.outputs)
            {
                id<MTLBuffer> buffer = getOrCreateOutputBuffer(nodeIdx);
                outputBuffers.push_back(buffer);
                outputNodeIdxs.push_back(nodeIdx);
                [targetTensors addObject:segment.graphSegment.outputTensors.at(name)];
            }

            MPSGraphTensorDataDictionary *results = [segment.graphSegment.graph runWithMTLCommandQueue:impl->queue
                                                                                                  feeds:feeds
                                                                                          targetTensors:targetTensors
                                                                                       targetOperations:nil];

            for (size_t j = 0; j < outputNodeIdxs.size(); ++j)
            {
                MPSGraphTensor *targetTensor = segment.graphSegment.outputTensors.at(segment.graphSegment.outputs[j].second);
                MPSGraphTensorData *resultData = results[targetTensor];
                [[resultData mpsndarray] readBytes:outputBuffers[j].contents strideBytes:nil];
            }
        }
    }

    return fence;
}

bool MpsBackend::waitFence(void *fenceNative, uint64_t)
{
    return ((MpsFence *)fenceNative)->signaled;
}

bool MpsBackend::isFenceSignaled(void *fenceNative)
{
    return ((MpsFence *)fenceNative)->signaled;
}

void MpsBackend::destroyFence(void *fenceNative)
{
    auto *fence = (MpsFence *)fenceNative;
    if (fence->owningGraph != nullptr)
    {
        for (id<MTLBuffer> buffer : fence->ownedBuffers)
        {
            fence->owningGraph->bufferPool[buffer.length].push_back(buffer);
        }
    }
    delete fence;
}

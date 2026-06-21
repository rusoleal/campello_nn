#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdexcept>
#include <unordered_map>
#include "mps_backend.hpp"
#include "../pi/ir.hpp"

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

    // Reshapes a 1D [lastDim] tensor (LayerNorm's scale/bias) to rank `rank` with
    // leading 1s, so MPSGraph broadcasts it against the last axis as intended.
    MPSGraphTensor *reshapeTrailing(MPSGraph *graph, MPSGraphTensor *t, size_t rank, int64_t lastDim)
    {
        std::vector<int64_t> shape(rank, 1);
        shape[rank - 1] = lastDim;
        return [graph reshapeTensor:t withShape:shapeFor(shape) name:nil];
    }

    // Reshapes Gemm's `c` operand (shape [M,N], [N], or [1]) to a canonical rank-2
    // shape so it broadcasts correctly against the [M,N] matmul result.
    MPSGraphTensor *reshapeGemmC(MPSGraph *graph, MPSGraphTensor *c, int64_t cElems, int64_t N)
    {
        std::vector<int64_t> shape = cElems == 1 ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1, N};
        if (cElems != 1 && cElems != N)
            return c; // already [M, N]
        return [graph reshapeTensor:c withShape:shapeFor(shape) name:nil];
    }

    // Reshapes a 1D [C] tensor (BatchNorm/InstanceNorm's per-channel mean/variance/
    // scale/bias) to [1, C, 1, 1] so MPSGraph broadcasts it against NCHW as intended.
    MPSGraphTensor *reshapeChannel(MPSGraph *graph, MPSGraphTensor *t, int64_t C)
    {
        return [graph reshapeTensor:t withShape:shapeFor({1, C, 1, 1}) name:nil];
    }
}

struct MpsTensor
{
    id<MTLBuffer> buffer;
    TensorDescriptor desc;
};

struct MpsCompiledGraph
{
    MPSGraph *graph;
    std::unordered_map<std::string, MPSGraphTensor *> inputTensors;
    std::unordered_map<std::string, MPSGraphTensor *> outputTensors;
};

struct MpsFence
{
    bool signaled = true;
};

struct MpsBackend::Impl
{
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

MpsBackend::MpsBackend()
{
    impl = new Impl();
    impl->device = MTLCreateSystemDefaultDevice();
    if (!impl->device)
        throw std::runtime_error("campello_nn: MTLCreateSystemDefaultDevice() returned nil");
    impl->queue = [impl->device newCommandQueue];
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

void *MpsBackend::compileGraph(const GraphIR &ir)
{
    MPSGraph *graph = [[MPSGraph alloc] init];
    std::vector<MPSGraphTensor *> tensors(ir.nodes.size());
    auto compiled = new MpsCompiledGraph{graph, {}, {}};

    for (size_t i = 0; i < ir.nodes.size(); i++)
    {
        const Node &node = ir.nodes[i];
        switch (node.kind)
        {
        case OpKind::Input:
        {
            MPSGraphTensor *t = [graph placeholderWithShape:shapeFor(node.shape)
                                                    dataType:mpsDataType(node.dataType)
                                                        name:nameFor(node.name)];
            tensors[i] = t;
            compiled->inputTensors[node.name] = t;
            break;
        }
        case OpKind::Constant:
        {
            NSData *data = [NSData dataWithBytes:node.constantBytes.data() length:node.constantBytes.size()];
            tensors[i] = [graph constantWithData:data shape:shapeFor(node.shape) dataType:mpsDataType(node.dataType)];
            break;
        }
        case OpKind::Add:
            tensors[i] = [graph additionWithPrimaryTensor:tensors[node.inputs[0]]
                                            secondaryTensor:tensors[node.inputs[1]]
                                                       name:nil];
            break;
        case OpKind::Mul:
            tensors[i] = [graph multiplicationWithPrimaryTensor:tensors[node.inputs[0]]
                                                  secondaryTensor:tensors[node.inputs[1]]
                                                             name:nil];
            break;
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
            tensors[i] = [graph multiplicationWithPrimaryTensor:halfX secondaryTensor:onePlusErf name:nil];
            break;
        }
        case OpKind::Relu:
            tensors[i] = [graph reLUWithTensor:tensors[node.inputs[0]] name:nil];
            break;
        case OpKind::Sigmoid:
            tensors[i] = [graph sigmoidWithTensor:tensors[node.inputs[0]] name:nil];
            break;
        case OpKind::Softmax:
            tensors[i] = [graph softMaxWithTensor:tensors[node.inputs[0]] axis:(NSInteger)node.axis name:nil];
            break;
        case OpKind::LayerNorm:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            NSInteger lastAxis = (NSInteger)node.shape.size() - 1;
            NSArray<NSNumber *> *axes = @[ @(lastAxis) ];
            MPSGraphTensor *mean = [graph meanOfTensor:x axes:axes name:nil];
            MPSGraphTensor *variance = [graph varianceOfTensor:x meanTensor:mean axes:axes name:nil];
            MPSGraphTensor *gamma = reshapeTrailing(graph, tensors[node.inputs[1]], node.shape.size(), node.shape.back());
            MPSGraphTensor *beta = reshapeTrailing(graph, tensors[node.inputs[2]], node.shape.size(), node.shape.back());
            tensors[i] = [graph normalizationWithTensor:x
                                              meanTensor:mean
                                          varianceTensor:variance
                                             gammaTensor:gamma
                                              betaTensor:beta
                                                 epsilon:node.floatAttr0
                                                    name:nil];
            break;
        }
        case OpKind::BatchNorm:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            int64_t C = node.shape[1];
            MPSGraphTensor *mean = reshapeChannel(graph, tensors[node.inputs[1]], C);
            MPSGraphTensor *variance = reshapeChannel(graph, tensors[node.inputs[2]], C);
            MPSGraphTensor *gamma = reshapeChannel(graph, tensors[node.inputs[3]], C);
            MPSGraphTensor *beta = reshapeChannel(graph, tensors[node.inputs[4]], C);
            tensors[i] = [graph normalizationWithTensor:x
                                              meanTensor:mean
                                          varianceTensor:variance
                                             gammaTensor:gamma
                                              betaTensor:beta
                                                 epsilon:node.floatAttr0
                                                    name:nil];
            break;
        }
        case OpKind::InstanceNorm:
        {
            MPSGraphTensor *x = tensors[node.inputs[0]];
            int64_t C = node.shape[1];
            NSArray<NSNumber *> *axes = @[ @2, @3 ]; // NCHW spatial axes
            MPSGraphTensor *mean = [graph meanOfTensor:x axes:axes name:nil];
            MPSGraphTensor *variance = [graph varianceOfTensor:x meanTensor:mean axes:axes name:nil];
            MPSGraphTensor *gamma = reshapeChannel(graph, tensors[node.inputs[1]], C);
            MPSGraphTensor *beta = reshapeChannel(graph, tensors[node.inputs[2]], C);
            tensors[i] = [graph normalizationWithTensor:x
                                              meanTensor:mean
                                          varianceTensor:variance
                                             gammaTensor:gamma
                                              betaTensor:beta
                                                 epsilon:node.floatAttr0
                                                    name:nil];
            break;
        }
        case OpKind::MatMul:
            tensors[i] = [graph matrixMultiplicationWithPrimaryTensor:tensors[node.inputs[0]]
                                                       secondaryTensor:tensors[node.inputs[1]]
                                                                  name:nil];
            break;
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
            // Reshape c for broadcasting based on its element count, read from the
            // MPSGraphTensor's own shape property (c's IR Node isn't in scope here).
            int64_t cCount = 1;
            for (NSNumber *d in c.shape)
                cCount *= d.longLongValue;
            MPSGraphTensor *cReshaped = reshapeGemmC(graph, c, cCount, N);
            MPSGraphTensor *scaledC = [graph multiplicationWithPrimaryTensor:cReshaped secondaryTensor:beta name:nil];
            tensors[i] = [graph additionWithPrimaryTensor:scaledMm secondaryTensor:scaledC name:nil];
            break;
        }
        case OpKind::Reshape:
            tensors[i] = [graph reshapeTensor:tensors[node.inputs[0]] withShape:shapeFor(node.shape) name:nil];
            break;
        case OpKind::Transpose:
        {
            NSMutableArray<NSNumber *> *perm = [NSMutableArray arrayWithCapacity:node.intAttr0.size()];
            for (auto p : node.intAttr0)
                [perm addObject:@(p)];
            tensors[i] = [graph transposeTensor:tensors[node.inputs[0]] permutation:perm name:nil];
            break;
        }
        case OpKind::Concat:
        {
            NSMutableArray<MPSGraphTensor *> *inputs = [NSMutableArray arrayWithCapacity:node.inputs.size()];
            for (auto idx : node.inputs)
                [inputs addObject:tensors[idx]];
            tensors[i] = [graph concatTensors:inputs dimension:(NSInteger)node.axis name:nil];
            break;
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
            tensors[i] = [graph sliceTensor:tensors[node.inputs[0]] starts:starts ends:ends strides:strides name:nil];
            break;
        }
        case OpKind::Gather:
            tensors[i] = [graph gatherWithUpdatesTensor:tensors[node.inputs[0]]
                                           indicesTensor:tensors[node.inputs[1]]
                                                    axis:(NSUInteger)node.axis
                                         batchDimensions:0
                                                    name:nil];
            break;
        case OpKind::QuantizeLinear:
            tensors[i] = [graph quantizeTensor:tensors[node.inputs[0]]
                                          scale:(double)node.floatAttr0
                                      zeroPoint:(double)node.floatAttr1
                                       dataType:MPSDataTypeInt8
                                           name:nil];
            break;
        case OpKind::DequantizeLinear:
            tensors[i] = [graph dequantizeTensor:tensors[node.inputs[0]]
                                            scale:(double)node.floatAttr0
                                        zeroPoint:(double)node.floatAttr1
                                         dataType:MPSDataTypeFloat32
                                             name:nil];
            break;
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
            tensors[i] = [graph convolution2DWithSourceTensor:tensors[node.inputs[0]]
                                                 weightsTensor:tensors[node.inputs[1]]
                                                    descriptor:convDesc
                                                          name:nil];
            break;
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
                tensors[i] = [graph maxPooling2DWithSourceTensor:tensors[node.inputs[0]] descriptor:poolDesc name:nil];
            else
                tensors[i] = [graph avgPooling2DWithSourceTensor:tensors[node.inputs[0]] descriptor:poolDesc name:nil];
            break;
        }
        case OpKind::Resize:
        {
            const ResizeDescriptor &p = node.resizeParams;
            if (p.mode == ResizeMode::Nearest && p.nearestRoundsDown)
            {
                // The generic resizeTensor:mode: call has no rounding-mode parameter
                // (defaults to RoundPreferCeil for Nearest) — only the Nearest-specific
                // call exposes it, and only as a sizeTensor (not a static MPSShape).
                int32_t sizeVals[2] = {(int32_t)p.outputHeight, (int32_t)p.outputWidth};
                NSData *sizeData = [NSData dataWithBytes:sizeVals length:sizeof(sizeVals)];
                MPSGraphTensor *sizeTensor = [graph constantWithData:sizeData
                                                                 shape:shapeFor({2})
                                                              dataType:MPSDataTypeInt32];
                tensors[i] = [graph resizeNearestWithTensor:tensors[node.inputs[0]]
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
                tensors[i] = [graph resizeTensor:tensors[node.inputs[0]]
                                             size:size
                                             mode:mode
                                     centerResult:p.centerResult
                                     alignCorners:p.alignCorners
                                           layout:MPSGraphTensorNamedDataLayoutNCHW
                                             name:nil];
            }
            break;
        }
        }
    }

    for (auto &[name, nodeIdx] : ir.outputs)
        compiled->outputTensors[name] = tensors[nodeIdx];

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

    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = [NSMutableDictionary dictionary];
    for (auto &[name, tensorNative] : inputs)
    {
        auto it = g->inputTensors.find(name);
        if (it == g->inputTensors.end())
            throw std::runtime_error("campello_nn: MPSGraph backend has no placeholder for input '" + name + "'");
        auto t = (MpsTensor *)tensorNative;
        MPSGraphTensorData *td = [[MPSGraphTensorData alloc] initWithMTLBuffer:t->buffer
                                                                          shape:shapeFor(t->desc.shape)
                                                                       dataType:mpsDataType(t->desc.dataType)];
        feeds[it->second] = td;
    }

    NSMutableArray<MPSGraphTensor *> *targetTensors = [NSMutableArray array];
    std::vector<std::string> targetNames;
    for (auto &[name, tensor] : g->outputTensors)
    {
        [targetTensors addObject:tensor];
        targetNames.push_back(name);
    }

    MPSGraphTensorDataDictionary *results = [g->graph runWithMTLCommandQueue:impl->queue
                                                                         feeds:feeds
                                                                 targetTensors:targetTensors
                                                              targetOperations:nil];

    for (auto &name : targetNames)
    {
        auto it = outputs.find(name);
        if (it == outputs.end())
            continue;
        auto outTensor = (MpsTensor *)it->second;
        MPSGraphTensor *targetTensor = g->outputTensors[name];
        MPSGraphTensorData *resultData = results[targetTensor];
        [[resultData mpsndarray] readBytes:outTensor->buffer.contents strideBytes:nil];
    }

    return new MpsFence{true};
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
    delete (MpsFence *)fenceNative;
}

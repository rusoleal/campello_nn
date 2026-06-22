#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <campello_nn/tflite_importer.hpp>
#include <campello_nn/graph_builder.hpp>
#include <campello_nn/operand.hpp>
#include "tflite_model.hpp"

using namespace systems::leal::campello_nn;

namespace
{
    // TFLite BuiltinOperator values this importer maps (mirrors the parser's
    // own copy in tflite_parser.cpp — kept separate since each file only needs
    // to name the ones it actually switches on).
    enum TfliteBuiltinOperator : int32_t
    {
        TFL_ADD = 0,
        TFL_AVERAGE_POOL_2D = 1,
        TFL_CONCATENATION = 2,
        TFL_CONV_2D = 3,
        TFL_DEPTHWISE_CONV_2D = 4,
        TFL_DEQUANTIZE = 6,
        TFL_FULLY_CONNECTED = 9,
        TFL_LOGISTIC = 14,
        TFL_MAX_POOL_2D = 17,
        TFL_MUL = 18,
        TFL_RELU = 19,
        TFL_RESHAPE = 22,
        TFL_RESIZE_BILINEAR = 23,
        TFL_SOFTMAX = 25,
        TFL_GATHER = 36,
        TFL_TRANSPOSE = 39,
        TFL_RESIZE_NEAREST_NEIGHBOR = 97,
        TFL_QUANTIZE = 114,
        TFL_BATCH_MATMUL = 126,
    };

    // Padding enum (schema.fbs `enum Padding { SAME, VALID }`).
    enum TflitePadding : int8_t
    {
        TFL_PADDING_SAME = 0,
        TFL_PADDING_VALID = 1,
    };

    // ActivationFunctionType values this importer handles (RELU6/TANH/etc. are
    // not yet supported and throw).
    enum TfliteActivation : int8_t
    {
        TFL_ACTIVATION_NONE = 0,
        TFL_ACTIVATION_RELU = 1,
    };

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

    // Generic N-dimensional byte permutation: dst[i]'s size/data comes from
    // src[perm[i]] — same convention as GraphBuilder::transpose()'s `perm`, so
    // this can express the exact same axis reordering on raw constant bytes at
    // import time instead of inserting a runtime transpose op. Used for
    // CONV_2D's OHWI->OIHW weights and FULLY_CONNECTED's [outputs,inputs] ->
    // [inputs,outputs] weights.
    std::vector<uint8_t> permuteBytes(const std::vector<uint8_t> &src, const std::vector<int64_t> &srcShape,
                                       const std::vector<int32_t> &perm, size_t elemSize)
    {
        size_t rank = srcShape.size();
        std::vector<int64_t> dstShape(rank);
        for (size_t i = 0; i < rank; i++)
            dstShape[i] = srcShape[perm[i]];

        std::vector<int64_t> srcStrides(rank), dstStrides(rank);
        srcStrides[rank - 1] = 1;
        dstStrides[rank - 1] = 1;
        for (int64_t i = (int64_t)rank - 2; i >= 0; i--)
        {
            srcStrides[i] = srcStrides[i + 1] * srcShape[i + 1];
            dstStrides[i] = dstStrides[i + 1] * dstShape[i + 1];
        }

        int64_t total = 1;
        for (auto d : srcShape)
            total *= d;

        std::vector<uint8_t> dst(src.size());
        std::vector<int64_t> dstIndex(rank, 0);
        for (int64_t flat = 0; flat < total; flat++)
        {
            int64_t rem = flat;
            for (size_t i = 0; i < rank; i++)
            {
                dstIndex[i] = rem / dstStrides[i];
                rem %= dstStrides[i];
            }
            int64_t srcOffset = 0;
            for (size_t i = 0; i < rank; i++)
                srcOffset += dstIndex[i] * srcStrides[perm[i]];
            std::memcpy(dst.data() + (size_t)flat * elemSize, src.data() + (size_t)srcOffset * elemSize, elemSize);
        }
        return dst;
    }

    struct PadAmounts
    {
        int64_t before;
        int64_t after;
    };

    // TF/TFLite's own SAME-padding formula (the same computation TF itself
    // uses to decide padding for "SAME" Conv/Pool): output = ceil(in/stride),
    // pad_total = max((output-1)*stride + effective_kernel - in, 0), split
    // before/after with any remainder going after.
    PadAmounts computeSamePadding(int64_t inputSize, int64_t kernelSize, int64_t stride, int64_t dilation)
    {
        int64_t effectiveKernel = (kernelSize - 1) * dilation + 1;
        int64_t outputSize = (inputSize + stride - 1) / stride;
        int64_t padTotal = std::max<int64_t>((outputSize - 1) * stride + effectiveKernel - inputSize, 0);
        int64_t padBefore = padTotal / 2;
        return {padBefore, padTotal - padBefore};
    }

    Operand applyFusedActivation(GraphBuilder &builder, Operand x, int8_t activation)
    {
        switch (activation)
        {
        case TFL_ACTIVATION_NONE:
            return x;
        case TFL_ACTIVATION_RELU:
            return builder.relu(x);
        default:
            throw std::runtime_error("campello_nn: TFLite fused activation function " + std::to_string(activation) +
                                      " is not yet supported by the TFLite importer");
        }
    }

    // TFLite expresses axis attributes in its own NHWC numbering (N=0,H=1,W=2,
    // C=3); operands that are still rank-4 at this point in the import are
    // internally NCHW (converted once at the graph-input boundary — see
    // importTfliteFromMemory), so remap N->0,H->2,W->3,C->1. Only correct if no
    // intervening RESHAPE/TRANSPOSE has already changed this operand's
    // dimension correspondence with the original NHWC tensor it traces back to
    // — a real, documented limitation for graphs that reshape/transpose before
    // a CONCATENATION/SOFTMAX/GATHER rather than only at the very output. Not
    // exercised by the common case (these ops appearing right after a
    // straight conv stack), but worth knowing about.
    int32_t remapAxisIfRank4(int32_t axis, const std::vector<int64_t> &shape)
    {
        if (shape.size() != 4)
            return axis;
        int32_t normalized = axis < 0 ? axis + 4 : axis;
        static const int32_t map[4] = {0, 2, 3, 1};
        return map[normalized];
    }

    // Identity permutation with the last two axes swapped — TFLite BATCH_MATMUL's
    // adj_x/adj_y (confirmed against tflite/kernels/batch_matmul.cc: "Transpose
    // the last two dimensions") mean exactly this, applied to x/y respectively
    // before the matmul, since GraphBuilder::matmul() has no transpose flag.
    std::vector<int32_t> swapLastTwoAxes(size_t rank)
    {
        std::vector<int32_t> perm(rank);
        for (size_t i = 0; i < rank; i++)
            perm[i] = (int32_t)i;
        std::swap(perm[rank - 1], perm[rank - 2]);
        return perm;
    }

    // Reads a constant INT32 tensor's raw bytes as a plain int64 list — for
    // TFLite's convention of passing shape/perm/size metadata (RESHAPE's
    // new_shape, TRANSPOSE's perm, RESIZE's size) as a 2nd *tensor* input
    // rather than a node attribute, unlike ONNX which mixes both styles.
    std::vector<int64_t> constantInt32Vector(const tflite::TfliteGraph &g, int32_t tensorIndex)
    {
        const tflite::TfliteTensor &t = g.tensors.at(tensorIndex);
        const tflite::TfliteBuffer &buf = g.buffers.at(t.bufferIndex);
        if (buf.data.empty())
            throw std::runtime_error("campello_nn: TFLite tensor '" + t.name + "' must be a constant for this use");
        size_t n = buf.data.size() / sizeof(int32_t);
        const int32_t *p = (const int32_t *)buf.data.data();
        return std::vector<int64_t>(p, p + n);
    }

    // Lazily binds a TFLite tensor as a campello_nn constant on first use.
    // Graph-input tensors are already bound (and transposed to NCHW) before
    // any operator runs — see importTfliteFromMemory — so reaching here means
    // this tensor must carry its own constant data (a weight, bias, or
    // shape/perm/size metadata tensor).
    Operand resolveTensor(GraphBuilder &builder, const tflite::TfliteGraph &g,
                           std::unordered_map<int32_t, Operand> &values, int32_t tensorIndex)
    {
        auto it = values.find(tensorIndex);
        if (it != values.end())
            return it->second;

        const tflite::TfliteTensor &t = g.tensors.at(tensorIndex);
        const tflite::TfliteBuffer &buf = g.buffers.at(t.bufferIndex);
        if (buf.data.empty())
            throw std::runtime_error("campello_nn: TFLite tensor '" + t.name + "' has no constant data and was "
                                      "never produced by an upstream operator (nor is it a graph input)");
        if (!tflite::tfliteTensorTypeHasDataType(t.type))
            throw std::runtime_error("campello_nn: TFLite tensor '" + t.name +
                                      "' has a type unsupported for binding as a constant");

        TensorDescriptor desc{t.toDataType(), t.shape, false, false};
        Operand op = builder.constant(desc, buf.data.data(), buf.data.size());
        values[tensorIndex] = op;
        return op;
    }

    // Translates one TFLite operator into the equivalent GraphBuilder call(s),
    // storing the result keyed by its (assumed single) output tensor index.
    // Mirrors onnx_importer.cpp's applyNode() in spirit: op coverage is
    // intentionally scoped to what's needed so far, each cut corner throws a
    // clear "not yet supported" error rather than silently producing wrong
    // output.
    void applyOperator(GraphBuilder &builder, const tflite::TfliteOperator &op, const tflite::TfliteGraph &g,
                        std::unordered_map<int32_t, Operand> &values)
    {
        auto in = [&](size_t idx) -> Operand
        {
            if (idx >= op.inputs.size() || op.inputs[idx] < 0)
                throw std::runtime_error("campello_nn: TFLite operator missing required input " + std::to_string(idx));
            return resolveTensor(builder, g, values, op.inputs[idx]);
        };

        if (op.outputs.empty())
            throw std::runtime_error("campello_nn: TFLite operator has no outputs");
        if (op.outputs.size() > 1)
            throw std::runtime_error("campello_nn: TFLite operator has multiple outputs, not yet supported by the TFLite importer");

        Operand result;
        switch (op.builtinCode)
        {
        case TFL_CONV_2D:
        {
            Operand x = in(0);
            std::vector<int64_t> inShape = internal::operandShapeForImport(x);
            if (inShape.size() != 4)
                throw std::runtime_error("campello_nn: TFLite CONV_2D on a non-rank-4 tensor is not yet supported");

            const tflite::TfliteTensor &wt = g.tensors.at(op.inputs[1]);
            const tflite::TfliteBuffer &wbuf = g.buffers.at(wt.bufferIndex);
            if (wbuf.data.empty())
                throw std::runtime_error("campello_nn: TFLite CONV_2D's weight input must be a constant");
            // OHWI -> OIHW (same {0,3,1,2} shape as the NHWC->NCHW activation
            // transpose: O plays N's role here).
            std::vector<uint8_t> oihwBytes =
                permuteBytes(wbuf.data, wt.shape, {0, 3, 1, 2}, elementByteSize(wt.toDataType()));
            std::vector<int64_t> oihwShape = {wt.shape[0], wt.shape[3], wt.shape[1], wt.shape[2]};
            Operand weights =
                builder.constant({wt.toDataType(), oihwShape, false, false}, oihwBytes.data(), oihwBytes.size());

            Conv2dDescriptor desc;
            desc.strideX = op.strideW;
            desc.strideY = op.strideH;
            desc.dilationX = op.dilationW;
            desc.dilationY = op.dilationH;
            if (op.padding == TFL_PADDING_SAME)
            {
                PadAmounts vert = computeSamePadding(inShape[2], wt.shape[1], op.strideH, op.dilationH);
                PadAmounts horiz = computeSamePadding(inShape[3], wt.shape[2], op.strideW, op.dilationW);
                desc.paddingTop = vert.before;
                desc.paddingBottom = vert.after;
                desc.paddingLeft = horiz.before;
                desc.paddingRight = horiz.after;
            }
            Operand convOut = builder.conv2d(x, weights, desc);

            if (op.inputs.size() > 2 && op.inputs[2] >= 0)
            {
                const tflite::TfliteTensor &bt = g.tensors.at(op.inputs[2]);
                const tflite::TfliteBuffer &bbuf = g.buffers.at(bt.bufferIndex);
                if (bbuf.data.empty())
                    throw std::runtime_error("campello_nn: TFLite CONV_2D's bias input must be a constant");
                int64_t C = oihwShape[0];
                Operand biasConst = builder.constant({bt.toDataType(), {1, C, 1, 1}, false, false},
                                                       bbuf.data.data(), bbuf.data.size());
                convOut = builder.add(convOut, biasConst);
            }
            result = applyFusedActivation(builder, convOut, op.fusedActivation);
            break;
        }
        case TFL_DEPTHWISE_CONV_2D:
        {
            // Weight layout confirmed against TFLite's own reference kernel
            // (tflite/kernels/internal/reference/depthwiseconv_float.h):
            // filter_shape is [1, filter_height, filter_width, output_depth]
            // (batch dim always indexed at 0, not output_depth-sized like a
            // regular Conv2D's weight leading dim), and the kernel's own output
            // channel numbering is `oc = m + ic * depth_multiplier` (m = index
            // within the multiplier, ic = input channel) — exactly the standard
            // "groups = input_channels" grouped-conv convention conv2d()'s
            // `groups` parameter already implements (and ONNX import already
            // relies on for ONNX Conv's `group` attribute), so no extra
            // channel-reordering trick is needed beyond the weight reshape below.
            Operand x = in(0);
            std::vector<int64_t> inShape = internal::operandShapeForImport(x);
            if (inShape.size() != 4)
                throw std::runtime_error("campello_nn: TFLite DEPTHWISE_CONV_2D on a non-rank-4 tensor is not yet supported");
            int64_t inputChannels = inShape[1];

            const tflite::TfliteTensor &wt = g.tensors.at(op.inputs[1]);
            const tflite::TfliteBuffer &wbuf = g.buffers.at(wt.bufferIndex);
            if (wbuf.data.empty())
                throw std::runtime_error("campello_nn: TFLite DEPTHWISE_CONV_2D's weight input must be a constant");
            if (wt.shape.size() != 4 || wt.shape[0] != 1)
                throw std::runtime_error("campello_nn: TFLite DEPTHWISE_CONV_2D weight tensor must be shaped [1,H,W,outC]");
            int64_t outputChannels = wt.shape[3];
            if (outputChannels != inputChannels * op.depthMultiplier)
                throw std::runtime_error("campello_nn: TFLite DEPTHWISE_CONV_2D weight's output channel count doesn't "
                                          "match input_channels * depth_multiplier");

            // [1,H,W,outC] -> [outC,1,H,W] (OIHW with inChannels/groups == 1).
            std::vector<uint8_t> oihwBytes =
                permuteBytes(wbuf.data, wt.shape, {3, 0, 1, 2}, elementByteSize(wt.toDataType()));
            std::vector<int64_t> oihwShape = {outputChannels, 1, wt.shape[1], wt.shape[2]};
            Operand weights =
                builder.constant({wt.toDataType(), oihwShape, false, false}, oihwBytes.data(), oihwBytes.size());

            Conv2dDescriptor desc;
            desc.strideX = op.strideW;
            desc.strideY = op.strideH;
            desc.dilationX = op.dilationW;
            desc.dilationY = op.dilationH;
            desc.groups = inputChannels;
            if (op.padding == TFL_PADDING_SAME)
            {
                PadAmounts vert = computeSamePadding(inShape[2], wt.shape[1], op.strideH, op.dilationH);
                PadAmounts horiz = computeSamePadding(inShape[3], wt.shape[2], op.strideW, op.dilationW);
                desc.paddingTop = vert.before;
                desc.paddingBottom = vert.after;
                desc.paddingLeft = horiz.before;
                desc.paddingRight = horiz.after;
            }
            Operand convOut = builder.conv2d(x, weights, desc);

            if (op.inputs.size() > 2 && op.inputs[2] >= 0)
            {
                const tflite::TfliteTensor &bt = g.tensors.at(op.inputs[2]);
                const tflite::TfliteBuffer &bbuf = g.buffers.at(bt.bufferIndex);
                if (bbuf.data.empty())
                    throw std::runtime_error("campello_nn: TFLite DEPTHWISE_CONV_2D's bias input must be a constant");
                Operand biasConst = builder.constant({bt.toDataType(), {1, outputChannels, 1, 1}, false, false},
                                                       bbuf.data.data(), bbuf.data.size());
                convOut = builder.add(convOut, biasConst);
            }
            result = applyFusedActivation(builder, convOut, op.fusedActivation);
            break;
        }
        case TFL_ADD:
            result = applyFusedActivation(builder, builder.add(in(0), in(1)), op.fusedActivation);
            break;
        case TFL_MUL:
            result = applyFusedActivation(builder, builder.mul(in(0), in(1)), op.fusedActivation);
            break;
        case TFL_RELU:
            result = builder.relu(in(0));
            break;
        case TFL_LOGISTIC:
            result = builder.sigmoid(in(0));
            break;
        case TFL_MAX_POOL_2D:
        case TFL_AVERAGE_POOL_2D:
        {
            Operand x = in(0);
            std::vector<int64_t> inShape = internal::operandShapeForImport(x);
            if (inShape.size() != 4)
                throw std::runtime_error("campello_nn: TFLite POOL on a non-rank-4 tensor is not yet supported");
            Pool2dDescriptor desc;
            desc.kernelWidth = op.filterWidth;
            desc.kernelHeight = op.filterHeight;
            desc.strideX = op.strideW;
            desc.strideY = op.strideH;
            if (op.padding == TFL_PADDING_SAME)
            {
                PadAmounts vert = computeSamePadding(inShape[2], op.filterHeight, op.strideH, 1);
                PadAmounts horiz = computeSamePadding(inShape[3], op.filterWidth, op.strideW, 1);
                desc.paddingTop = vert.before;
                desc.paddingBottom = vert.after;
                desc.paddingLeft = horiz.before;
                desc.paddingRight = horiz.after;
            }
            Operand pooled = op.builtinCode == TFL_MAX_POOL_2D ? builder.maxPool2d(x, desc) : builder.avgPool2d(x, desc);
            result = applyFusedActivation(builder, pooled, op.fusedActivation);
            break;
        }
        case TFL_RESHAPE:
        {
            Operand x = in(0);
            std::vector<int64_t> target;
            if (!op.newShape.empty())
                target.assign(op.newShape.begin(), op.newShape.end());
            else if (op.inputs.size() > 1 && op.inputs[1] >= 0)
                target = constantInt32Vector(g, op.inputs[1]);
            else
                throw std::runtime_error("campello_nn: TFLite RESHAPE without a new_shape option or shape input is not yet supported");
            result = builder.reshape(x, target);
            break;
        }
        case TFL_TRANSPOSE:
        {
            if (op.inputs.size() < 2 || op.inputs[1] < 0)
                throw std::runtime_error("campello_nn: TFLite TRANSPOSE without a perm input is not yet supported");
            std::vector<int64_t> permI64 = constantInt32Vector(g, op.inputs[1]);
            std::vector<int32_t> perm(permI64.begin(), permI64.end());
            result = builder.transpose(in(0), perm);
            break;
        }
        case TFL_SOFTMAX:
        {
            Operand x = in(0);
            std::vector<int64_t> shape = internal::operandShapeForImport(x);
            if (op.softmaxBeta != 1.0f)
            {
                float betaVal = op.softmaxBeta;
                Operand betaConst = builder.constant({DataType::Float32, {1}, false, false}, &betaVal, sizeof(float));
                x = builder.mul(x, betaConst);
            }
            // TFLite SOFTMAX always operates over the last (channel) axis.
            int32_t axis = remapAxisIfRank4(-1, shape);
            result = builder.softmax(x, axis);
            break;
        }
        case TFL_CONCATENATION:
        {
            std::vector<Operand> xs;
            for (size_t i = 0; i < op.inputs.size(); i++)
                xs.push_back(in(i));
            std::vector<int64_t> shape = internal::operandShapeForImport(xs[0]);
            int32_t axis = remapAxisIfRank4(op.axis, shape);
            result = applyFusedActivation(builder, builder.concat(xs, axis), op.fusedActivation);
            break;
        }
        case TFL_GATHER:
        {
            if (op.batchDims != 0)
                throw std::runtime_error("campello_nn: TFLite GATHER with batch_dims != 0 is not yet supported");
            Operand data = in(0);
            std::vector<int64_t> shape = internal::operandShapeForImport(data);
            int32_t axis = remapAxisIfRank4(op.axis, shape);
            result = builder.gather(data, in(1), axis);
            break;
        }
        case TFL_FULLY_CONNECTED:
        {
            Operand x = in(0);
            const tflite::TfliteTensor &wt = g.tensors.at(op.inputs[1]);
            const tflite::TfliteBuffer &wbuf = g.buffers.at(wt.bufferIndex);
            if (wbuf.data.empty())
                throw std::runtime_error("campello_nn: TFLite FULLY_CONNECTED's weight input must be a constant");
            // TFLite stores [outputs, inputs] (already "pre-transposed" relative
            // to a plain x @ W); flip to [inputs, outputs] for gemm()'s B.
            std::vector<uint8_t> transposed =
                permuteBytes(wbuf.data, wt.shape, {1, 0}, elementByteSize(wt.toDataType()));
            std::vector<int64_t> transposedShape = {wt.shape[1], wt.shape[0]};
            Operand weights = builder.constant({wt.toDataType(), transposedShape, false, false}, transposed.data(),
                                                 transposed.size());

            bool hasBias = op.inputs.size() > 2 && op.inputs[2] >= 0;
            Operand fcOut;
            if (hasBias)
            {
                const tflite::TfliteTensor &bt = g.tensors.at(op.inputs[2]);
                const tflite::TfliteBuffer &bbuf = g.buffers.at(bt.bufferIndex);
                Operand biasOp =
                    builder.constant({bt.toDataType(), bt.shape, false, false}, bbuf.data.data(), bbuf.data.size());
                fcOut = builder.gemm(x, weights, biasOp, 1.0f, 1.0f);
            }
            else
            {
                fcOut = builder.matmul(x, weights);
            }
            result = applyFusedActivation(builder, fcOut, op.fusedActivation);
            break;
        }
        case TFL_RESIZE_BILINEAR:
        case TFL_RESIZE_NEAREST_NEIGHBOR:
        {
            Operand x = in(0);
            std::vector<int64_t> inShape = internal::operandShapeForImport(x);
            if (inShape.size() != 4)
                throw std::runtime_error("campello_nn: TFLite RESIZE on a non-rank-4 tensor is not yet supported");
            if (op.inputs.size() < 2 || op.inputs[1] < 0)
                throw std::runtime_error("campello_nn: TFLite RESIZE without a size input is not yet supported");
            std::vector<int64_t> sizes = constantInt32Vector(g, op.inputs[1]); // [newHeight, newWidth]
            if (sizes.size() < 2)
                throw std::runtime_error("campello_nn: TFLite RESIZE size input must have 2 entries");
            ResizeDescriptor desc;
            desc.outputHeight = sizes[0];
            desc.outputWidth = sizes[1];
            desc.mode = op.builtinCode == TFL_RESIZE_BILINEAR ? ResizeMode::Bilinear : ResizeMode::Nearest;
            desc.alignCorners = op.alignCorners;
            desc.centerResult = op.halfPixelCenters;
            result = builder.resize(x, desc);
            break;
        }
        case TFL_QUANTIZE:
        {
            const tflite::TfliteTensor &outT = g.tensors.at(op.outputs[0]);
            if (!outT.hasQuantization)
                throw std::runtime_error("campello_nn: TFLite QUANTIZE output tensor has no quantization parameters");
            result = builder.quantizeLinear(in(0), outT.quantScale, outT.quantZeroPoint);
            break;
        }
        case TFL_DEQUANTIZE:
        {
            const tflite::TfliteTensor &inT = g.tensors.at(op.inputs[0]);
            if (!inT.hasQuantization)
                throw std::runtime_error("campello_nn: TFLite DEQUANTIZE input tensor has no quantization parameters");
            result = builder.dequantizeLinear(in(0), inT.quantScale, inT.quantZeroPoint);
            break;
        }
        case TFL_BATCH_MATMUL:
        {
            Operand x = in(0);
            Operand y = in(1);
            if (op.adjX)
            {
                std::vector<int64_t> xShape = internal::operandShapeForImport(x);
                if (xShape.size() < 2)
                    throw std::runtime_error("campello_nn: TFLite BATCH_MATMUL operand must be at least rank 2");
                x = builder.transpose(x, swapLastTwoAxes(xShape.size()));
            }
            if (op.adjY)
            {
                std::vector<int64_t> yShape = internal::operandShapeForImport(y);
                if (yShape.size() < 2)
                    throw std::runtime_error("campello_nn: TFLite BATCH_MATMUL operand must be at least rank 2");
                y = builder.transpose(y, swapLastTwoAxes(yShape.size()));
            }
            result = builder.matmul(x, y);
            break;
        }
        default:
            throw std::runtime_error("campello_nn: TFLite builtin operator code " + std::to_string(op.builtinCode) +
                                      " is not yet supported by the TFLite importer");
        }

        values[op.outputs[0]] = result;
    }
} // namespace

TfliteImportResult systems::leal::campello_nn::importTfliteFromMemory(std::shared_ptr<Context> context,
                                                                        const uint8_t *data, size_t size)
{
    std::vector<uint8_t> bytes(data, data + size);
    tflite::TfliteGraph g = tflite::parseTfliteModel(bytes);

    GraphBuilder builder(context);
    std::unordered_map<int32_t, Operand> values;
    TfliteImportResult result;

    // Graph inputs: bind with TFLite's own declared shape (NHWC for rank-4),
    // then immediately convert rank-4 inputs to NCHW so every downstream op
    // operates in campello_nn's native layout — see tflite_importer.hpp.
    for (int32_t tensorIndex : g.graphInputs)
    {
        const tflite::TfliteTensor &t = g.tensors.at(tensorIndex);
        if (!tflite::tfliteTensorTypeHasDataType(t.type))
            throw std::runtime_error("campello_nn: TFLite graph input '" + t.name + "' has an unsupported type");
        std::string name = t.name.empty() ? ("input" + std::to_string(tensorIndex)) : t.name;
        TensorDescriptor desc{t.toDataType(), t.shape, false, true};
        Operand x = builder.input(name, desc);
        result.inputs[name] = desc;
        if (t.shape.size() == 4)
            x = builder.transpose(x, {0, 3, 1, 2});
        values[tensorIndex] = x;
    }

    // TFLite's operators array is already in execution order (schema.fbs:
    // "All operators, in execution order"), same single-linear-pass assumption
    // the ONNX importer makes.
    for (auto &op : g.operators)
        applyOperator(builder, op, g, values);

    // Convert rank-4 outputs back to NHWC before finalizing, so the returned
    // TensorDescriptor matches what the original .tflite file itself declares
    // for that output — mirrors the ONNX importer's principle of trusting
    // internal::operandShapeForImport() for the *true* current shape.
    std::unordered_map<std::string, Operand> outputOperands;
    for (int32_t tensorIndex : g.graphOutputs)
    {
        const tflite::TfliteTensor &t = g.tensors.at(tensorIndex);
        Operand outOp = values.at(tensorIndex);
        std::vector<int64_t> currentShape = internal::operandShapeForImport(outOp);
        if (currentShape.size() == 4)
            outOp = builder.transpose(outOp, {0, 2, 3, 1});
        std::string name = t.name.empty() ? ("output" + std::to_string(tensorIndex)) : t.name;
        outputOperands[name] = outOp;
        result.outputs[name] = TensorDescriptor{t.toDataType(), internal::operandShapeForImport(outOp), true, false};
    }
    result.graph = builder.build(outputOperands);
    return result;
}

TfliteImportResult systems::leal::campello_nn::importTfliteFromFile(std::shared_ptr<Context> context, const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("campello_nn: importTfliteFromFile() cannot open '" + path + "'");
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return importTfliteFromMemory(context, bytes.data(), bytes.size());
}

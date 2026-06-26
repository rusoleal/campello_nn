#include <cmath>
#include <fstream>
#include <stdexcept>
#include <campello_nn/onnx_importer.hpp>
#include <campello_nn/graph_builder.hpp>
#include <campello_nn/operand.hpp>
#include "onnx_model.hpp"

using namespace systems::leal::campello_nn;

namespace
{
    std::vector<int64_t> getIntsAttr(const onnx::OnnxNode &node, const std::string &name,
                                      std::vector<int64_t> defaultVal)
    {
        const onnx::OnnxAttribute *a = node.findAttribute(name);
        return (a && !a->ints.empty()) ? a->ints : defaultVal;
    }

    int64_t getIntAttr(const onnx::OnnxNode &node, const std::string &name, int64_t defaultVal)
    {
        const onnx::OnnxAttribute *a = node.findAttribute(name);
        return a ? a->i : defaultVal;
    }

    float getFloatAttr(const onnx::OnnxNode &node, const std::string &name, float defaultVal)
    {
        const onnx::OnnxAttribute *a = node.findAttribute(name);
        return a ? a->f : defaultVal;
    }

    std::string getStringAttr(const onnx::OnnxNode &node, const std::string &name, const std::string &defaultVal)
    {
        const onnx::OnnxAttribute *a = node.findAttribute(name);
        return (a && !a->s.empty()) ? a->s : defaultVal;
    }

    // Resolves ONNX Reshape's 0 ("copy from input shape") and -1 ("infer from
    // total element count") sentinels into a concrete shape.
    std::vector<int64_t> resolveReshapeTarget(const std::vector<int64_t> &targetShape,
                                               const std::vector<int64_t> &inputShape)
    {
        std::vector<int64_t> resolved = targetShape;
        int64_t inferIndex = -1;
        int64_t knownProduct = 1;
        for (size_t i = 0; i < resolved.size(); i++)
        {
            if (resolved[i] == 0)
            {
                if (i >= inputShape.size())
                    throw std::runtime_error("campello_nn: ONNX Reshape '0' dimension has no corresponding input dimension");
                resolved[i] = inputShape[i];
            }
            if (resolved[i] == -1)
                inferIndex = (int64_t)i;
            else
                knownProduct *= resolved[i];
        }
        if (inferIndex >= 0)
        {
            int64_t totalInput = 1;
            for (auto d : inputShape)
                totalInput *= d;
            if (knownProduct == 0 || totalInput % knownProduct != 0)
                throw std::runtime_error("campello_nn: ONNX Reshape's inferred (-1) dimension does not evenly divide the input element count");
            resolved[inferIndex] = totalInput / knownProduct;
        }
        return resolved;
    }

    // Translates one ONNX node into the equivalent GraphBuilder call(s). Op
    // coverage is intentionally scoped to what's needed so far — each cut corner
    // (Conv with a fused bias input, Gemm's transA/transB, multi-output ops) throws
    // a clear "not yet supported" error rather than silently producing wrong output.
    Operand applyNode(GraphBuilder &builder, const onnx::OnnxNode &node,
                       std::unordered_map<std::string, Operand> &values,
                       const onnx::OnnxGraph &onnxGraph)
    {
        auto in = [&](size_t idx) -> Operand
        {
            if (idx >= node.inputs.size())
                throw std::runtime_error("campello_nn: ONNX node '" + node.opType + "' missing input " + std::to_string(idx));
            return values.at(node.inputs[idx]);
        };

        if (node.opType == "Conv")
        {
            auto strides = getIntsAttr(node, "strides", {1, 1});
            auto pads = getIntsAttr(node, "pads", {0, 0, 0, 0});
            auto dilations = getIntsAttr(node, "dilations", {1, 1});
            Conv2dDescriptor desc;
            desc.strideY = strides[0];
            desc.strideX = strides[1];
            desc.dilationY = dilations[0];
            desc.dilationX = dilations[1];
            desc.paddingTop = pads[0];
            desc.paddingLeft = pads[1];
            desc.paddingBottom = pads[2];
            desc.paddingRight = pads[3];
            desc.groups = getIntAttr(node, "group", 1);
            Operand convOut = builder.conv2d(in(0), in(1), desc);

            if (node.inputs.size() <= 2)
                return convOut;

            auto biasIt = onnxGraph.initializers.find(node.inputs[2]);
            if (biasIt == onnxGraph.initializers.end())
                throw std::runtime_error("campello_nn: ONNX Conv's bias input must be a constant/initializer "
                                          "(a computed bias is not yet supported by the ONNX importer)");
            std::vector<int64_t> outShape = internal::operandShapeForImport(convOut);
            if (outShape.size() != 4)
                throw std::runtime_error("campello_nn: ONNX Conv output must be rank-4 (NCHW)");
            int64_t C = outShape[1];
            // Bias is per-output-channel; reshape to [1,C,1,1] and let add()'s
            // NumPy-style broadcasting expand it across N/H/W.
            Operand biasConst = builder.constant({DataType::Float32, {1, C, 1, 1}, false, false},
                                                  biasIt->second.bytes.data(), biasIt->second.bytes.size());
            return builder.add(convOut, biasConst);
        }
        if (node.opType == "Add")
            return builder.add(in(0), in(1));
        if (node.opType == "Mul")
            return builder.mul(in(0), in(1));
        if (node.opType == "Relu")
            return builder.relu(in(0));
        if (node.opType == "Sigmoid")
            return builder.sigmoid(in(0));
        if (node.opType == "MatMul")
            return builder.matmul(in(0), in(1));
        if (node.opType == "Gemm")
        {
            if (node.inputs.size() < 3)
                throw std::runtime_error("campello_nn: ONNX Gemm without a C input is not yet supported by the ONNX importer");
            float alpha = getFloatAttr(node, "alpha", 1.0f);
            float beta = getFloatAttr(node, "beta", 1.0f);
            Operand a = in(0);
            Operand b = in(1);
            if (getIntAttr(node, "transA", 0) != 0)
                a = builder.transpose(a, {1, 0});
            if (getIntAttr(node, "transB", 0) != 0)
                b = builder.transpose(b, {1, 0});
            return builder.gemm(a, b, in(2), alpha, beta);
        }
        if (node.opType == "BatchNormalization")
        {
            // ONNX input order is (X, scale, B, mean, var); campello_nn's batchNorm
            // takes (x, mean, variance, scale, bias, eps) — different order.
            float eps = getFloatAttr(node, "epsilon", 1e-5f);
            return builder.batchNorm(in(0), in(3), in(4), in(1), in(2), eps);
        }
        if (node.opType == "Transpose")
        {
            auto permI64 = getIntsAttr(node, "perm", {});
            if (permI64.empty())
                throw std::runtime_error("campello_nn: ONNX Transpose without a 'perm' attribute is not yet supported by the ONNX importer");
            std::vector<int32_t> perm(permI64.begin(), permI64.end());
            return builder.transpose(in(0), perm);
        }
        if (node.opType == "MaxPool" || node.opType == "AveragePool")
        {
            auto kernel = getIntsAttr(node, "kernel_shape", {});
            if (kernel.empty())
                throw std::runtime_error("campello_nn: ONNX " + node.opType + " without 'kernel_shape' is not yet supported by the ONNX importer");
            auto strides = getIntsAttr(node, "strides", {1, 1});
            auto pads = getIntsAttr(node, "pads", {0, 0, 0, 0});
            Pool2dDescriptor desc;
            desc.kernelHeight = kernel[0];
            desc.kernelWidth = kernel[1];
            desc.strideY = strides[0];
            desc.strideX = strides[1];
            desc.paddingTop = pads[0];
            desc.paddingLeft = pads[1];
            desc.paddingBottom = pads[2];
            desc.paddingRight = pads[3];
            return node.opType == "MaxPool" ? builder.maxPool2d(in(0), desc) : builder.avgPool2d(in(0), desc);
        }
        if (node.opType == "GlobalAveragePool")
        {
            Operand x = in(0);
            std::vector<int64_t> inputShape = internal::operandShapeForImport(x);
            if (inputShape.size() != 4)
                throw std::runtime_error("campello_nn: ONNX GlobalAveragePool expects a rank-4 (NCHW) input");
            Pool2dDescriptor desc;
            desc.kernelHeight = inputShape[2];
            desc.kernelWidth = inputShape[3];
            desc.strideY = inputShape[2];
            desc.strideX = inputShape[3];
            return builder.avgPool2d(x, desc);
        }
        if (node.opType == "Flatten")
        {
            Operand x = in(0);
            std::vector<int64_t> inputShape = internal::operandShapeForImport(x);
            int64_t axis = getIntAttr(node, "axis", 1);
            if (axis < 0)
                axis += static_cast<int64_t>(inputShape.size());
            if (axis < 0 || axis > static_cast<int64_t>(inputShape.size()))
                throw std::runtime_error("campello_nn: ONNX Flatten 'axis' out of range");
            int64_t outer = 1, inner = 1;
            for (int64_t i = 0; i < axis; ++i)
                outer *= inputShape[i];
            for (size_t i = axis; i < inputShape.size(); ++i)
                inner *= inputShape[i];
            return builder.reshape(x, {outer, inner});
        }
        if (node.opType == "Reshape")
        {
            auto it = onnxGraph.initializers.find(node.inputs[1]);
            if (it == onnxGraph.initializers.end())
                throw std::runtime_error("campello_nn: ONNX Reshape's shape input must be a constant/initializer "
                                          "(a computed shape input is not yet supported by the ONNX importer)");
            Operand x = in(0);
            std::vector<int64_t> target = resolveReshapeTarget(it->second.toInt64Vector(), internal::operandShapeForImport(x));
            return builder.reshape(x, target);
        }
        if (node.opType == "Resize")
        {
            std::string coordMode = getStringAttr(node, "coordinate_transformation_mode", "half_pixel");
            std::string mode = getStringAttr(node, "mode", "nearest");
            std::string nearestMode = getStringAttr(node, "nearest_mode", "round_prefer_floor");

            ResizeDescriptor desc;
            if (mode == "nearest")
                desc.mode = ResizeMode::Nearest;
            else if (mode == "linear")
                desc.mode = ResizeMode::Bilinear;
            else
                throw std::runtime_error("campello_nn: ONNX Resize mode '" + mode + "' is not yet supported by the ONNX importer");

            if (coordMode == "asymmetric")
            {
                desc.centerResult = false;
                desc.alignCorners = false;
            }
            else if (coordMode == "align_corners")
            {
                desc.alignCorners = true;
            }
            else if (coordMode == "half_pixel" || coordMode == "pytorch_half_pixel")
            {
                desc.centerResult = true;
                desc.alignCorners = false;
            }
            else
            {
                throw std::runtime_error("campello_nn: ONNX Resize coordinate_transformation_mode '" + coordMode +
                                          "' is not yet supported by the ONNX importer");
            }
            desc.nearestRoundsDown = (mode == "nearest" && nearestMode == "floor");

            Operand x = in(0);
            std::vector<int64_t> inputShape = internal::operandShapeForImport(x);
            if (inputShape.size() != 4)
                throw std::runtime_error("campello_nn: ONNX Resize on a non-rank-4 tensor is not yet supported by the ONNX importer");

            // Resize(X, roi, scales, sizes): scales and sizes are mutually exclusive,
            // whichever is present (non-empty) is the one this node actually uses.
            if (node.inputs.size() > 3 && !node.inputs[3].empty() && onnxGraph.initializers.count(node.inputs[3]))
            {
                auto sizes = onnxGraph.initializers.at(node.inputs[3]).toInt64Vector();
                desc.outputHeight = sizes[2];
                desc.outputWidth = sizes[3];
            }
            else if (node.inputs.size() > 2 && !node.inputs[2].empty() && onnxGraph.initializers.count(node.inputs[2]))
            {
                auto it = onnxGraph.initializers.find(node.inputs[2]);
                size_t n = it->second.bytes.size() / sizeof(float);
                if (n != 4)
                    throw std::runtime_error("campello_nn: ONNX Resize 'scales' must have 4 entries (NCHW)");
                const float *scales = (const float *)it->second.bytes.data();
                desc.outputHeight = (int64_t)std::lround(inputShape[2] * scales[2]);
                desc.outputWidth = (int64_t)std::lround(inputShape[3] * scales[3]);
            }
            else
            {
                throw std::runtime_error("campello_nn: ONNX Resize without a constant 'scales' or 'sizes' input is not yet supported");
            }
            return builder.resize(x, desc);
        }

        throw std::runtime_error("campello_nn: ONNX op '" + node.opType + "' is not yet supported by the ONNX importer");
    }
}

OnnxImportResult systems::leal::campello_nn::importOnnxFromMemory(std::shared_ptr<Context> context, const uint8_t *data, size_t size)
{
    std::vector<uint8_t> bytes(data, data + size);
    onnx::OnnxGraph onnxGraph = onnx::parseOnnxModel(bytes);

    GraphBuilder builder(context);
    std::unordered_map<std::string, Operand> values;
    OnnxImportResult result;

    for (auto &vi : onnxGraph.inputs)
    {
        // ONNX models often list initializers (weights) as graph inputs too.
        // Treat them as constants below; don't expose them as user-provided inputs.
        if (onnxGraph.initializers.count(vi.name))
            continue;
        TensorDescriptor desc{onnx::onnxElemTypeToDataType(vi.elemType), vi.shape, false, true};
        values[vi.name] = builder.input(vi.name, desc);
        result.inputs[vi.name] = desc;
    }

    for (auto &[name, tensor] : onnxGraph.initializers)
    {
        // Some initializers (e.g. a Reshape/Resize shape or scales constant,
        // commonly INT64) are only ever consumed at import time as raw values —
        // see applyNode — and never need to become a campello_nn Tensor/Operand.
        if (!onnx::onnxElemTypeHasDataType(tensor.elemType))
            continue;
        TensorDescriptor desc{tensor.toDataType(), tensor.shape, false, false};
        values[name] = builder.constant(desc, tensor.bytes.data(), tensor.bytes.size());
    }

    // ONNX requires nodes to already be in topological order, same assumption
    // campello_nn's own IR makes — a single linear pass is enough.
    for (auto &node : onnxGraph.nodes)
    {
        if (node.outputs.empty())
            throw std::runtime_error("campello_nn: ONNX node '" + node.opType + "' has no outputs");
        if (node.outputs.size() > 1)
            throw std::runtime_error("campello_nn: ONNX op '" + node.opType + "' has multiple outputs, not yet supported by the ONNX importer");
        values[node.outputs[0]] = applyNode(builder, node, values, onnxGraph);
    }

    // Uses the actual built graph's inferred shape (via operandShapeForImport),
    // not the file's declared output shape — more robust against a model that
    // declares a dynamic/symbolic output dimension, which the parser can only
    // default to 1 (see parseDimension in onnx_parser.cpp) without knowing better.
    std::unordered_map<std::string, Operand> outputOperands;
    for (auto &vi : onnxGraph.outputs)
    {
        Operand op = values.at(vi.name);
        outputOperands[vi.name] = op;
        result.outputs[vi.name] = TensorDescriptor{onnx::onnxElemTypeToDataType(vi.elemType),
                                                     internal::operandShapeForImport(op), true, false};
    }
    result.info = internal::graphInfoForImport(builder, outputOperands);
    result.graph = builder.build(outputOperands);
    return result;
}

OnnxImportResult systems::leal::campello_nn::importOnnxFromFile(std::shared_ptr<Context> context, const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("campello_nn: importOnnxFromFile() cannot open '" + path + "'");
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return importOnnxFromMemory(context, bytes.data(), bytes.size());
}

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <campello_nn/constants/data_type.hpp>

namespace systems::leal::campello_nn::onnx
{

    /// Maps an ONNX `TensorProto.DataType`/`TypeProto.Tensor.elem_type` value to
    /// `campello_nn::DataType`. Throws for ONNX types campello_nn doesn't represent
    /// (e.g. INT64, BOOL, DOUBLE, STRING).
    DataType onnxElemTypeToDataType(int32_t elemType);

    /// Non-throwing check for the above — lets callers skip an ONNX tensor that's
    /// only ever consumed at import time (e.g. a `Reshape`/`Resize` shape/scale
    /// constant, commonly INT64) instead of bound as a campello_nn `Tensor`.
    bool onnxElemTypeHasDataType(int32_t elemType);

    /**
     * @brief Raw tensor data straight off the wire, before any campello_nn-specific
     * interpretation. `elemType` is ONNX's own `TensorProto.DataType` enum value
     * (1=FLOAT, 10=FLOAT16, 6=INT32, 12=UINT32, 3=INT8, 7=INT64, ...) — not yet
     * mapped to `campello_nn::DataType`, since some ONNX tensors (e.g. an int64
     * shape constant feeding `Reshape`) are consumed at import time as plain C++
     * values and never become a campello_nn `Tensor` at all.
     */
    struct OnnxTensor
    {
        std::vector<int64_t> shape;
        int32_t elemType = 0;
        std::vector<uint8_t> bytes;

        /// Maps `elemType` to `campello_nn::DataType`. Throws for ONNX types
        /// campello_nn doesn't represent (e.g. INT64, BOOL, DOUBLE, STRING).
        DataType toDataType() const;

        /// Decodes `bytes` as a list of int64 values, regardless of `elemType`
        /// (INT64, INT32, or FLOAT — the last truncates). For reading ONNX
        /// `Reshape`/`Squeeze`-style shape constants, which are plain import-time
        /// values, not campello_nn tensors.
        std::vector<int64_t> toInt64Vector() const;
    };

    struct OnnxAttribute
    {
        std::string name;
        float f = 0.0f;
        int64_t i = 0;
        std::string s;
        std::vector<float> floats;
        std::vector<int64_t> ints;
        OnnxTensor tensor;
        bool hasTensor = false;
    };

    struct OnnxNode
    {
        std::string opType;
        std::string name;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::unordered_map<std::string, OnnxAttribute> attributes;

        const OnnxAttribute *findAttribute(const std::string &attrName) const
        {
            auto it = attributes.find(attrName);
            return it == attributes.end() ? nullptr : &it->second;
        }
    };

    struct OnnxValueInfo
    {
        std::string name;
        std::vector<int64_t> shape;
        int32_t elemType = 0;
    };

    struct OnnxGraph
    {
        std::vector<OnnxNode> nodes;
        std::unordered_map<std::string, OnnxTensor> initializers;
        std::vector<OnnxValueInfo> inputs;
        std::vector<OnnxValueInfo> outputs;
    };

    /// Parses a `.onnx` file's bytes (a serialized `ModelProto`) into an `OnnxGraph`.
    /// Throws `std::runtime_error` on malformed/truncated input.
    OnnxGraph parseOnnxModel(const std::vector<uint8_t> &fileBytes);

} // namespace systems::leal::campello_nn::onnx

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <campello_nn/constants/data_type.hpp>

namespace systems::leal::campello_nn::tflite
{

    /// Maps a TFLite `TensorType` value to `campello_nn::DataType`. Throws for
    /// TFLite types campello_nn doesn't represent (e.g. INT64, BOOL, STRING).
    DataType tfliteTensorTypeToDataType(int8_t tensorType);

    /// Non-throwing check for the above.
    bool tfliteTensorTypeHasDataType(int8_t tensorType);

    /// A raw data buffer (used for constant tensors), copied out of the
    /// FlatBuffer's `Buffer.data` field. Empty for the mandatory sentinel
    /// buffer 0 and for tensors with no constant data (graph inputs,
    /// intermediate results).
    struct TfliteBuffer
    {
        std::vector<uint8_t> data;
    };

    /// A parsed `Tensor` table. `shape` is TFLite's own declared shape — NHWC
    /// for rank-4 tensors, not yet converted to campello_nn's NCHW convention
    /// (the importer does that at the graph boundary, see tflite_importer.cpp).
    struct TfliteTensor
    {
        std::vector<int64_t> shape;
        int8_t type = 0; // TFLite TensorType enum value
        uint32_t bufferIndex = 0;
        std::string name;

        // Per-tensor (scalar scale/zero_point) quantization only — TFLite's
        // per-channel/blockwise QuantizationDetails union is not yet supported.
        bool hasQuantization = false;
        float quantScale = 1.0f;
        int32_t quantZeroPoint = 0;

        /// Maps `type` to `campello_nn::DataType`. Throws for unsupported types.
        DataType toDataType() const;
    };

    /// A parsed `Operator` table. Unlike the ONNX importer's generic
    /// name-keyed `OnnxAttribute` map (ONNX's `AttributeProto` is itself
    /// generic), TFLite's builtin-options are per-op-type FlatBuffers tables
    /// known entirely from `builtinCode` — so the parser eagerly extracts every
    /// field any *supported* op might need into these flat named fields,
    /// rather than keeping a generic options blob the importer would have to
    /// re-interpret itself.
    struct TfliteOperator
    {
        int32_t builtinCode = -1; // BuiltinOperator value
        std::vector<int32_t> inputs;  // tensor indices; -1 marks an absent optional input
        std::vector<int32_t> outputs; // tensor indices

        // Conv2DOptions / Pool2DOptions (shared shape; DepthwiseConv2DOptions is
        // not parsed — DEPTHWISE_CONV_2D is not yet supported, see importer).
        int32_t padding = 0; // Padding enum: 0=SAME, 1=VALID
        int32_t strideW = 1, strideH = 1;
        int32_t dilationW = 1, dilationH = 1;     // Conv2DOptions only
        int32_t filterWidth = 1, filterHeight = 1; // Pool2DOptions only
        int32_t fusedActivation = 0;                // ActivationFunctionType enum value

        // ReshapeOptions
        std::vector<int32_t> newShape;

        // ConcatenationOptions / GatherOptions / SoftmaxOptions
        int32_t axis = 0;
        int32_t batchDims = 0;
        float softmaxBeta = 1.0f;

        // ResizeBilinearOptions / ResizeNearestNeighborOptions
        bool alignCorners = false;
        bool halfPixelCenters = false;
    };

    /// A parsed TFLite model's *primary* (subgraph 0) graph — campello_nn, like
    /// the ONNX importer, only ever imports a single graph; control-flow ops
    /// (IF/WHILE/CALL) that reference other subgraphs are out of scope.
    struct TfliteGraph
    {
        std::vector<TfliteTensor> tensors;
        std::vector<TfliteOperator> operators; // in execution order, per the schema
        std::vector<int32_t> graphInputs;  // indices into tensors
        std::vector<int32_t> graphOutputs; // indices into tensors
        std::vector<TfliteBuffer> buffers;
    };

    /// Parses a `.tflite` file's bytes (a serialized FlatBuffers `Model`,
    /// subgraph 0 only) into a `TfliteGraph`. Throws `std::runtime_error` if
    /// the `"TFL3"` file identifier is missing or the buffer is malformed.
    TfliteGraph parseTfliteModel(const std::vector<uint8_t> &fileBytes);

} // namespace systems::leal::campello_nn::tflite

#include <algorithm>
#include <stdexcept>
#include <flatbuffers/flatbuffers.h>
#include "tflite_model.hpp"

using namespace systems::leal::campello_nn;
namespace fb = flatbuffers;

namespace
{
    // TFLite TensorType enum values (schema.fbs, `enum TensorType : byte`).
    enum TfliteTensorTypeValue : int8_t
    {
        TFL_FLOAT32 = 0,
        TFL_FLOAT16 = 1,
        TFL_INT32 = 2,
        TFL_INT8 = 9,
        TFL_UINT32 = 15,
    };

    // TFLite BuiltinOperator enum values (schema.fbs, `enum BuiltinOperator : int32`)
    // — only the ones this importer maps onto a GraphBuilder call.
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

    // Field-id constants per FlatBuffers table, in declared schema order — see
    // tflite_parser.cpp's header comment / the plan this was built from for the
    // full verified field lists (every table also has fields beyond what's
    // listed here; only what this importer reads is named). Vtable field id is
    // 0-indexed declaration order; every `union` field implicitly consumes an
    // extra slot for its hidden `<name>_type` discriminant immediately before
    // it, which is why some field ids below skip a number.
    namespace ModelField
    {
        enum
        {
            Version = 0,
            OperatorCodes = 1,
            Subgraphs = 2,
            Buffers = 4
        };
    }
    namespace SubGraphField
    {
        enum
        {
            Tensors = 0,
            Inputs = 1,
            Outputs = 2,
            Operators = 3
        };
    }
    namespace OperatorField
    {
        enum
        {
            OpcodeIndex = 0,
            Inputs = 1,
            Outputs = 2,
            // 3 = implicit builtin_options_type
            BuiltinOptions = 4
        };
    }
    namespace OperatorCodeField
    {
        enum
        {
            DeprecatedBuiltinCode = 0,
            BuiltinCode = 3
        };
    }
    namespace TensorField
    {
        enum
        {
            Shape = 0,
            Type = 1,
            Buffer = 2,
            Name = 3,
            Quantization = 4
        };
    }
    namespace BufferField
    {
        enum
        {
            Data = 0
        };
    }
    namespace QuantParamField
    {
        enum
        {
            Scale = 2,
            ZeroPoint = 3
        };
    }
    namespace Conv2DOptionsField
    {
        enum
        {
            Padding = 0,
            StrideW = 1,
            StrideH = 2,
            FusedActivation = 3,
            DilationW = 4,
            DilationH = 5
        };
    }
    namespace Pool2DOptionsField
    {
        enum
        {
            Padding = 0,
            StrideW = 1,
            StrideH = 2,
            FilterWidth = 3,
            FilterHeight = 4,
            FusedActivation = 5
        };
    }
    namespace DepthwiseConv2DOptionsField
    {
        enum
        {
            Padding = 0,
            StrideW = 1,
            StrideH = 2,
            DepthMultiplier = 3,
            FusedActivation = 4,
            DilationW = 5,
            DilationH = 6
        };
    }
    namespace ReshapeOptionsField
    {
        enum
        {
            NewShape = 0
        };
    }
    namespace BatchMatMulOptionsField
    {
        enum
        {
            AdjX = 0,
            AdjY = 1
        };
    }
    namespace ActivationOnlyOptionsField // AddOptions / MulOptions / FullyConnectedOptions
    {
        enum
        {
            FusedActivation = 0
        };
    }
    namespace SoftmaxOptionsField
    {
        enum
        {
            Beta = 0
        };
    }
    namespace ConcatenationOptionsField
    {
        enum
        {
            Axis = 0,
            FusedActivation = 1
        };
    }
    namespace GatherOptionsField
    {
        enum
        {
            Axis = 0,
            BatchDims = 1
        };
    }
    namespace ResizeOptionsField // ResizeBilinearOptions / ResizeNearestNeighborOptions share this shape
    {
        enum
        {
            AlignCorners = 0,
            HalfPixelCenters = 1
        };
    }

    fb::voffset_t fieldOffset(int fieldId)
    {
        return (fb::voffset_t)((fieldId + 2) * sizeof(fb::voffset_t));
    }

    template <typename T>
    T getScalar(const fb::Table *t, int fieldId, T defaultVal)
    {
        return t->GetField<T>(fieldOffset(fieldId), defaultVal);
    }

    const fb::Table *getSubTable(const fb::Table *t, int fieldId)
    {
        return t->GetPointer<const fb::Table *>(fieldOffset(fieldId));
    }

    template <typename T>
    const fb::Vector<T> *getVector(const fb::Table *t, int fieldId)
    {
        return t->GetPointer<const fb::Vector<T> *>(fieldOffset(fieldId));
    }

    const fb::Vector<fb::Offset<fb::Table>> *getTableVector(const fb::Table *t, int fieldId)
    {
        return t->GetPointer<const fb::Vector<fb::Offset<fb::Table>> *>(fieldOffset(fieldId));
    }

    const fb::String *getStringField(const fb::Table *t, int fieldId)
    {
        return t->GetPointer<const fb::String *>(fieldOffset(fieldId));
    }

    // Extracts the builtin-options fields this importer needs into `to`'s flat
    // named fields — see tflite_model.hpp's TfliteOperator comment for why this
    // is done eagerly here rather than keeping a generic options blob.
    void parseOptionsForOp(const fb::Table *opts, tflite::TfliteOperator &to)
    {
        switch (to.builtinCode)
        {
        case TFL_CONV_2D:
            to.padding = getScalar<int8_t>(opts, Conv2DOptionsField::Padding, 0);
            to.strideW = getScalar<int32_t>(opts, Conv2DOptionsField::StrideW, 1);
            to.strideH = getScalar<int32_t>(opts, Conv2DOptionsField::StrideH, 1);
            to.fusedActivation = getScalar<int8_t>(opts, Conv2DOptionsField::FusedActivation, 0);
            to.dilationW = getScalar<int32_t>(opts, Conv2DOptionsField::DilationW, 1);
            to.dilationH = getScalar<int32_t>(opts, Conv2DOptionsField::DilationH, 1);
            break;
        case TFL_MAX_POOL_2D:
        case TFL_AVERAGE_POOL_2D:
            to.padding = getScalar<int8_t>(opts, Pool2DOptionsField::Padding, 0);
            to.strideW = getScalar<int32_t>(opts, Pool2DOptionsField::StrideW, 1);
            to.strideH = getScalar<int32_t>(opts, Pool2DOptionsField::StrideH, 1);
            to.filterWidth = getScalar<int32_t>(opts, Pool2DOptionsField::FilterWidth, 1);
            to.filterHeight = getScalar<int32_t>(opts, Pool2DOptionsField::FilterHeight, 1);
            to.fusedActivation = getScalar<int8_t>(opts, Pool2DOptionsField::FusedActivation, 0);
            break;
        case TFL_DEPTHWISE_CONV_2D:
            to.padding = getScalar<int8_t>(opts, DepthwiseConv2DOptionsField::Padding, 0);
            to.strideW = getScalar<int32_t>(opts, DepthwiseConv2DOptionsField::StrideW, 1);
            to.strideH = getScalar<int32_t>(opts, DepthwiseConv2DOptionsField::StrideH, 1);
            to.depthMultiplier = getScalar<int32_t>(opts, DepthwiseConv2DOptionsField::DepthMultiplier, 1);
            to.fusedActivation = getScalar<int8_t>(opts, DepthwiseConv2DOptionsField::FusedActivation, 0);
            to.dilationW = getScalar<int32_t>(opts, DepthwiseConv2DOptionsField::DilationW, 1);
            to.dilationH = getScalar<int32_t>(opts, DepthwiseConv2DOptionsField::DilationH, 1);
            break;
        case TFL_BATCH_MATMUL:
            to.adjX = getScalar<bool>(opts, BatchMatMulOptionsField::AdjX, false);
            to.adjY = getScalar<bool>(opts, BatchMatMulOptionsField::AdjY, false);
            break;
        case TFL_RESHAPE:
        {
            auto newShapeVec = getVector<int32_t>(opts, ReshapeOptionsField::NewShape);
            if (newShapeVec)
                for (fb::uoffset_t i = 0; i < newShapeVec->size(); i++)
                    to.newShape.push_back(newShapeVec->Get(i));
            break;
        }
        case TFL_FULLY_CONNECTED:
        case TFL_ADD:
        case TFL_MUL:
            to.fusedActivation = getScalar<int8_t>(opts, ActivationOnlyOptionsField::FusedActivation, 0);
            break;
        case TFL_SOFTMAX:
            to.softmaxBeta = getScalar<float>(opts, SoftmaxOptionsField::Beta, 1.0f);
            break;
        case TFL_CONCATENATION:
            to.axis = getScalar<int32_t>(opts, ConcatenationOptionsField::Axis, 0);
            to.fusedActivation = getScalar<int8_t>(opts, ConcatenationOptionsField::FusedActivation, 0);
            break;
        case TFL_GATHER:
            to.axis = getScalar<int32_t>(opts, GatherOptionsField::Axis, 0);
            to.batchDims = getScalar<int32_t>(opts, GatherOptionsField::BatchDims, 0);
            break;
        case TFL_RESIZE_BILINEAR:
        case TFL_RESIZE_NEAREST_NEIGHBOR:
            to.alignCorners = getScalar<bool>(opts, ResizeOptionsField::AlignCorners, false);
            to.halfPixelCenters = getScalar<bool>(opts, ResizeOptionsField::HalfPixelCenters, false);
            break;
        default:
            break; // RELU/LOGISTIC/TRANSPOSE/QUANTIZE/DEQUANTIZE have no relevant options
        }
    }
} // namespace

DataType systems::leal::campello_nn::tflite::tfliteTensorTypeToDataType(int8_t tensorType)
{
    switch (tensorType)
    {
    case TFL_FLOAT32:
        return DataType::Float32;
    case TFL_FLOAT16:
        return DataType::Float16;
    case TFL_INT32:
        return DataType::Int32;
    case TFL_INT8:
        return DataType::Int8;
    case TFL_UINT32:
        return DataType::Uint32;
    default:
        throw std::runtime_error("campello_nn: TFLite TensorType " + std::to_string(tensorType) +
                                  " has no campello_nn::DataType equivalent");
    }
}

bool systems::leal::campello_nn::tflite::tfliteTensorTypeHasDataType(int8_t tensorType)
{
    switch (tensorType)
    {
    case TFL_FLOAT32:
    case TFL_FLOAT16:
    case TFL_INT32:
    case TFL_INT8:
    case TFL_UINT32:
        return true;
    default:
        return false;
    }
}

DataType systems::leal::campello_nn::tflite::TfliteTensor::toDataType() const
{
    return tfliteTensorTypeToDataType(type);
}

// Model: version=0, operator_codes=1, subgraphs=2, buffers=4 (description=3,
// metadata*/signature_defs/external_buffer* not read). Only subgraph 0 is
// imported — see TfliteGraph's comment.
tflite::TfliteGraph systems::leal::campello_nn::tflite::parseTfliteModel(const std::vector<uint8_t> &fileBytes)
{
    if (fileBytes.size() < 8)
        throw std::runtime_error("campello_nn: TFLite file too small to be valid");
    const uint8_t *buf = fileBytes.data();
    if (!fb::BufferHasIdentifier(buf, "TFL3"))
        throw std::runtime_error("campello_nn: not a TFLite file (missing 'TFL3' file identifier)");

    // NOTE: deliberately skips flatbuffers::Verifier structural validation — it
    // requires flatc-generated per-table ::Verify methods this parser doesn't
    // have, since it avoids the flatc codegen step entirely (see
    // CMakeLists.txt's flatbuffers FetchContent comment). A malformed file
    // could cause out-of-bounds reads; this importer assumes a well-formed,
    // trusted .tflite file, the same trust boundary flatbuffers' own generated
    // accessor code has without an explicit Verifier pass.
    const fb::Table *model = fb::GetRoot<fb::Table>(buf);

    auto subgraphsVec = getTableVector(model, ModelField::Subgraphs);
    if (!subgraphsVec || subgraphsVec->size() == 0)
        throw std::runtime_error("campello_nn: TFLite model has no subgraphs");
    const fb::Table *subgraph = subgraphsVec->Get(0);

    auto operatorCodesVec = getTableVector(model, ModelField::OperatorCodes);
    auto buffersVec = getTableVector(model, ModelField::Buffers);

    TfliteGraph g;

    if (buffersVec)
    {
        g.buffers.resize(buffersVec->size());
        for (fb::uoffset_t i = 0; i < buffersVec->size(); i++)
        {
            auto dataVec = getVector<uint8_t>(buffersVec->Get(i), BufferField::Data);
            if (dataVec)
                g.buffers[i].data.assign(dataVec->data(), dataVec->data() + dataVec->size());
        }
    }

    auto tensorsVec = getTableVector(subgraph, SubGraphField::Tensors);
    if (tensorsVec)
    {
        g.tensors.resize(tensorsVec->size());
        for (fb::uoffset_t i = 0; i < tensorsVec->size(); i++)
        {
            const fb::Table *t = tensorsVec->Get(i);
            TfliteTensor &tt = g.tensors[i];

            auto shapeVec = getVector<int32_t>(t, TensorField::Shape);
            if (shapeVec)
                for (fb::uoffset_t d = 0; d < shapeVec->size(); d++)
                    tt.shape.push_back(shapeVec->Get(d));

            tt.type = getScalar<int8_t>(t, TensorField::Type, 0);
            tt.bufferIndex = getScalar<uint32_t>(t, TensorField::Buffer, 0);

            auto nameStr = getStringField(t, TensorField::Name);
            if (nameStr)
                tt.name = nameStr->str();

            const fb::Table *quant = getSubTable(t, TensorField::Quantization);
            if (quant)
            {
                auto scaleVec = getVector<float>(quant, QuantParamField::Scale);
                if (scaleVec && scaleVec->size() > 0)
                {
                    tt.hasQuantization = true;
                    tt.quantScale = scaleVec->Get(0);
                    auto zpVec = getVector<int64_t>(quant, QuantParamField::ZeroPoint);
                    tt.quantZeroPoint = (zpVec && zpVec->size() > 0) ? (int32_t)zpVec->Get(0) : 0;
                }
            }
        }
    }

    // Resolve each OperatorCode's effective builtin op value up front, indexed
    // the same way Operator.opcode_index references it.
    std::vector<int32_t> resolvedOpCodes;
    if (operatorCodesVec)
    {
        resolvedOpCodes.resize(operatorCodesVec->size());
        for (fb::uoffset_t i = 0; i < operatorCodesVec->size(); i++)
        {
            const fb::Table *oc = operatorCodesVec->Get(i);
            int8_t deprecated = getScalar<int8_t>(oc, OperatorCodeField::DeprecatedBuiltinCode, 0);
            int32_t modern = getScalar<int32_t>(oc, OperatorCodeField::BuiltinCode, 0);
            // Mirrors TFLite's own GetBuiltinCode() (schema_utils.cc): take
            // whichever field is larger. Modern exporters populate the wider
            // `builtin_code` field correctly for every op; only very old files
            // rely solely on the single deprecated byte (range 0-127).
            resolvedOpCodes[i] = std::max((int32_t)deprecated, modern);
        }
    }

    auto operatorsVec = getTableVector(subgraph, SubGraphField::Operators);
    if (operatorsVec)
    {
        g.operators.resize(operatorsVec->size());
        for (fb::uoffset_t i = 0; i < operatorsVec->size(); i++)
        {
            const fb::Table *op = operatorsVec->Get(i);
            TfliteOperator &to = g.operators[i];

            uint32_t opcodeIndex = getScalar<uint32_t>(op, OperatorField::OpcodeIndex, 0);
            to.builtinCode = opcodeIndex < resolvedOpCodes.size() ? resolvedOpCodes[opcodeIndex] : -1;

            auto opInputsVec = getVector<int32_t>(op, OperatorField::Inputs);
            if (opInputsVec)
                for (fb::uoffset_t k = 0; k < opInputsVec->size(); k++)
                    to.inputs.push_back(opInputsVec->Get(k));

            auto opOutputsVec = getVector<int32_t>(op, OperatorField::Outputs);
            if (opOutputsVec)
                for (fb::uoffset_t k = 0; k < opOutputsVec->size(); k++)
                    to.outputs.push_back(opOutputsVec->Get(k));

            const fb::Table *opts = getSubTable(op, OperatorField::BuiltinOptions);
            if (opts)
                parseOptionsForOp(opts, to);
        }
    }

    auto graphInputsVec = getVector<int32_t>(subgraph, SubGraphField::Inputs);
    if (graphInputsVec)
        for (fb::uoffset_t i = 0; i < graphInputsVec->size(); i++)
            g.graphInputs.push_back(graphInputsVec->Get(i));

    auto graphOutputsVec = getVector<int32_t>(subgraph, SubGraphField::Outputs);
    if (graphOutputsVec)
        for (fb::uoffset_t i = 0; i < graphOutputsVec->size(); i++)
            g.graphOutputs.push_back(graphOutputsVec->Get(i));

    return g;
}

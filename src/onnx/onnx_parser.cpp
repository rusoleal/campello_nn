#include <cstring>
#include <stdexcept>
#include "onnx_model.hpp"
#include "proto_reader.hpp"

using namespace systems::leal::campello_nn;
using namespace systems::leal::campello_nn::onnx;

namespace
{
    // ONNX TensorProto.DataType enum values (stable since ONNX v1; see onnx.proto).
    enum OnnxElemType : int32_t
    {
        ONNX_FLOAT = 1,
        ONNX_UINT8 = 2,
        ONNX_INT8 = 3,
        ONNX_UINT16 = 4,
        ONNX_INT16 = 5,
        ONNX_INT32 = 6,
        ONNX_INT64 = 7,
        ONNX_STRING = 8,
        ONNX_BOOL = 9,
        ONNX_FLOAT16 = 10,
        ONNX_DOUBLE = 11,
        ONNX_UINT32 = 12,
        ONNX_UINT64 = 13,
    };

    // Real ONNX exporters were observed (against onnx==1.19.1's own output) to use
    // the *unpacked* form (one tag+value per element) for repeated scalar fields,
    // but the protobuf spec also allows the *packed* form (a single length-delimited
    // run of concatenated values) — handle both.
    void appendVarints(ProtoReader &r, WireType wt, std::vector<int64_t> &out)
    {
        if (wt == WireType::Varint)
        {
            out.push_back((int64_t)r.readVarint());
        }
        else if (wt == WireType::LengthDelimited)
        {
            ProtoReader pr(r.readLengthDelimited());
            while (!pr.atEnd())
                out.push_back((int64_t)pr.readVarint());
        }
        else
        {
            r.skip(wt);
        }
    }

    void appendFloats(ProtoReader &r, WireType wt, std::vector<float> &out)
    {
        if (wt == WireType::Fixed32)
        {
            out.push_back(r.readFloat());
        }
        else if (wt == WireType::LengthDelimited)
        {
            ProtoReader pr(r.readLengthDelimited());
            while (!pr.atEnd())
                out.push_back(pr.readFloat());
        }
        else
        {
            r.skip(wt);
        }
    }

    // TensorProto: dims=1, data_type=2, float_data=4, int32_data=5, int64_data=7,
    // name=8, raw_data=9, double_data=10. Returns {name, tensor}.
    std::pair<std::string, OnnxTensor> parseTensorProto(std::string_view bytes)
    {
        OnnxTensor t;
        std::string name;
        std::vector<float> floatData;
        std::vector<int64_t> intData; // int32_data and int64_data both land here, widened

        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        while (r.nextField(fn, wt))
        {
            switch (fn)
            {
            case 1:
                appendVarints(r, wt, t.shape);
                break;
            case 2:
                t.elemType = (int32_t)r.readVarint();
                break;
            case 4:
                appendFloats(r, wt, floatData);
                break;
            case 5:
            case 7:
                appendVarints(r, wt, intData);
                break;
            case 8:
                name = std::string(r.readLengthDelimited());
                break;
            case 9:
            {
                auto raw = r.readLengthDelimited();
                t.bytes.assign(raw.begin(), raw.end());
                break;
            }
            default:
                r.skip(wt);
                break;
            }
        }

        // Some exporters use the typed-array fields instead of raw_data for small
        // tensors; reconstruct raw bytes from whichever one is actually populated.
        if (t.bytes.empty() && !floatData.empty())
        {
            t.bytes.resize(floatData.size() * sizeof(float));
            std::memcpy(t.bytes.data(), floatData.data(), t.bytes.size());
        }
        else if (t.bytes.empty() && !intData.empty())
        {
            if (t.elemType == ONNX_INT64)
            {
                t.bytes.resize(intData.size() * sizeof(int64_t));
                std::memcpy(t.bytes.data(), intData.data(), t.bytes.size());
            }
            else
            {
                std::vector<int32_t> narrowed(intData.begin(), intData.end());
                t.bytes.resize(narrowed.size() * sizeof(int32_t));
                std::memcpy(t.bytes.data(), narrowed.data(), t.bytes.size());
            }
        }

        return {name, t};
    }

    // AttributeProto: name=1, f=2, i=3, s=4, t=5, floats=7, ints=8, type=20 (ignored
    // — see OnnxAttribute's comment: consumers read the slot they expect by name).
    OnnxAttribute parseAttributeProto(std::string_view bytes)
    {
        OnnxAttribute a;
        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        while (r.nextField(fn, wt))
        {
            switch (fn)
            {
            case 1:
                a.name = std::string(r.readLengthDelimited());
                break;
            case 2:
                a.f = r.readFloat();
                break;
            case 3:
                a.i = (int64_t)r.readVarint();
                break;
            case 4:
                a.s = std::string(r.readLengthDelimited());
                break;
            case 5:
            {
                auto [tensorName, tensor] = parseTensorProto(r.readLengthDelimited());
                (void)tensorName;
                a.tensor = tensor;
                a.hasTensor = true;
                break;
            }
            case 7:
                appendFloats(r, wt, a.floats);
                break;
            case 8:
                appendVarints(r, wt, a.ints);
                break;
            default:
                r.skip(wt);
                break;
            }
        }
        return a;
    }

    // NodeProto: input=1, output=2, name=3, op_type=4, attribute=5, domain=7.
    OnnxNode parseNodeProto(std::string_view bytes)
    {
        OnnxNode n;
        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        while (r.nextField(fn, wt))
        {
            switch (fn)
            {
            case 1:
                n.inputs.push_back(std::string(r.readLengthDelimited()));
                break;
            case 2:
                n.outputs.push_back(std::string(r.readLengthDelimited()));
                break;
            case 3:
                n.name = std::string(r.readLengthDelimited());
                break;
            case 4:
                n.opType = std::string(r.readLengthDelimited());
                break;
            case 5:
            {
                OnnxAttribute attr = parseAttributeProto(r.readLengthDelimited());
                n.attributes[attr.name] = std::move(attr);
                break;
            }
            default:
                r.skip(wt);
                break;
            }
        }
        return n;
    }

    // Dimension: dim_value=1 (varint, oneof) | dim_param=2 (string, oneof — a
    // symbolic/dynamic dim, e.g. a batch axis). campello_nn needs static shapes, so
    // a dynamic dimension defaults to 1 (single-item inference), the practical
    // choice for "run this model on one image".
    int64_t parseDimension(std::string_view bytes)
    {
        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        int64_t value = 1;
        bool sawDimValue = false;
        while (r.nextField(fn, wt))
        {
            if (fn == 1 && wt == WireType::Varint)
            {
                value = (int64_t)r.readVarint();
                sawDimValue = true;
            }
            else
            {
                r.skip(wt);
            }
        }
        return sawDimValue ? value : 1;
    }

    // TensorShapeProto: dim=1 (repeated Dimension).
    std::vector<int64_t> parseTensorShapeProto(std::string_view bytes)
    {
        std::vector<int64_t> shape;
        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        while (r.nextField(fn, wt))
        {
            if (fn == 1 && wt == WireType::LengthDelimited)
                shape.push_back(parseDimension(r.readLengthDelimited()));
            else
                r.skip(wt);
        }
        return shape;
    }

    // TypeProto.Tensor: elem_type=1 (varint), shape=2 (TensorShapeProto).
    // TypeProto: tensor_type=1 (the Tensor message above; only oneof case handled).
    void parseTypeProto(std::string_view bytes, int32_t &elemType, std::vector<int64_t> &shape)
    {
        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        while (r.nextField(fn, wt))
        {
            if (fn == 1 && wt == WireType::LengthDelimited)
            {
                ProtoReader tr(r.readLengthDelimited());
                uint32_t tfn;
                WireType twt;
                while (tr.nextField(tfn, twt))
                {
                    if (tfn == 1 && twt == WireType::Varint)
                        elemType = (int32_t)tr.readVarint();
                    else if (tfn == 2 && twt == WireType::LengthDelimited)
                        shape = parseTensorShapeProto(tr.readLengthDelimited());
                    else
                        tr.skip(twt);
                }
            }
            else
            {
                r.skip(wt);
            }
        }
    }

    // ValueInfoProto: name=1 (string), type=2 (TypeProto).
    OnnxValueInfo parseValueInfoProto(std::string_view bytes)
    {
        OnnxValueInfo v;
        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        while (r.nextField(fn, wt))
        {
            if (fn == 1 && wt == WireType::LengthDelimited)
                v.name = std::string(r.readLengthDelimited());
            else if (fn == 2 && wt == WireType::LengthDelimited)
                parseTypeProto(r.readLengthDelimited(), v.elemType, v.shape);
            else
                r.skip(wt);
        }
        return v;
    }

    // GraphProto: node=1, name=2, initializer=5, input=11, output=12, value_info=13.
    OnnxGraph parseGraphProto(std::string_view bytes)
    {
        OnnxGraph g;
        ProtoReader r(bytes);
        uint32_t fn;
        WireType wt;
        while (r.nextField(fn, wt))
        {
            switch (fn)
            {
            case 1:
                g.nodes.push_back(parseNodeProto(r.readLengthDelimited()));
                break;
            case 5:
            {
                auto [name, tensor] = parseTensorProto(r.readLengthDelimited());
                g.initializers[name] = std::move(tensor);
                break;
            }
            case 11:
                g.inputs.push_back(parseValueInfoProto(r.readLengthDelimited()));
                break;
            case 12:
                g.outputs.push_back(parseValueInfoProto(r.readLengthDelimited()));
                break;
            default:
                r.skip(wt);
                break;
            }
        }
        return g;
    }
}

DataType systems::leal::campello_nn::onnx::onnxElemTypeToDataType(int32_t elemType)
{
    switch (elemType)
    {
    case ONNX_FLOAT:
        return DataType::Float32;
    case ONNX_FLOAT16:
        return DataType::Float16;
    case ONNX_INT32:
        return DataType::Int32;
    case ONNX_UINT32:
        return DataType::Uint32;
    case ONNX_INT8:
        return DataType::Int8;
    default:
        throw std::runtime_error(
            "campello_nn: ONNX elem_type " + std::to_string(elemType) +
            " has no campello_nn::DataType equivalent");
    }
}

bool systems::leal::campello_nn::onnx::onnxElemTypeHasDataType(int32_t elemType)
{
    switch (elemType)
    {
    case ONNX_FLOAT:
    case ONNX_FLOAT16:
    case ONNX_INT32:
    case ONNX_UINT32:
    case ONNX_INT8:
        return true;
    default:
        return false;
    }
}

DataType OnnxTensor::toDataType() const
{
    return onnxElemTypeToDataType(elemType);
}

std::vector<int64_t> OnnxTensor::toInt64Vector() const
{
    std::vector<int64_t> out;
    if (elemType == ONNX_INT64)
    {
        size_t n = bytes.size() / sizeof(int64_t);
        out.resize(n);
        std::memcpy(out.data(), bytes.data(), n * sizeof(int64_t));
    }
    else if (elemType == ONNX_INT32 || elemType == ONNX_UINT32)
    {
        size_t n = bytes.size() / sizeof(int32_t);
        const int32_t *p = (const int32_t *)bytes.data();
        out.assign(p, p + n);
    }
    else if (elemType == ONNX_FLOAT)
    {
        size_t n = bytes.size() / sizeof(float);
        const float *p = (const float *)bytes.data();
        out.resize(n);
        for (size_t i = 0; i < n; i++)
            out[i] = (int64_t)p[i];
    }
    else
    {
        throw std::runtime_error("campello_nn: cannot interpret ONNX elem_type " +
                                  std::to_string(elemType) + " as an integer shape/index list");
    }
    return out;
}

// ModelProto: ir_version=1, producer_name=2, graph=7.
OnnxGraph systems::leal::campello_nn::onnx::parseOnnxModel(const std::vector<uint8_t> &fileBytes)
{
    ProtoReader r(fileBytes.data(), fileBytes.size());
    uint32_t fn;
    WireType wt;
    while (r.nextField(fn, wt))
    {
        if (fn == 7 && wt == WireType::LengthDelimited)
            return parseGraphProto(r.readLengthDelimited());
        r.skip(wt);
    }
    throw std::runtime_error("campello_nn: ONNX file has no graph (ModelProto.graph)");
}

#include "ir_serialization.hpp"
#include <cstring>
#include <stdexcept>

using namespace systems::leal::campello_nn;

namespace
{
    constexpr uint32_t kMagic = 0x474E4E43; // "CNNG" little-endian
    constexpr uint32_t kVersion = 1;

    class Writer
    {
    public:
        std::vector<uint8_t> bytes;

        template <typename T>
        void writePod(const T &v)
        {
            static_assert(std::is_trivially_copyable<T>::value, "writePod requires a trivially copyable type");
            const uint8_t *p = (const uint8_t *)&v;
            bytes.insert(bytes.end(), p, p + sizeof(T));
        }

        void writeBytes(const void *data, size_t size)
        {
            writePod<uint64_t>((uint64_t)size);
            const uint8_t *p = (const uint8_t *)data;
            bytes.insert(bytes.end(), p, p + size);
        }

        void writeString(const std::string &s)
        {
            writePod<uint32_t>((uint32_t)s.size());
            bytes.insert(bytes.end(), s.begin(), s.end());
        }

        void writeInt64Vector(const std::vector<int64_t> &v)
        {
            writePod<uint32_t>((uint32_t)v.size());
            for (int64_t x : v)
                writePod<int64_t>(x);
        }

        void writeSizeVector(const std::vector<size_t> &v)
        {
            writePod<uint32_t>((uint32_t)v.size());
            for (size_t x : v)
                writePod<uint32_t>((uint32_t)x);
        }
    };

    class Reader
    {
    public:
        const uint8_t *data;
        size_t size;
        size_t offset = 0;

        Reader(const uint8_t *d, size_t s) : data(d), size(s) {}

        void requireAvailable(size_t n) const
        {
            if (offset + n > size)
                throw std::runtime_error("campello_nn: deserializeGraphIR() truncated buffer");
        }

        template <typename T>
        T readPod()
        {
            static_assert(std::is_trivially_copyable<T>::value, "readPod requires a trivially copyable type");
            requireAvailable(sizeof(T));
            T v;
            std::memcpy(&v, data + offset, sizeof(T));
            offset += sizeof(T);
            return v;
        }

        std::vector<uint8_t> readBytes()
        {
            uint64_t n = readPod<uint64_t>();
            requireAvailable((size_t)n);
            std::vector<uint8_t> v(data + offset, data + offset + n);
            offset += (size_t)n;
            return v;
        }

        std::string readString()
        {
            uint32_t n = readPod<uint32_t>();
            requireAvailable(n);
            std::string s((const char *)(data + offset), n);
            offset += n;
            return s;
        }

        std::vector<int64_t> readInt64Vector()
        {
            uint32_t n = readPod<uint32_t>();
            std::vector<int64_t> v(n);
            for (uint32_t i = 0; i < n; i++)
                v[i] = readPod<int64_t>();
            return v;
        }

        std::vector<size_t> readSizeVector()
        {
            uint32_t n = readPod<uint32_t>();
            std::vector<size_t> v(n);
            for (uint32_t i = 0; i < n; i++)
                v[i] = (size_t)readPod<uint32_t>();
            return v;
        }
    };

    void writeNode(Writer &w, const Node &node)
    {
        w.writePod<uint8_t>((uint8_t)node.kind);
        w.writeSizeVector(node.inputs);
        w.writePod<uint8_t>((uint8_t)node.dataType);
        w.writeInt64Vector(node.shape);
        w.writeString(node.name);
        w.writeBytes(node.constantBytes.data(), node.constantBytes.size());
        w.writeInt64Vector(node.intAttr0);
        w.writeInt64Vector(node.intAttr1);
        w.writePod<int32_t>(node.axis);
        w.writePod<float>(node.floatAttr0);
        w.writePod<float>(node.floatAttr1);

        const Conv2dDescriptor &c = node.convParams;
        w.writePod<int64_t>(c.strideX);
        w.writePod<int64_t>(c.strideY);
        w.writePod<int64_t>(c.dilationX);
        w.writePod<int64_t>(c.dilationY);
        w.writePod<int64_t>(c.paddingLeft);
        w.writePod<int64_t>(c.paddingRight);
        w.writePod<int64_t>(c.paddingTop);
        w.writePod<int64_t>(c.paddingBottom);
        w.writePod<int64_t>(c.groups);

        const Pool2dDescriptor &p = node.poolParams;
        w.writePod<int64_t>(p.kernelHeight);
        w.writePod<int64_t>(p.kernelWidth);
        w.writePod<int64_t>(p.strideX);
        w.writePod<int64_t>(p.strideY);
        w.writePod<int64_t>(p.paddingLeft);
        w.writePod<int64_t>(p.paddingRight);
        w.writePod<int64_t>(p.paddingTop);
        w.writePod<int64_t>(p.paddingBottom);

        const ResizeDescriptor &r = node.resizeParams;
        w.writePod<int64_t>(r.outputHeight);
        w.writePod<int64_t>(r.outputWidth);
        w.writePod<uint8_t>((uint8_t)r.mode);
        w.writePod<uint8_t>((uint8_t)r.centerResult);
        w.writePod<uint8_t>((uint8_t)r.alignCorners);
        w.writePod<uint8_t>((uint8_t)r.nearestRoundsDown);
    }

    Node readNode(Reader &r)
    {
        Node node;
        node.kind = (OpKind)r.readPod<uint8_t>();
        node.inputs = r.readSizeVector();
        node.dataType = (DataType)r.readPod<uint8_t>();
        node.shape = r.readInt64Vector();
        node.name = r.readString();
        node.constantBytes = r.readBytes();
        node.intAttr0 = r.readInt64Vector();
        node.intAttr1 = r.readInt64Vector();
        node.axis = r.readPod<int32_t>();
        node.floatAttr0 = r.readPod<float>();
        node.floatAttr1 = r.readPod<float>();

        Conv2dDescriptor &c = node.convParams;
        c.strideX = r.readPod<int64_t>();
        c.strideY = r.readPod<int64_t>();
        c.dilationX = r.readPod<int64_t>();
        c.dilationY = r.readPod<int64_t>();
        c.paddingLeft = r.readPod<int64_t>();
        c.paddingRight = r.readPod<int64_t>();
        c.paddingTop = r.readPod<int64_t>();
        c.paddingBottom = r.readPod<int64_t>();
        c.groups = r.readPod<int64_t>();

        Pool2dDescriptor &p = node.poolParams;
        p.kernelHeight = r.readPod<int64_t>();
        p.kernelWidth = r.readPod<int64_t>();
        p.strideX = r.readPod<int64_t>();
        p.strideY = r.readPod<int64_t>();
        p.paddingLeft = r.readPod<int64_t>();
        p.paddingRight = r.readPod<int64_t>();
        p.paddingTop = r.readPod<int64_t>();
        p.paddingBottom = r.readPod<int64_t>();

        ResizeDescriptor &res = node.resizeParams;
        res.outputHeight = r.readPod<int64_t>();
        res.outputWidth = r.readPod<int64_t>();
        res.mode = (ResizeMode)r.readPod<uint8_t>();
        res.centerResult = (bool)r.readPod<uint8_t>();
        res.alignCorners = (bool)r.readPod<uint8_t>();
        res.nearestRoundsDown = (bool)r.readPod<uint8_t>();

        return node;
    }
}

std::vector<uint8_t> systems::leal::campello_nn::serializeGraphIR(const GraphIR &ir)
{
    Writer w;
    w.writePod<uint32_t>(kMagic);
    w.writePod<uint32_t>(kVersion);

    w.writePod<uint32_t>((uint32_t)ir.nodes.size());
    for (const Node &node : ir.nodes)
        writeNode(w, node);

    w.writePod<uint32_t>((uint32_t)ir.outputs.size());
    for (const auto &[name, nodeIndex] : ir.outputs)
    {
        w.writeString(name);
        w.writePod<uint32_t>((uint32_t)nodeIndex);
    }

    return std::move(w.bytes);
}

GraphIR systems::leal::campello_nn::deserializeGraphIR(const uint8_t *data, size_t size)
{
    Reader r(data, size);
    uint32_t magic = r.readPod<uint32_t>();
    if (magic != kMagic)
        throw std::runtime_error("campello_nn: deserializeGraphIR() bad magic — not a serialized campello_nn graph");
    uint32_t version = r.readPod<uint32_t>();
    if (version != kVersion)
        throw std::runtime_error("campello_nn: deserializeGraphIR() unsupported version " + std::to_string(version));

    GraphIR ir;
    uint32_t nodeCount = r.readPod<uint32_t>();
    ir.nodes.reserve(nodeCount);
    for (uint32_t i = 0; i < nodeCount; i++)
        ir.nodes.push_back(readNode(r));

    uint32_t outputCount = r.readPod<uint32_t>();
    ir.outputs.reserve(outputCount);
    for (uint32_t i = 0; i < outputCount; i++)
    {
        std::string name = r.readString();
        size_t nodeIndex = (size_t)r.readPod<uint32_t>();
        ir.outputs.push_back({name, nodeIndex});
    }

    return ir;
}

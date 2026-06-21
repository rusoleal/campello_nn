#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace systems::leal::campello_nn::onnx
{

    /**
     * @brief Minimal protobuf wire-format reader, scoped to reading only.
     *
     * ONNX's `.onnx` files are protobuf-encoded. Generating/validating against an
     * arbitrary `.proto` schema needs the full protobuf library + `protoc`; reading a
     * known, stable schema (ONNX's field numbers haven't changed since v1, for wire
     * compatibility) does not — it's just varints, tags, and length-delimited byte
     * ranges. This avoids a heavy build-time dependency for a project that has stayed
     * dependency-light so far.
     *
     * Field numbers/wire types used by callers are documented next to where they're
     * consumed (see `onnx_parser.cpp`), not here — this class only knows the wire
     * format, not ONNX's schema.
     */
    enum class WireType : uint32_t
    {
        Varint = 0,
        Fixed64 = 1,
        LengthDelimited = 2,
        Fixed32 = 5
    };

    class ProtoReader
    {
    public:
        ProtoReader(const uint8_t *data, size_t size) : data_(data), size_(size), pos_(0) {}
        explicit ProtoReader(std::string_view sv) : data_((const uint8_t *)sv.data()), size_(sv.size()), pos_(0) {}

        bool atEnd() const { return pos_ >= size_; }

        /// Reads the next field's tag. Returns false once the buffer is exhausted.
        bool nextField(uint32_t &fieldNumber, WireType &wireType)
        {
            if (atEnd())
                return false;
            uint64_t tag = readVarint();
            fieldNumber = (uint32_t)(tag >> 3);
            wireType = (WireType)(tag & 0x7u);
            return true;
        }

        uint64_t readVarint()
        {
            uint64_t result = 0;
            int shift = 0;
            while (true)
            {
                if (pos_ >= size_)
                    throw std::runtime_error("campello_nn: truncated protobuf varint");
                uint8_t b = data_[pos_++];
                result |= (uint64_t)(b & 0x7Fu) << shift;
                if ((b & 0x80u) == 0)
                    break;
                shift += 7;
            }
            return result;
        }

        std::string_view readLengthDelimited()
        {
            uint64_t len = readVarint();
            if (pos_ + len > size_)
                throw std::runtime_error("campello_nn: truncated protobuf length-delimited field");
            std::string_view sv((const char *)(data_ + pos_), (size_t)len);
            pos_ += (size_t)len;
            return sv;
        }

        uint32_t readFixed32()
        {
            if (pos_ + 4 > size_)
                throw std::runtime_error("campello_nn: truncated protobuf fixed32");
            uint32_t v;
            std::memcpy(&v, data_ + pos_, 4);
            pos_ += 4;
            return v;
        }

        uint64_t readFixed64()
        {
            if (pos_ + 8 > size_)
                throw std::runtime_error("campello_nn: truncated protobuf fixed64");
            uint64_t v;
            std::memcpy(&v, data_ + pos_, 8);
            pos_ += 8;
            return v;
        }

        float readFloat()
        {
            uint32_t v = readFixed32();
            float f;
            std::memcpy(&f, &v, 4);
            return f;
        }

        double readDouble()
        {
            uint64_t v = readFixed64();
            double d;
            std::memcpy(&d, &v, 8);
            return d;
        }

        /// Skips the current field's value when the caller doesn't care about it.
        void skip(WireType wireType)
        {
            switch (wireType)
            {
            case WireType::Varint:
                readVarint();
                break;
            case WireType::Fixed64:
                readFixed64();
                break;
            case WireType::LengthDelimited:
                readLengthDelimited();
                break;
            case WireType::Fixed32:
                readFixed32();
                break;
            }
        }

    private:
        const uint8_t *data_;
        size_t size_;
        size_t pos_;
    };

} // namespace systems::leal::campello_nn::onnx

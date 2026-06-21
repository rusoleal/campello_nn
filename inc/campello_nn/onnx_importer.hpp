#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <campello_nn/context.hpp>
#include <campello_nn/graph.hpp>
#include <campello_nn/descriptors/tensor_descriptor.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Result of `importOnnx()` — the compiled `Graph` plus what's needed to
     * actually use it: the input/output `Tensor` shapes/dtypes the caller must
     * create to match, by name.
     */
    struct OnnxImportResult
    {
        std::shared_ptr<Graph> graph;
        std::unordered_map<std::string, TensorDescriptor> inputs;
        std::unordered_map<std::string, TensorDescriptor> outputs;
    };

    /**
     * @brief Imports an ONNX model from a memory buffer directly into a compiled `Graph`.
     *
     * The primary entry point — `importOnnxFromFile()` is a convenience wrapper
     * around this. Takes bytes rather than assuming a filesystem path so callers
     * can supply model data from anywhere: Android's `AAssetManager` (APK assets
     * aren't real filesystem paths), an iOS bundle, a network download, a
     * decrypted blob, etc. Mirrors `campello_image::Image::fromFile`/`fromMemory`'s
     * split for the same reason.
     *
     * ONNX files carry full graph topology alongside their weights, so this is a
     * 1:1 translation from the file's own node list into `GraphBuilder` calls — no
     * architecture-specific knowledge needed (see `NN_ARCHITECTURE.md` §3/§5).
     * Op coverage is limited to whatever maps onto `GraphBuilder`'s op set; an
     * unsupported ONNX op throws `std::runtime_error` naming the op type.
     *
     * @param context Backend to compile the imported graph against.
     * @param data Pointer to the encoded `.onnx` model bytes.
     * @param size Number of bytes at `data`.
     */
    OnnxImportResult importOnnxFromMemory(std::shared_ptr<Context> context, const uint8_t *data, size_t size);

    /**
     * @brief Imports an ONNX model file directly into a compiled `Graph`.
     *
     * Convenience wrapper: reads `path` into memory and calls `importOnnxFromMemory()`.
     * Throws `std::runtime_error` if the file cannot be opened.
     *
     * @param context Backend to compile the imported graph against.
     * @param path Filesystem path to a `.onnx` file.
     */
    OnnxImportResult importOnnxFromFile(std::shared_ptr<Context> context, const std::string &path);

} // namespace systems::leal::campello_nn

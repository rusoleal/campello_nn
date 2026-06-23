#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <campello_nn/context.hpp>
#include <campello_nn/graph.hpp>
#include <campello_nn/graph_info.hpp>
#include <campello_nn/descriptors/tensor_descriptor.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Result of `importTflite*()` — the compiled `Graph` plus what's needed
     * to actually use it: the input/output `Tensor` shapes/dtypes the caller must
     * create to match, by name.
     *
     * Shapes here are TFLite's own declared shapes (NHWC for 4D tensors) — the
     * importer internally converts to/from campello_nn's NCHW convention at the
     * graph boundary, but the descriptors returned here match what the original
     * `.tflite` file itself declares, so callers don't need to know that
     * conversion happened.
     */
    struct TfliteImportResult
    {
        std::shared_ptr<Graph> graph;
        std::unordered_map<std::string, TensorDescriptor> inputs;
        std::unordered_map<std::string, TensorDescriptor> outputs;
        // Topology of the imported graph, in campello_nn's own NCHW convention (post-
        // conversion) — for inspection/visualization, not execution. See `info` on
        // `OnnxImportResult` for the equivalent ONNX-side field.
        GraphInfo info;
    };

    /**
     * @brief Imports a TFLite model from a memory buffer directly into a compiled `Graph`.
     *
     * The primary entry point — `importTfliteFromFile()` is a convenience wrapper
     * around this. Takes bytes rather than assuming a filesystem path for the same
     * reason `importOnnxFromMemory()` does (Android `AAssetManager`, iOS bundles,
     * network downloads, ...) — see `onnx_importer.hpp`.
     *
     * TFLite files are FlatBuffers (not protobuf), and store activations as NHWC
     * with OHWI conv weights, unlike campello_nn's (and ONNX's) NCHW/OIHW
     * convention — see `src/tflite/tflite_importer.cpp` for the conversion
     * strategy (a single transpose at each graph input/output boundary, plus
     * pre-transposing constant weight bytes at import time so no extra runtime
     * ops are needed for those). Op coverage is limited to whatever maps onto
     * `GraphBuilder`'s op set; an unsupported TFLite op throws `std::runtime_error`
     * naming the op.
     *
     * @param context Backend to compile the imported graph against.
     * @param data Pointer to the encoded `.tflite` model bytes.
     * @param size Number of bytes at `data`.
     */
    TfliteImportResult importTfliteFromMemory(std::shared_ptr<Context> context, const uint8_t *data, size_t size);

    /**
     * @brief Imports a TFLite model file directly into a compiled `Graph`.
     *
     * Convenience wrapper: reads `path` into memory and calls `importTfliteFromMemory()`.
     * Throws `std::runtime_error` if the file cannot be opened.
     *
     * @param context Backend to compile the imported graph against.
     * @param path Filesystem path to a `.tflite` file.
     */
    TfliteImportResult importTfliteFromFile(std::shared_ptr<Context> context, const std::string &path);

} // namespace systems::leal::campello_nn

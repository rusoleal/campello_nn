#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <campello_nn/context.hpp>
#include <campello_nn/graph.hpp>
#include <campello_nn/descriptors/tensor_descriptor.hpp>

namespace systems::leal::campello_nn
{

    /**
     * @brief Result of `loadGraphFromMemory()`/`loadGraphFromFile()` — the compiled
     * `Graph` plus the input/output `Tensor` shapes/dtypes needed to use it, by name.
     * Mirrors `OnnxImportResult`'s shape.
     */
    struct GraphCacheResult
    {
        std::shared_ptr<Graph> graph;
        std::unordered_map<std::string, TensorDescriptor> inputs;
        std::unordered_map<std::string, TensorDescriptor> outputs;
    };

    /**
     * @brief Loads a graph previously produced by `GraphBuilder::serialize()` from a
     * memory buffer, compiling it directly against `context` — skipping GraphBuilder
     * reconstruction (op-by-op calls, or re-parsing a source model file such as ONNX).
     *
     * @param context Backend to compile the cached graph against.
     * @param data Pointer to the serialized graph bytes.
     * @param size Number of bytes at `data`.
     */
    GraphCacheResult loadGraphFromMemory(std::shared_ptr<Context> context, const uint8_t *data, size_t size);

    /**
     * @brief Loads a graph previously produced by `GraphBuilder::serialize()` (and
     * written to disk, e.g. via `saveGraphToFile()`) from a file.
     *
     * Convenience wrapper: reads `path` into memory and calls `loadGraphFromMemory()`.
     * Throws `std::runtime_error` if the file cannot be opened.
     */
    GraphCacheResult loadGraphFromFile(std::shared_ptr<Context> context, const std::string &path);

    /**
     * @brief Writes bytes produced by `GraphBuilder::serialize()` to a file.
     * Throws `std::runtime_error` if the file cannot be opened for writing.
     */
    void saveGraphToFile(const std::vector<uint8_t> &serializedGraph, const std::string &path);

} // namespace systems::leal::campello_nn

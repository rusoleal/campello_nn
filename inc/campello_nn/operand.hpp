#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <campello_nn/graph_info.hpp>

namespace systems::leal::campello_nn
{

    class Operand;
    class GraphBuilder;

    namespace internal
    {
        /// Returns a copy of `op`'s already-inferred shape. Internal-only — for
        /// model importers (e.g. ONNX's `Reshape`/`Resize`, which need an
        /// intermediate tensor's concrete shape at import time, not just an opaque
        /// `Operand` handle) to reuse `GraphBuilder`'s shape inference instead of
        /// duplicating it. Returns a copy, not a reference: the underlying IR node
        /// list can reallocate on subsequent `GraphBuilder` calls.
        std::vector<int64_t> operandShapeForImport(const Operand &op);

        /// Describes the graph `builder` has accumulated so far, as if `outputs` were
        /// passed to `build()` — without compiling it. Internal-only — lets model
        /// importers (ONNX/TFLite) populate `OnnxImportResult::info`/
        /// `TfliteImportResult::info` alongside the compiled `Graph`, reusing the same
        /// IR `build()` already produces instead of re-deriving it.
        GraphInfo graphInfoForImport(const GraphBuilder &builder, const std::unordered_map<std::string, Operand> &outputs);
    }

    /**
     * @brief Opaque handle to a node within a `GraphBuilder`'s in-progress graph.
     *
     * Operands are produced by `GraphBuilder` op methods (`input`, `add`, `matmul`, ...)
     * and consumed by other op methods or `GraphBuilder::build()`. An `Operand` is only
     * valid for the `GraphBuilder` that created it.
     */
    class Operand
    {
        friend class GraphBuilder;
        friend std::vector<int64_t> internal::operandShapeForImport(const Operand &op);
        friend GraphInfo internal::graphInfoForImport(const GraphBuilder &builder, const std::unordered_map<std::string, Operand> &outputs);
        void *builder;
        size_t nodeId;

        Operand(void *builder, size_t nodeId);

    public:
        Operand();
    };

} // namespace systems::leal::campello_nn

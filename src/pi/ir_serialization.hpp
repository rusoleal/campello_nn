#pragma once

#include <cstdint>
#include <vector>
#include "ir.hpp"

namespace systems::leal::campello_nn
{

    // Binary (de)serialization of the backend-agnostic GraphIR — lets a compiled
    // graph's IR be cached to disk and reloaded without re-running GraphBuilder
    // calls or re-parsing a source model file (ONNX/TFLite). Versioned via a
    // magic+version header; deserializeGraphIR() throws std::runtime_error on a
    // bad magic, an unsupported version, or a truncated buffer.
    std::vector<uint8_t> serializeGraphIR(const GraphIR &ir);
    GraphIR deserializeGraphIR(const uint8_t *data, size_t size);

} // namespace systems::leal::campello_nn

#pragma once

#include <vector>
#include "../pi/ir.hpp"
#include "cpu_value.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief Evaluates one IR node on the CPU, reading already-computed `values[node.inputs[*]]`
     * and writing the result into `values[selfIndex]`.
     *
     * Does not handle `Input`/`Constant` — those are populated directly by the caller
     * (binding the dispatch-time tensor / baked constant bytes) before this runs.
     */
    void evalNode(const Node &node, size_t selfIndex, std::vector<CpuValue> &values);

} // namespace systems::leal::campello_nn

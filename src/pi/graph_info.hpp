#pragma once

#include <campello_nn/graph_info.hpp>
#include "ir.hpp"

namespace systems::leal::campello_nn
{

    /// Converts the internal IR into the public, display-safe GraphInfo.
    GraphInfo describeGraphIR(const GraphIR &ir);

} // namespace systems::leal::campello_nn

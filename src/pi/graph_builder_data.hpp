#pragma once

#include <memory>
#include <campello_nn/context.hpp>
#include "ir.hpp"

namespace systems::leal::campello_nn
{

    struct GraphBuilderData
    {
        std::shared_ptr<Context> context;
        GraphIR ir;
    };

} // namespace systems::leal::campello_nn

#pragma once

namespace systems::leal::campello_nn
{

    /**
     * @brief Opaque compiled/optimized graph executable.
     *
     * Analogous to a compiled GPU pipeline: build once via `GraphBuilder::build()`,
     * then dispatch many times via `Context::dispatch()`.
     */
    class Graph
    {
        friend class GraphBuilder;
        friend class Context;
        void *native;

        Graph(void *pd);

    public:
        ~Graph();
    };

} // namespace systems::leal::campello_nn

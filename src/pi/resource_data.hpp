#pragma once

#include "backend.hpp"

namespace systems::leal::campello_nn
{

    /**
     * @brief Backend-tagged native handle shared by `Tensor`, `Graph`, and `Fence`.
     *
     * Storing the owning `Backend*` alongside the backend-specific native pointer
     * lets each resource's destructor/accessors call back into the right backend
     * without the public classes knowing which backend created them.
     */
    struct ResourceData
    {
        Backend *backend;
        void *native;
    };

    using TensorData = ResourceData;
    using GraphData = ResourceData;
    using FenceData = ResourceData;

} // namespace systems::leal::campello_nn

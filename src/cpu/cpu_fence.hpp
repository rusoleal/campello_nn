#pragma once

namespace systems::leal::campello_nn
{

    /**
     * @brief CPU dispatch is synchronous, so a CpuFence is always already signaled.
     */
    struct CpuFence
    {
        bool signaled = true;
    };

} // namespace systems::leal::campello_nn

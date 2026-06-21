#pragma once
#include <cstdint>

namespace systems::leal::campello_nn
{
    /**
     * @brief Binary completion fence returned by `Context::dispatch()`.
     *
     * A Fence is signaled once the backend finishes executing a dispatched
     * `Graph`. The CPU can block on it (wait) or poll (isSignaled) to know
     * when it is safe to read output `Tensor`s.
     *
     * @code
     * auto fence = context->dispatch(*graph, inputs, outputs);
     * fence->wait();   // block until the backend is done
     * outputs["logits"]->read(buf, size);
     * @endcode
     */
    class Fence
    {
        friend class Context;
        void *native;
        Fence(void *pd);
    public:
        ~Fence();
        /**
         * @brief Block until the backend signals this fence.
         * @param timeoutNs Maximum time to wait in nanoseconds.
         *                  Default (UINT64_MAX) waits forever.
         * @return true if the fence was signaled, false if the timeout expired.
         */
        bool wait(uint64_t timeoutNs = UINT64_MAX);
        /**
         * @brief Poll whether the fence has been signaled without blocking.
         * @return true if the backend has completed the associated dispatch.
         */
        bool isSignaled() const;
    };

} // namespace systems::leal::campello_nn

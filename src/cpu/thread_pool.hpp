#pragma once

#include <cstdint>
#include <functional>

namespace systems::leal::campello_nn
{

    /**
     * @brief Fixed-size worker pool backing `parallelFor()` below.
     *
     * Internal to the CPU backend — not exposed via any public header. A single
     * process-wide instance is shared by every `CpuBackend::dispatch()` call (see
     * `sharedThreadPool()`), so there's no per-dispatch thread creation cost.
     */
    class ThreadPool
    {
    public:
        ThreadPool();
        ~ThreadPool();

        ThreadPool(const ThreadPool &) = delete;
        ThreadPool &operator=(const ThreadPool &) = delete;

        /// Number of worker threads (== std::thread::hardware_concurrency(), at least 1).
        unsigned threadCount() const;

        /// Enqueues `task` to run on a worker thread.
        void submit(std::function<void()> task);

    private:
        struct Impl;
        Impl *impl;
    };

    /// Process-wide pool instance, lazily constructed on first use.
    ThreadPool &sharedThreadPool();

#if !(defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__))
#define CAMPELLO_NN_CPU_THREADING_ENABLED 1
#endif

#ifdef CAMPELLO_NN_CPU_THREADING_ENABLED
    /// Real implementation, defined in thread_pool.cpp (kept out of the header
    /// so <thread>/<latch>/<mutex> stay out of every TU that includes this file).
    void parallelForImpl(int64_t begin, int64_t end, int64_t grainSize,
                          const std::function<void(int64_t, int64_t)> &body);
#endif

    /**
     * @brief Fork-join parallel-for over `[begin, end)`, chunked by `grainSize`.
     *
     * `body(chunkBegin, chunkEnd)` is invoked once per chunk, possibly concurrently
     * from multiple threads — every targeted kernel in `ops.cpp` writes only to a
     * disjoint output range per chunk and reads shared (but not concurrently
     * written) input tensors, so no extra synchronization is needed inside `body`.
     *
     * If `end - begin` doesn't fill at least two `grainSize`-sized chunks, `body`
     * is called once inline with the full range and the thread pool is never
     * touched — this is what keeps every existing small-tensor test on a plain
     * serial path with zero behavior change.
     *
     * **Invariant:** `body` must never itself call `parallelFor()` (no nested
     * usage exists anywhere in `ops.cpp` — every chunk body is pure math over a
     * sub-range, never a recursive call back into `evalNode()`). Nesting would
     * risk deadlocking against this same pool's worker threads.
     *
     * On Emscripten builds without pthreads enabled (`cmake/wasm.cmake` doesn't
     * pass `-pthread`/`-sUSE_PTHREADS=1`), `std::thread` isn't usable at all, so
     * this degrades to always calling `body(begin, end)` inline — see
     * `CAMPELLO_NN_CPU_THREADING_ENABLED` above.
     */
    template <typename F>
    void parallelFor(int64_t begin, int64_t end, int64_t grainSize, F &&body)
    {
#ifdef CAMPELLO_NN_CPU_THREADING_ENABLED
        parallelForImpl(begin, end, grainSize, std::function<void(int64_t, int64_t)>(body));
#else
        body(begin, end);
#endif
    }

} // namespace systems::leal::campello_nn

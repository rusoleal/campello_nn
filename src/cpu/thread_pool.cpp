#include "thread_pool.hpp"

#ifdef CAMPELLO_NN_CPU_THREADING_ENABLED

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <latch>
#include <mutex>
#include <thread>
#include <vector>

using namespace systems::leal::campello_nn;

struct ThreadPool::Impl
{
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> queue;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;

    void workerLoop()
    {
        for (;;)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return stop || !queue.empty(); });
                if (stop && queue.empty())
                    return;
                task = std::move(queue.front());
                queue.pop_front();
            }
            task();
        }
    }
};

ThreadPool::ThreadPool() : impl(new Impl())
{
    unsigned n = std::max(1u, std::thread::hardware_concurrency());
    impl->workers.reserve(n);
    for (unsigned i = 0; i < n; i++)
        impl->workers.emplace_back([this] { impl->workerLoop(); });
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        impl->stop = true;
    }
    impl->cv.notify_all();
    for (auto &t : impl->workers)
        t.join();
    delete impl;
}

unsigned ThreadPool::threadCount() const
{
    return static_cast<unsigned>(impl->workers.size());
}

void ThreadPool::submit(std::function<void()> task)
{
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        impl->queue.push_back(std::move(task));
    }
    impl->cv.notify_one();
}

ThreadPool &systems::leal::campello_nn::sharedThreadPool()
{
    static ThreadPool pool;
    return pool;
}

void systems::leal::campello_nn::parallelForImpl(
    int64_t begin, int64_t end, int64_t grainSize,
    const std::function<void(int64_t, int64_t)> &body)
{
    int64_t total = end - begin;
    ThreadPool &pool = sharedThreadPool();
    int64_t maxChunks = static_cast<int64_t>(pool.threadCount());

    int64_t numChunks = std::min(maxChunks, total / std::max<int64_t>(1, grainSize));
    if (numChunks < 2)
    {
        body(begin, end);
        return;
    }

    int64_t chunkSize = (total + numChunks - 1) / numChunks;
    numChunks = (total + chunkSize - 1) / chunkSize;

    std::latch done(numChunks - 1);
    for (int64_t i = 0; i < numChunks - 1; i++)
    {
        int64_t chunkBegin = begin + i * chunkSize;
        int64_t chunkEnd = std::min(chunkBegin + chunkSize, end);
        pool.submit([&body, chunkBegin, chunkEnd, &done] {
            body(chunkBegin, chunkEnd);
            done.count_down();
        });
    }

    // Run the last chunk inline on the calling thread instead of submitting it,
    // so the calling thread does useful work while waiting rather than idling.
    int64_t lastBegin = begin + (numChunks - 1) * chunkSize;
    body(lastBegin, end);

    done.wait();
}

#endif // CAMPELLO_NN_CPU_THREADING_ENABLED

#include <campello_nn/fence.hpp>
#include "resource_data.hpp"

using namespace systems::leal::campello_nn;

Fence::Fence(void *pd) : native(pd) {}

Fence::~Fence()
{
    auto data = (FenceData *)native;
    data->backend->destroyFence(data->native);
    delete data;
}

bool Fence::wait(uint64_t timeoutNs)
{
    auto data = (FenceData *)native;
    return data->backend->waitFence(data->native, timeoutNs);
}

bool Fence::isSignaled() const
{
    auto data = (FenceData *)native;
    return data->backend->isFenceSignaled(data->native);
}

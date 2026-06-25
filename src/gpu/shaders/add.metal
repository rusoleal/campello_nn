#include <metal_stdlib>
using namespace metal;

// Metal. See relu.metal's comment for why the thread-0 gate is needed here
// (campello_gpu's dispatchWorkgroups() always uses multiple threads per
// threadgroup on this backend, not a value this shader controls).

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void computeMain(const device float *aBuf [[buffer(0)]],
                         const device float *bBuf [[buffer(1)]],
                         device float *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    if (idx >= params.count)
        return;
    outputBuf[idx] = aBuf[idx] + bBuf[idx];
}

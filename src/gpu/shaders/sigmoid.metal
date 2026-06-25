#include <metal_stdlib>
using namespace metal;

// Metal. See relu.metal's comment for the thread-0 gate rationale.

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void computeMain(const device float *inputBuf [[buffer(0)]],
                         device float *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    if (idx >= params.count)
        return;
    outputBuf[idx] = 1.0f / (1.0f + exp(-inputBuf[idx]));
}

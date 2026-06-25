#include <metal_stdlib>
using namespace metal;

// Metal. See relu.metal's comment for the thread-0 gate rationale, and
// matmul.comp's comment for the rank-2/unbatched/naive-K-loop scope.

struct Params
{
    uint m;
    uint k;
    uint n;
    uint pad0;
};

kernel void computeMain(const device float *aBuf [[buffer(0)]],
                         const device float *bBuf [[buffer(1)]],
                         device float *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint2 groupId [[threadgroup_position_in_grid]],
                         uint2 localId [[thread_position_in_threadgroup]])
{
    if (localId.x != 0)
        return;
    uint n = groupId.x;
    uint m = groupId.y;
    if (m >= params.m || n >= params.n)
        return;
    float sum = 0.0f;
    for (uint i = 0; i < params.k; i++)
        sum += aBuf[m * params.k + i] * bBuf[i * params.n + n];
    outputBuf[m * params.n + n] = sum;
}

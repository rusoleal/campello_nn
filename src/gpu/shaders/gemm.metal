#include <metal_stdlib>
using namespace metal;

// Metal. See relu.metal's comment for the thread-0 gate rationale, and
// gemm.comp's comment for the alpha/beta/C-broadcast scope.

struct Params
{
    uint m, k, n, cElems;
    float alpha, beta;
    uint pad0, pad1;
};

kernel void computeMain(const device float *aBuf [[buffer(0)]],
                         const device float *bBuf [[buffer(1)]],
                         const device float *cBuf [[buffer(2)]],
                         device float *outputBuf [[buffer(3)]],
                         constant Params &params [[buffer(4)]],
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
    float cv = params.cElems == 1 ? cBuf[0]
               : params.cElems == params.n ? cBuf[n]
                                            : cBuf[m * params.n + n];
    outputBuf[m * params.n + n] = params.alpha * sum + params.beta * cv;
}

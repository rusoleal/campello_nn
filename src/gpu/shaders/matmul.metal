#include <metal_stdlib>
using namespace metal;

// Metal. Batched matmul with 1D column tiling. See matmul.comp.
// campello_gpu's Metal dispatch uses the pipeline's threadExecutionWidth as
// the threadgroup size, so we gate to the first tileWidth threads and have
// each compute one column of the tile. tileWidth is passed at runtime.
struct Params
{
    uint m;
    uint k;
    uint n;
    uint batchCount;
    uint tileWidth;
};

kernel void computeMain(const device float *aBuf [[buffer(0)]],
                         const device float *bBuf [[buffer(1)]],
                         device float *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint3 groupId [[threadgroup_position_in_grid]],
                         uint3 localId [[thread_position_in_threadgroup]])
{
    if (localId.x >= params.tileWidth)
        return;
    uint n = groupId.x * params.tileWidth + localId.x;
    uint m = groupId.y;
    uint batch = groupId.z;
    if (m >= params.m || n >= params.n || batch >= params.batchCount)
        return;

    uint mn = params.m * params.n;
    uint mk = params.m * params.k;
    uint kn = params.k * params.n;
    uint aOffset = batch * mk;
    uint bOffset = batch * kn;
    uint outOffset = batch * mn;

    float sum = 0.0f;
    for (uint i = 0; i < params.k; i++)
        sum += aBuf[aOffset + m * params.k + i] * bBuf[bOffset + i * params.n + n];
    outputBuf[outOffset + m * params.n + n] = sum;
}

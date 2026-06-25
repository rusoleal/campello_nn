#include <metal_stdlib>
using namespace metal;

// Metal. Same row-per-workgroup dispatch model as layernorm.metal — see that
// file's comment.

struct Params
{
    uint lastDim;
    float eps;
    uint pad1;
    uint pad2;
};

kernel void computeMain(const device float *xBuf [[buffer(0)]],
                         const device float *scaleBuf [[buffer(1)]],
                         device float *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint row = groupId;
    uint base = row * params.lastDim;

    float meanSquare = 0.0f;
    for (uint k = 0; k < params.lastDim; k++)
    {
        float v = xBuf[base + k];
        meanSquare += v * v;
    }
    meanSquare /= float(params.lastDim);
    float invRms = 1.0f / sqrt(meanSquare + params.eps);

    for (uint k = 0; k < params.lastDim; k++)
        outputBuf[base + k] = xBuf[base + k] * invRms * scaleBuf[k];
}

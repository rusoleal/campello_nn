#include <metal_stdlib>
using namespace metal;

// Metal. Same row-per-workgroup dispatch model as layernorm.comp — see that
// file's comment. Thread-0 gate for the same reason as relu.metal.

struct Params
{
    uint lastDim;
    float eps;
    uint pad1;
    uint pad2;
};

kernel void computeMain(const device float *xBuf [[buffer(0)]],
                         const device float *scaleBuf [[buffer(1)]],
                         const device float *biasBuf [[buffer(2)]],
                         device float *outputBuf [[buffer(3)]],
                         constant Params &params [[buffer(4)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint row = groupId;
    uint base = row * params.lastDim;

    float mean = 0.0f;
    for (uint k = 0; k < params.lastDim; k++)
        mean += xBuf[base + k];
    mean /= float(params.lastDim);

    float var = 0.0f;
    for (uint k = 0; k < params.lastDim; k++)
    {
        float d = xBuf[base + k] - mean;
        var += d * d;
    }
    var /= float(params.lastDim);
    float invStd = 1.0f / sqrt(var + params.eps);

    for (uint k = 0; k < params.lastDim; k++)
        outputBuf[base + k] = (xBuf[base + k] - mean) * invStd * scaleBuf[k] + biasBuf[k];
}

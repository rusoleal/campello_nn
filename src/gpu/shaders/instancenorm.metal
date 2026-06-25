#include <metal_stdlib>
using namespace metal;

// Metal. Same row(=plane)-per-workgroup two-pass model as
// instancenorm.comp — see that file's comment. Thread-0 gate for the same
// reason as relu.metal.

struct Params
{
    uint spatial;
    uint C;
    float eps;
    uint pad0;
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
    uint c = row % params.C;
    uint base = row * params.spatial;

    float mean = 0.0f;
    for (uint k = 0; k < params.spatial; k++)
        mean += xBuf[base + k];
    mean /= float(params.spatial);

    float var = 0.0f;
    for (uint k = 0; k < params.spatial; k++)
    {
        float d = xBuf[base + k] - mean;
        var += d * d;
    }
    var /= float(params.spatial);
    float invStd = 1.0f / sqrt(var + params.eps);

    float s = scaleBuf[c];
    float b = biasBuf[c];
    for (uint k = 0; k < params.spatial; k++)
        outputBuf[base + k] = (xBuf[base + k] - mean) * invStd * s + b;
}

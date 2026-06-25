#include <metal_stdlib>
using namespace metal;

// Metal. Same plain one-workgroup-per-element model as batchnorm.comp —
// see that file's comment. Thread-0 gate for the same reason as relu.metal.

struct Params
{
    uint count;
    uint C;
    uint spatial;
    float eps;
};

kernel void computeMain(const device float *xBuf [[buffer(0)]],
                         const device float *meanBuf [[buffer(1)]],
                         const device float *varBuf [[buffer(2)]],
                         const device float *scaleBuf [[buffer(3)]],
                         const device float *biasBuf [[buffer(4)]],
                         device float *outputBuf [[buffer(5)]],
                         constant Params &params [[buffer(6)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    if (idx >= params.count)
        return;
    uint c = (idx / params.spatial) % params.C;
    float invStd = 1.0f / sqrt(varBuf[c] + params.eps);
    outputBuf[idx] = (xBuf[idx] - meanBuf[c]) * invStd * scaleBuf[c] + biasBuf[c];
}

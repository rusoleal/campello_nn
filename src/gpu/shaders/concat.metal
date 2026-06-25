#include <metal_stdlib>
using namespace metal;

// Metal. Same per-input-dispatch outer/axis/inner model as concat.comp —
// see that file's comment.

struct Params
{
    uint count;
    uint axisSizeIn;
    uint axisSizeOut;
    uint innerSize;
    uint axisOffset;
    uint pad0, pad1, pad2;
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
    uint inner = idx % params.innerSize;
    uint t = idx / params.innerSize;
    uint axisIdx = t % params.axisSizeIn;
    uint outer = t / params.axisSizeIn;
    uint dst = (outer * params.axisSizeOut + (axisIdx + params.axisOffset)) * params.innerSize + inner;
    outputBuf[dst] = inputBuf[idx];
}

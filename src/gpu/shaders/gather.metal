#include <metal_stdlib>
using namespace metal;

// Metal. Same outer/axisSize/innerSize decomposition as gather.comp — see
// that file's comment.

struct Params
{
    uint outerSize;
    uint axisSize;
    uint innerSize;
    uint numIndices;
};

kernel void computeMain(const device float *dataBuf [[buffer(0)]],
                         const device uint *indicesBuf [[buffer(1)]],
                         device float *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    uint count = params.outerSize * params.numIndices * params.innerSize;
    if (idx >= count)
        return;
    uint inner = idx % params.innerSize;
    uint t = idx / params.innerSize;
    uint ii = t % params.numIndices;
    uint outer = t / params.numIndices;
    uint gatheredIdx = indicesBuf[ii];
    uint src = (outer * params.axisSize + gatheredIdx) * params.innerSize + inner;
    outputBuf[idx] = dataBuf[src];
}

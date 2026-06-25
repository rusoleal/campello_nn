#include <metal_stdlib>
using namespace metal;

// Metal. Same row-per-workgroup dispatch model and generic per-dim decode as
// softmax.comp — see that file's comment. Thread-0 gate for the same reason
// as relu.metal.

struct Params
{
    uint outerRank, axisSize, axisStride, outerTotal;
    uint divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
    uint origStride0, origStride1, origStride2, origStride3, origStride4, origStride5, origStride6, origStride7;
};

kernel void computeMain(const device float *inputBuf [[buffer(0)]],
                         device float *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint outerIdx = groupId;
    if (outerIdx >= params.outerTotal)
        return;

    uint divisor[8] = {params.divisor0, params.divisor1, params.divisor2, params.divisor3,
                        params.divisor4, params.divisor5, params.divisor6, params.divisor7};
    uint origStride[8] = {params.origStride0, params.origStride1, params.origStride2, params.origStride3,
                           params.origStride4, params.origStride5, params.origStride6, params.origStride7};

    uint remaining = outerIdx;
    uint base = 0;
    for (uint d = 0; d < params.outerRank; d++)
    {
        uint idx = remaining / divisor[d];
        remaining -= idx * divisor[d];
        base += idx * origStride[d];
    }

    float maxV = inputBuf[base];
    for (uint k = 1; k < params.axisSize; k++)
        maxV = max(maxV, inputBuf[base + k * params.axisStride]);

    float sum = 0.0f;
    for (uint k = 0; k < params.axisSize; k++)
    {
        float e = exp(inputBuf[base + k * params.axisStride] - maxV);
        outputBuf[base + k * params.axisStride] = e;
        sum += e;
    }

    for (uint k = 0; k < params.axisSize; k++)
        outputBuf[base + k * params.axisStride] /= sum;
}

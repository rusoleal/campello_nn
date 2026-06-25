#include <metal_stdlib>
using namespace metal;

// Metal. Same one-workgroup-per-element model and thread-0 gate as
// relu.metal; same generic per-dim decode as slice.comp — see that file's
// comment.

struct Params
{
    uint rank;
    uint count;
    uint baseOffset;
    uint pad0;
    uint divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
    uint mult0, mult1, mult2, mult3, mult4, mult5, mult6, mult7;
};

kernel void computeMain(const device float *inputBuf [[buffer(0)]],
                         device float *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint outIdx = groupId;
    if (outIdx >= params.count)
        return;

    uint divisor[8] = {params.divisor0, params.divisor1, params.divisor2, params.divisor3,
                        params.divisor4, params.divisor5, params.divisor6, params.divisor7};
    uint multiplier[8] = {params.mult0, params.mult1, params.mult2, params.mult3,
                           params.mult4, params.mult5, params.mult6, params.mult7};

    uint remaining = outIdx;
    uint srcOffset = params.baseOffset;
    for (uint d = 0; d < params.rank; d++)
    {
        uint idx = remaining / divisor[d];
        remaining -= idx * divisor[d];
        srcOffset += idx * multiplier[d];
    }
    outputBuf[outIdx] = inputBuf[srcOffset];
}

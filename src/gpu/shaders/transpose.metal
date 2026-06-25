#include <metal_stdlib>
using namespace metal;

// Metal. Same one-workgroup-per-element model and thread-0 gate as
// relu.metal; same generic arbitrary-rank permute as transpose.comp — see
// that file's comment for the divisor/gatherStride math and why Params uses
// flat named fields instead of real arrays.

struct Params
{
    uint rank;
    uint count;
    uint divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
    uint gather0, gather1, gather2, gather3, gather4, gather5, gather6, gather7;
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
    uint gatherStride[8] = {params.gather0, params.gather1, params.gather2, params.gather3,
                             params.gather4, params.gather5, params.gather6, params.gather7};

    uint remaining = outIdx;
    uint srcOffset = 0;
    for (uint d = 0; d < params.rank; d++)
    {
        uint idx = remaining / divisor[d];
        remaining -= idx * divisor[d];
        srcOffset += idx * gatherStride[d];
    }
    outputBuf[outIdx] = inputBuf[srcOffset];
}

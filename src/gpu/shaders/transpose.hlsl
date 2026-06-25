// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// generic arbitrary-rank permute as transpose.comp — see that file's
// comment for the divisor/gatherStride math and why Params uses flat named
// fields instead of real arrays.

struct Params
{
    uint rank;
    uint count;
    uint divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
    uint gather0, gather1, gather2, gather3, gather4, gather5, gather6, gather7;
};

StructuredBuffer<float> inputBuf : register(t0);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint outIdx = groupId.x;
    if (outIdx >= params.count)
        return;

    uint divisor[8] = { params.divisor0, params.divisor1, params.divisor2, params.divisor3,
                         params.divisor4, params.divisor5, params.divisor6, params.divisor7 };
    uint gatherStride[8] = { params.gather0, params.gather1, params.gather2, params.gather3,
                              params.gather4, params.gather5, params.gather6, params.gather7 };

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

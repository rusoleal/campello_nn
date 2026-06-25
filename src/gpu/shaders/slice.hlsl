// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// generic per-dim decode as slice.comp — see that file's comment.

struct Params
{
    uint rank;
    uint count;
    uint baseOffset;
    uint pad0;
    uint divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
    uint mult0, mult1, mult2, mult3, mult4, mult5, mult6, mult7;
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
    uint multiplier[8] = { params.mult0, params.mult1, params.mult2, params.mult3,
                            params.mult4, params.mult5, params.mult6, params.mult7 };

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

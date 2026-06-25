// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// row-per-workgroup dispatch model and generic per-dim decode as
// softmax.comp — see that file's comment.

struct Params
{
    uint outerRank, axisSize, axisStride, outerTotal;
    uint divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
    uint origStride0, origStride1, origStride2, origStride3, origStride4, origStride5, origStride6, origStride7;
};

StructuredBuffer<float> inputBuf : register(t0);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint outerIdx = groupId.x;
    if (outerIdx >= params.outerTotal)
        return;

    uint divisor[8] = { params.divisor0, params.divisor1, params.divisor2, params.divisor3,
                         params.divisor4, params.divisor5, params.divisor6, params.divisor7 };
    uint origStride[8] = { params.origStride0, params.origStride1, params.origStride2, params.origStride3,
                            params.origStride4, params.origStride5, params.origStride6, params.origStride7 };

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

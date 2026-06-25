// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// per-input-dispatch outer/axis/inner model as concat.comp — see that
// file's comment.

struct Params
{
    uint count;
    uint axisSizeIn;
    uint axisSizeOut;
    uint innerSize;
    uint axisOffset;
    uint pad0, pad1, pad2;
};

StructuredBuffer<float> inputBuf : register(t0);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;
    uint inner = idx % params.innerSize;
    uint t = idx / params.innerSize;
    uint axisIdx = t % params.axisSizeIn;
    uint outer = t / params.axisSizeIn;
    uint dst = (outer * params.axisSizeOut + (axisIdx + params.axisOffset)) * params.innerSize + inner;
    outputBuf[dst] = inputBuf[idx];
}

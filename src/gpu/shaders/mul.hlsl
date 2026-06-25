// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float> aBuf : register(t0);
StructuredBuffer<float> bBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;
    outputBuf[idx] = aBuf[idx] * bBuf[idx];
}

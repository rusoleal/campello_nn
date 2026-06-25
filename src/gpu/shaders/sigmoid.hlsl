// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
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
    outputBuf[idx] = 1.0f / (1.0f + exp(-inputBuf[idx]));
}

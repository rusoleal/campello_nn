// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// outer/axisSize/innerSize decomposition as gather.comp — see that file's
// comment.

struct Params
{
    uint outerSize;
    uint axisSize;
    uint innerSize;
    uint numIndices;
};

StructuredBuffer<float> dataBuf : register(t0);
StructuredBuffer<uint> indicesBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
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

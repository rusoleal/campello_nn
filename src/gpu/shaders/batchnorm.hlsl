// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// plain one-workgroup-per-element model as batchnorm.comp.

struct Params
{
    uint count;
    uint C;
    uint spatial;
    float eps;
};

StructuredBuffer<float> xBuf : register(t0);
StructuredBuffer<float> meanBuf : register(t1);
StructuredBuffer<float> varBuf : register(t2);
StructuredBuffer<float> scaleBuf : register(t3);
StructuredBuffer<float> biasBuf : register(t4);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;
    uint c = (idx / params.spatial) % params.C;
    float invStd = 1.0f / sqrt(varBuf[c] + params.eps);
    outputBuf[idx] = (xBuf[idx] - meanBuf[c]) * invStd * scaleBuf[c] + biasBuf[c];
}

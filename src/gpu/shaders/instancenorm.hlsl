// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// row(=plane)-per-workgroup two-pass model as instancenorm.comp.

struct Params
{
    uint spatial;
    uint C;
    float eps;
    uint pad0;
};

StructuredBuffer<float> xBuf : register(t0);
StructuredBuffer<float> scaleBuf : register(t1);
StructuredBuffer<float> biasBuf : register(t2);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint row = groupId.x;
    uint c = row % params.C;
    uint base = row * params.spatial;

    float mean = 0.0f;
    for (uint k = 0; k < params.spatial; k++)
        mean += xBuf[base + k];
    mean /= (float)params.spatial;

    float var = 0.0f;
    for (uint k = 0; k < params.spatial; k++)
    {
        float d = xBuf[base + k] - mean;
        var += d * d;
    }
    var /= (float)params.spatial;
    float invStd = 1.0f / sqrt(var + params.eps);

    float s = scaleBuf[c];
    float b = biasBuf[c];
    for (uint k = 0; k < params.spatial; k++)
        outputBuf[base + k] = (xBuf[base + k] - mean) * invStd * s + b;
}

// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// row-per-workgroup dispatch model as layernorm.comp.

struct Params
{
    uint lastDim;
    float eps;
    uint pad1;
    uint pad2;
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
    uint base = row * params.lastDim;

    float mean = 0.0f;
    for (uint k = 0; k < params.lastDim; k++)
        mean += xBuf[base + k];
    mean /= (float)params.lastDim;

    float var = 0.0f;
    for (uint k = 0; k < params.lastDim; k++)
    {
        float d = xBuf[base + k] - mean;
        var += d * d;
    }
    var /= (float)params.lastDim;
    float invStd = 1.0f / sqrt(var + params.eps);

    for (uint k = 0; k < params.lastDim; k++)
        outputBuf[base + k] = (xBuf[base + k] - mean) * invStd * scaleBuf[k] + biasBuf[k];
}

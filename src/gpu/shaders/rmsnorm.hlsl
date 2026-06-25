// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// row-per-workgroup dispatch model as layernorm.hlsl.

struct Params
{
    uint lastDim;
    float eps;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float> xBuf : register(t0);
StructuredBuffer<float> scaleBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint row = groupId.x;
    uint base = row * params.lastDim;

    float meanSquare = 0.0f;
    for (uint k = 0; k < params.lastDim; k++)
    {
        float v = xBuf[base + k];
        meanSquare += v * v;
    }
    meanSquare /= (float)params.lastDim;
    float invRms = 1.0f / sqrt(meanSquare + params.eps);

    for (uint k = 0; k < params.lastDim; k++)
        outputBuf[base + k] = xBuf[base + k] * invRms * scaleBuf[k];
}

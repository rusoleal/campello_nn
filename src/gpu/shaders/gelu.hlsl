// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// erf approximation as gelu.comp/gelu.metal (HLSL has no built-in erf either).

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

float erfApprox(float x)
{
    float s = x < 0.0f ? -1.0f : 1.0f;
    float ax = abs(x);
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p = 0.3275911f;
    float t = 1.0f / (1.0f + p * ax);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return s * y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;
    float v = inputBuf[idx];
    outputBuf[idx] = 0.5f * v * (1.0f + erfApprox(v * 0.70710678118654752f));
}

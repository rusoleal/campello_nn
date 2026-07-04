// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// erf approximation as gelu.comp/gelu.metal (HLSL has no built-in erf either).

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<T> inputBuf : register(t0);
RWStructuredBuffer<T> outputBuf : register(u0);
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

T4 geluT4(T4 v)
{
    T4 r;
    for (int i = 0; i < 4; ++i)
    {
        float x = v[i];
        r[i] = 0.5f * x * (1.0f + erfApprox(x * 0.70710678118654752f));
    }
    return r;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint base = groupId.x * 4u;
    if (base >= params.count)
        return;

    uint end = min(base + 4u, params.count);
    if (end - base == 4u)
    {
        T4 v = T4(inputBuf[base], inputBuf[base + 1u],
                  inputBuf[base + 2u], inputBuf[base + 3u]);
        v = geluT4(v);
        outputBuf[base]      = v.x;
        outputBuf[base + 1u] = v.y;
        outputBuf[base + 2u] = v.z;
        outputBuf[base + 3u] = v.w;
    }
    else
    {
        for (uint i = base; i < end; ++i)
        {
            float v = inputBuf[i];
            outputBuf[i] = 0.5f * v * (1.0f + erfApprox(v * 0.70710678118654752f));
        }
    }
}

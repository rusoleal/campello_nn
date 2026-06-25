// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// rank-2/unbatched/naive-K-loop scope as matmul.comp.

struct Params
{
    uint m;
    uint k;
    uint n;
    uint pad0;
};

StructuredBuffer<float> aBuf : register(t0);
StructuredBuffer<float> bBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint n = groupId.x;
    uint m = groupId.y;
    if (m >= params.m || n >= params.n)
        return;
    float sum = 0.0f;
    for (uint i = 0; i < params.k; i++)
        sum += aBuf[m * params.k + i] * bBuf[i * params.n + n];
    outputBuf[m * params.n + n] = sum;
}

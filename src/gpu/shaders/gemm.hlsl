// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// alpha/beta/C-broadcast scope as gemm.comp.

struct Params
{
    uint m, k, n, cElems;
    float alpha, beta;
    uint pad0, pad1;
};

StructuredBuffer<float> aBuf : register(t0);
StructuredBuffer<float> bBuf : register(t1);
StructuredBuffer<float> cBuf : register(t2);
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
    float cv = params.cElems == 1 ? cBuf[0]
               : params.cElems == params.n ? cBuf[n]
                                            : cBuf[m * params.n + n];
    outputBuf[m * params.n + n] = params.alpha * sum + params.beta * cv;
}

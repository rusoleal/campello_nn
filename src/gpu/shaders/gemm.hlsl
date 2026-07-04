// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// alpha/beta/C-broadcast scope as gemm.comp.

struct Params
{
    uint m, k, n, cElems;
    float alpha, beta;
};

StructuredBuffer<T> aBuf : register(t0);
StructuredBuffer<T> bBuf : register(t1);
StructuredBuffer<T> cBuf : register(t2);
RWStructuredBuffer<T> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

float4 toFloat4(T4 v)
{
    return float4(TO_FLOAT(v.x), TO_FLOAT(v.y), TO_FLOAT(v.z), TO_FLOAT(v.w));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint n = groupId.x;
    uint m = groupId.y;
    if (m >= params.m || n >= params.n)
        return;

    uint aRow = m * params.k;
    uint nVal = n;
    uint kDim = params.k;
    uint nDim = params.n;

    float sum = 0.0f;
    uint i = 0u;
    for (; i + 3u < kDim; i += 4u)
    {
        T4 a = T4(aBuf[aRow + i],
                  aBuf[aRow + i + 1u],
                  aBuf[aRow + i + 2u],
                  aBuf[aRow + i + 3u]);
        float b0 = TO_FLOAT(bBuf[(i + 0u) * nDim + nVal]);
        float b1 = TO_FLOAT(bBuf[(i + 1u) * nDim + nVal]);
        float b2 = TO_FLOAT(bBuf[(i + 2u) * nDim + nVal]);
        float b3 = TO_FLOAT(bBuf[(i + 3u) * nDim + nVal]);
        sum += dot(toFloat4(a), float4(b0, b1, b2, b3));
    }
    for (; i < kDim; ++i)
        sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[i * nDim + nVal]);

    float cv = params.cElems == 1 ? TO_FLOAT(cBuf[0])
               : params.cElems == params.n ? TO_FLOAT(cBuf[nVal])
                                            : TO_FLOAT(cBuf[m * nDim + nVal]);
    outputBuf[m * nDim + nVal] = TO_T(params.alpha * sum + params.beta * cv);
}

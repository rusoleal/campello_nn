// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// Batched matmul with 1D column tiling. See matmul.comp.

struct Params
{
    uint m;
    uint k;
    uint n;
    uint batchCount;
    uint tileWidth;
};

StructuredBuffer<T> aBuf : register(t0);
StructuredBuffer<T> bBuf : register(t1);
RWStructuredBuffer<T> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

float4 toFloat4(T4 v)
{
    return float4(TO_FLOAT(v.x), TO_FLOAT(v.y), TO_FLOAT(v.z), TO_FLOAT(v.w));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    if (localId.x >= params.tileWidth)
        return;
    uint n = groupId.x * params.tileWidth + localId.x;
    uint m = groupId.y;
    uint batch = groupId.z;
    if (m >= params.m || n >= params.n || batch >= params.batchCount)
        return;

    uint mn = params.m * params.n;
    uint mk = params.m * params.k;
    uint kn = params.k * params.n;
    uint aOffset = batch * mk;
    uint bOffset = batch * kn;
    uint outOffset = batch * mn;

    uint aRow = aOffset + m * params.k;
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
        float b0 = TO_FLOAT(bBuf[bOffset + (i + 0u) * nDim + nVal]);
        float b1 = TO_FLOAT(bBuf[bOffset + (i + 1u) * nDim + nVal]);
        float b2 = TO_FLOAT(bBuf[bOffset + (i + 2u) * nDim + nVal]);
        float b3 = TO_FLOAT(bBuf[bOffset + (i + 3u) * nDim + nVal]);
        sum += dot(toFloat4(a), float4(b0, b1, b2, b3));
    }
    for (; i < kDim; ++i)
        sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[bOffset + i * nDim + nVal]);

    outputBuf[outOffset + m * nDim + nVal] = TO_T(sum);
}

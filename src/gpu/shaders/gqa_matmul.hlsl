// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// Grouped-query batched matmul. See gqa_matmul.comp for the full description.

struct Params
{
    uint m;
    uint k;
    uint n;
    uint batchA;
    uint groupSize;
    uint transposeB;
    uint tileWidth;
};

StructuredBuffer<T> aBuf : register(t0);
StructuredBuffer<T> bBuf : register(t1);
RWStructuredBuffer<T> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    if (localId.x >= params.tileWidth)
        return;
    uint n = groupId.x * params.tileWidth + localId.x;
    uint m = groupId.y;
    uint batch = groupId.z;
    if (m >= params.m || n >= params.n || batch >= params.batchA)
        return;

    uint kDim = params.k;
    uint nDim = params.n;
    uint kv = batch / params.groupSize;

    uint aRow = batch * params.m * kDim + m * kDim;
    uint bBase = kv * kDim * nDim;

    float sum = 0.0f;
    if (params.transposeB != 0u)
    {
        uint bRow = bBase + n * kDim;
        for (uint i = 0u; i < kDim; ++i)
            sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[bRow + i]);
    }
    else
    {
        for (uint i = 0u; i < kDim; ++i)
            sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[bBase + i * nDim + n]);
    }

    outputBuf[batch * params.m * nDim + m * nDim + n] = TO_T(sum);
}

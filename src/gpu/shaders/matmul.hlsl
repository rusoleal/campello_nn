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

StructuredBuffer<float> aBuf : register(t0);
StructuredBuffer<float> bBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

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

    float sum = 0.0f;
    for (uint i = 0; i < params.k; i++)
        sum += aBuf[aOffset + m * params.k + i] * bBuf[bOffset + i * params.n + n];
    outputBuf[outOffset + m * params.n + n] = sum;
}

// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// GEMM step of im2col-based conv2d for NCHW float32 (groups == 1).
// A = im2col matrix [M, K] row-major (input to this shader)
// B = weights [O, K] row-major (flattened OIHW)
// C = NCHW output [N, O, outH, outW]
struct Params
{
    uint M, K, O;
    uint outH, outW;
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

    uint o = groupId.x * params.tileWidth + localId.x;
    uint m = groupId.y;
    if (m >= params.M || o >= params.O)
        return;

    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++)
    {
        float a = aBuf[m * params.K + k];
        float b = bBuf[o * params.K + k];
        sum += a * b;
    }

    uint outPlane = params.outH * params.outW;
    uint n = m / outPlane;
    uint r = m % outPlane;
    uint oh = r / params.outW;
    uint ow = r % params.outW;
    outputBuf[((n * params.O + o) * params.outH + oh) * params.outW + ow] = sum;
}

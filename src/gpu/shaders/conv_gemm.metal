#include <metal_stdlib>
using namespace metal;

// Metal. GEMM step of im2col-based conv2d for NCHW float32 (groups == 1).
// A = im2col matrix [M, K] row-major (input to this shader)
// B = weights [O, K] row-major (flattened OIHW)
// C = NCHW output [N, O, outH, outW]
// Each thread computes C[m, o] and writes it directly into NCHW layout.
struct Params
{
    uint M, K, O;
    uint outH, outW;
    uint tileWidth;
};

kernel void computeMain(const device float *aBuf [[buffer(0)]],
                         const device float *bBuf [[buffer(1)]],
                         device float *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint3 groupId [[threadgroup_position_in_grid]],
                         uint3 localId [[thread_position_in_threadgroup]])
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

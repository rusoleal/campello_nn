#include <metal_stdlib>
using namespace metal;

// Metal. im2col for NCHW float32 conv2d (groups == 1).
// Output is a row-major [M, K] matrix where:
//   M = N * outH * outW
//   K = C * KH * KW
// Row m = (n * outH + oh) * outW + ow
// Col k = (c * KH + kh) * KW + kw
struct Params
{
    uint N, C, H, W;
    uint KH, KW;
    uint outH, outW;
    uint strideX, strideY;
    uint dilationX, dilationY;
    uint paddingLeft, paddingTop;
    uint tileWidth;
};

kernel void computeMain(const device float *xBuf [[buffer(0)]],
                         device float *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint3 groupId [[threadgroup_position_in_grid]],
                         uint3 localId [[thread_position_in_threadgroup]])
{
    if (localId.x >= params.tileWidth)
        return;

    uint K = params.C * params.KH * params.KW;
    uint k = groupId.x * params.tileWidth + localId.x;
    uint m = groupId.y;
    if (m >= params.N * params.outH * params.outW || k >= K)
        return;

    uint outPlane = params.outH * params.outW;
    uint n = m / outPlane;
    uint r = m % outPlane;
    uint oh = r / params.outW;
    uint ow = r % params.outW;

    uint c = k / (params.KH * params.KW);
    uint rem = k % (params.KH * params.KW);
    uint kh = rem / params.KW;
    uint kw = rem % params.KW;

    int ih = int(oh * params.strideY) + int(kh * params.dilationY) - int(params.paddingTop);
    int iw = int(ow * params.strideX) + int(kw * params.dilationX) - int(params.paddingLeft);

    float val = 0.0f;
    if (ih >= 0 && ih < int(params.H) && iw >= 0 && iw < int(params.W))
    {
        val = xBuf[((n * params.C + c) * params.H + uint(ih)) * params.W + uint(iw)];
    }
    outputBuf[m * K + k] = val;
}

#include <metal_stdlib>
using namespace metal;

// Metal. Same nearest/bilinear model as resize.comp — see that file's
// comment. Thread-0 gate for the same reason as relu.metal.

struct Params
{
    uint H, W, outH, outW;
    uint mode, centerResult, alignCorners, nearestRoundsDown;
};

float resizeSrcCoord(uint dst, uint inSize, uint outSize, constant Params &params)
{
    if (params.alignCorners != 0)
    {
        float scale = outSize > 1 ? float(inSize - 1) / float(outSize - 1) : 0.0f;
        return float(dst) * scale;
    }
    float scale = float(inSize) / float(outSize);
    float src = params.centerResult != 0 ? (float(dst) + 0.5f) * scale - 0.5f : float(dst) * scale;
    return clamp(src, 0.0f, float(inSize - 1));
}

kernel void computeMain(const device float *inputBuf [[buffer(0)]],
                         device float *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint3 groupId [[threadgroup_position_in_grid]],
                         uint3 localId [[thread_position_in_threadgroup]])
{
    if (localId.x != 0)
        return;
    uint ow = groupId.x;
    uint oh = groupId.y;
    uint nc = groupId.z;

    float srcH = resizeSrcCoord(oh, params.H, params.outH, params);
    float srcW = resizeSrcCoord(ow, params.W, params.outW, params);
    float value;
    if (params.mode == 0) // Nearest
    {
        int ih, iw;
        if (params.nearestRoundsDown != 0)
        {
            ih = int(floor(srcH));
            iw = int(floor(srcW));
        }
        else
        {
            ih = int(round(srcH));
            iw = int(round(srcW));
        }
        ih = clamp(ih, 0, int(params.H) - 1);
        iw = clamp(iw, 0, int(params.W) - 1);
        value = inputBuf[nc * params.H * params.W + uint(ih) * params.W + uint(iw)];
    }
    else // Bilinear
    {
        int h0 = clamp(int(floor(srcH)), 0, int(params.H) - 1);
        int h1 = min(h0 + 1, int(params.H) - 1);
        int w0 = clamp(int(floor(srcW)), 0, int(params.W) - 1);
        int w1 = min(w0 + 1, int(params.W) - 1);
        float fh = srcH - float(h0);
        float fw = srcW - float(w0);
        uint base = nc * params.H * params.W;
        float top = inputBuf[base + uint(h0) * params.W + uint(w0)] * (1.0f - fw) +
                    inputBuf[base + uint(h0) * params.W + uint(w1)] * fw;
        float bottom = inputBuf[base + uint(h1) * params.W + uint(w0)] * (1.0f - fw) +
                       inputBuf[base + uint(h1) * params.W + uint(w1)] * fw;
        value = top * (1.0f - fh) + bottom * fh;
    }
    outputBuf[(nc * params.outH + oh) * params.outW + ow] = value;
}

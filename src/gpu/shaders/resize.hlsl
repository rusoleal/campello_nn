// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// nearest/bilinear model as resize.comp.

struct Params
{
    uint H, W, outH, outW;
    uint mode, centerResult, alignCorners, nearestRoundsDown;
};

StructuredBuffer<float> inputBuf : register(t0);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

float resizeSrcCoord(uint dst, uint inSize, uint outSize)
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

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint ow = groupId.x;
    uint oh = groupId.y;
    uint nc = groupId.z;

    float srcH = resizeSrcCoord(oh, params.H, params.outH);
    float srcW = resizeSrcCoord(ow, params.W, params.outW);
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

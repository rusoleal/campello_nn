// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Shared
// by both OpKind::MaxPool2d and OpKind::AvgPool2d — see pool2d.comp's
// comment.

struct Params
{
    uint H, W, outH, outW;
    uint kernelHeight, kernelWidth, strideX, strideY, paddingLeft, paddingTop;
    uint isMax;
};

StructuredBuffer<float> inputBuf : register(t0);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint ow = groupId.x;
    uint oh = groupId.y;
    uint nc = groupId.z;

    // No portable HLSL infinity literal — build -infinity from its
    // IEEE-754 bit pattern, same trick as pool2d.comp.
    float acc = params.isMax != 0 ? asfloat(0xFF800000u) : 0.0f;
    int count = 0;
    for (uint kh = 0; kh < params.kernelHeight; kh++)
    {
        int ih = int(oh * params.strideY) - int(params.paddingTop) + int(kh);
        if (ih < 0 || ih >= int(params.H))
            continue;
        for (uint kw = 0; kw < params.kernelWidth; kw++)
        {
            int iw = int(ow * params.strideX) - int(params.paddingLeft) + int(kw);
            if (iw < 0 || iw >= int(params.W))
                continue;
            float v = inputBuf[nc * params.H * params.W + uint(ih) * params.W + uint(iw)];
            acc = params.isMax != 0 ? max(acc, v) : acc + v;
            count++;
        }
    }
    outputBuf[(nc * params.outH + oh) * params.outW + ow] =
        params.isMax != 0 ? acc : (count > 0 ? acc / float(count) : 0.0f);
}

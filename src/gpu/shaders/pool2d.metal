#include <metal_stdlib>
using namespace metal;

// Metal. Shared by both OpKind::MaxPool2d and OpKind::AvgPool2d — see
// pool2d.comp's comment. Thread-0 gate for the same reason as relu.metal.

struct Params
{
    uint H, W, outH, outW;
    uint kernelHeight, kernelWidth, strideX, strideY, paddingLeft, paddingTop;
    uint isMax;
};

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

    float acc = params.isMax != 0 ? -INFINITY : 0.0f;
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

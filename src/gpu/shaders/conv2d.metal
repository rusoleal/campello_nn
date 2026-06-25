#include <metal_stdlib>
using namespace metal;

// Metal. Same NCHW/grouped/dilated model as conv2d.comp. 1D output tiling:
// each workgroup computes TILE_WIDTH consecutive flattened output elements.
// campello_gpu's Metal dispatch uses the pipeline's threadExecutionWidth as
// the threadgroup size, so we gate to the first TILE_WIDTH threads.
#define TILE_WIDTH 8

struct Params
{
    uint N, O, C, H, W, Cg, KH, KW, outH, outW;
    uint strideX, strideY, dilationX, dilationY, paddingLeft, paddingTop;
    uint inPerGroup, outPerGroup;
};

kernel void computeMain(const device float *xBuf [[buffer(0)]],
                         const device float *wBuf [[buffer(1)]],
                         device float *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint3 groupId [[threadgroup_position_in_grid]],
                         uint3 localId [[thread_position_in_threadgroup]])
{
    if (localId.x >= TILE_WIDTH)
        return;

    uint totalOut = params.outH * params.outW;
    uint totalOutputs = params.N * params.O * totalOut;
    uint flatIdx = groupId.x * TILE_WIDTH + localId.x;
    if (flatIdx >= totalOutputs)
        return;

    uint no = flatIdx / totalOut;
    uint spatial = flatIdx % totalOut;
    uint oh = spatial / params.outW;
    uint ow = spatial % params.outW;
    uint n = no / params.O;
    uint o = no % params.O;
    uint group = o / params.outPerGroup;
    uint inChannelBase = group * params.inPerGroup;

    float sum = 0.0f;
    for (uint ci = 0; ci < params.Cg; ci++)
    {
        uint c = inChannelBase + ci;
        for (uint kh = 0; kh < params.KH; kh++)
        {
            int ih = int(oh * params.strideY) - int(params.paddingTop) + int(kh * params.dilationY);
            if (ih < 0 || ih >= int(params.H))
                continue;
            for (uint kw = 0; kw < params.KW; kw++)
            {
                int iw = int(ow * params.strideX) - int(params.paddingLeft) + int(kw * params.dilationX);
                if (iw < 0 || iw >= int(params.W))
                    continue;
                float xv = xBuf[((n * params.C + c) * params.H + uint(ih)) * params.W + uint(iw)];
                float wv = wBuf[((o * params.Cg + ci) * params.KH + kh) * params.KW + kw];
                sum += xv * wv;
            }
        }
    }
    outputBuf[(no * params.outH + oh) * params.outW + ow] = sum;
}

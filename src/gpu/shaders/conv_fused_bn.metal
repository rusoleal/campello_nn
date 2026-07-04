// Metal. Fused Conv2d + BatchNorm + ReLU for NCHW float32 (inference mode).
// Inputs:  x (NCHW), w (OIHW), scale_factor (1D [O]), folded_bias (1D [O])
// Output:  relu(conv(x,w) * scale_factor[o] + folded_bias[o]) in NCHW
//
// This implements the inference-time BN folding:
//   BN(x) = (x - mean) / sqrt(var + eps) * scale + bias
//         = x * scale_factor + folded_bias
// where scale_factor = scale / sqrt(var + eps)
//       folded_bias  = bias - mean * scale_factor
struct Params
{
    uint N, O, C, H, W, Cg, KH, KW, outH, outW;
    uint strideX, strideY, dilationX, dilationY, paddingLeft, paddingTop;
    uint inPerGroup, outPerGroup;
    uint tileWidth;
};

kernel void computeMain(const device T *xBuf [[buffer(0)]],
                         const device T *wBuf [[buffer(1)]],
                         const device T *scaleBuf [[buffer(2)]],
                         const device T *biasBuf [[buffer(3)]],
                         device T *outputBuf [[buffer(4)]],
                         constant Params &params [[buffer(5)]],
                         uint3 groupId [[threadgroup_position_in_grid]],
                         uint3 localId [[thread_position_in_threadgroup]])
{
    uint tileColsPerRow = (params.outW + params.tileWidth - 1) / params.tileWidth;
    uint owTile = groupId.x % tileColsPerRow;
    uint no = groupId.x / tileColsPerRow;
    uint n = no / params.O;
    uint o = no % params.O;
    uint oh = groupId.y;
    uint ow = owTile * params.tileWidth + localId.x;

    bool isActive = (localId.x < params.tileWidth) && (oh < params.outH) && (ow < params.outW);

    uint group = o / params.outPerGroup;
    uint inChannelBase = group * params.inPerGroup;

    float sum = 0.0f;
    if (isActive)
    {
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
                    float xv = TO_FLOAT(xBuf[((n * params.C + c) * params.H + uint(ih)) * params.W + uint(iw)]);
                    float wv = TO_FLOAT(wBuf[((o * params.Cg + ci) * params.KH + kh) * params.KW + kw]);
                    sum += xv * wv;
                }
            }
        }
    }

    if (isActive)
    {
        sum = sum * TO_FLOAT(scaleBuf[o]) + TO_FLOAT(biasBuf[o]);
        outputBuf[(no * params.outH + oh) * params.outW + ow] = TO_T(max(0.0f, sum));
    }
}

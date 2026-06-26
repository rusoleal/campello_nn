// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// Fused Conv2d + Add[bias] + ReLU for NCHW float32.
// Inputs:  x (NCHW), w (OIHW), bias (1D [O])
// Output:  relu(conv(x,w) + bias[o]) in NCHW
struct Params
{
    uint N, O, C, H, W, Cg, KH, KW, outH, outW;
    uint strideX, strideY, dilationX, dilationY, paddingLeft, paddingTop;
    uint inPerGroup, outPerGroup;
    uint tileWidth;
};

StructuredBuffer<float> xBuf : register(t0);
StructuredBuffer<float> wBuf : register(t1);
StructuredBuffer<float> biasBuf : register(t2);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
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
                    float xv = xBuf[((n * params.C + c) * params.H + uint(ih)) * params.W + uint(iw)];
                    float wv = wBuf[((o * params.Cg + ci) * params.KH + kh) * params.KW + kw];
                    sum += xv * wv;
                }
            }
        }
    }

    if (isActive)
    {
        sum += biasBuf[o];
        outputBuf[(no * params.outH + oh) * params.outW + ow] = max(0.0f, sum);
    }
}

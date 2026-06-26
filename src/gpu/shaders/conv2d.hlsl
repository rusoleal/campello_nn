// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// NCHW/grouped/dilated conv2d with shared-memory spatial tiling. See conv2d.comp.
#define MAX_TILE_OW 32
#define TILE_CG 16
#define MAX_KH 3
#define MAX_KW 3
#define MAX_TILE_IW 65
// Shared-memory tiling is implemented but currently disabled on DirectX because
// it has not been tuned yet; the 2D dispatch shape itself is still a win.
#define USE_SHARED_MEMORY 0

struct Params
{
    uint N, O, C, H, W, Cg, KH, KW, outH, outW;
    uint strideX, strideY, dilationX, dilationY, paddingLeft, paddingTop;
    uint inPerGroup, outPerGroup;
    uint tileWidth;
};

StructuredBuffer<float> xBuf : register(t0);
StructuredBuffer<float> wBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

groupshared float sharedX[TILE_CG][MAX_KH][MAX_TILE_IW];
groupshared float sharedW[TILE_CG][MAX_KH][MAX_KW];

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

    bool useShared = (USE_SHARED_MEMORY != 0) && (params.KH <= MAX_KH) && (params.KW <= MAX_KW) &&
                     (params.strideX <= 2) && (params.dilationX == 1) && (params.dilationY == 1);

    float sum = 0.0f;

    if (useShared)
    {
        int inHStart = int(oh * params.strideY) - int(params.paddingTop);
        int inWStart = int(owTile * params.tileWidth * params.strideX) - int(params.paddingLeft);
        uint tileIW = (params.tileWidth - 1) * params.strideX + (params.KW - 1) * params.dilationX + 1;

        for (uint cBase = 0; cBase < params.Cg; cBase += TILE_CG)
        {
            uint cEnd = min((uint)TILE_CG, params.Cg - cBase);

            uint xLoadCount = cEnd * params.KH * tileIW;
            for (uint idx = localId.x; idx < xLoadCount; idx += params.tileWidth)
            {
                uint ci = idx / (params.KH * tileIW);
                uint rem = idx % (params.KH * tileIW);
                uint kh = rem / tileIW;
                uint iw = rem % tileIW;
                uint c = inChannelBase + cBase + ci;
                int ih = inHStart + int(kh * params.dilationY);
                int iwReal = inWStart + int(iw * params.dilationX);
                float val = 0.0f;
                if (ih >= 0 && ih < int(params.H) && iwReal >= 0 && iwReal < int(params.W))
                {
                    val = xBuf[((n * params.C + c) * params.H + uint(ih)) * params.W + uint(iwReal)];
                }
                sharedX[ci][kh][iw] = val;
            }

            uint wLoadCount = cEnd * params.KH * params.KW;
            for (uint idx = localId.x; idx < wLoadCount; idx += params.tileWidth)
            {
                uint ci = idx / (params.KH * params.KW);
                uint rem = idx % (params.KH * params.KW);
                uint kh = rem / params.KW;
                uint kw = rem % params.KW;
                uint c = cBase + ci;
                float val = 0.0f;
                if (c < params.Cg)
                {
                    val = wBuf[((o * params.Cg + c) * params.KH + kh) * params.KW + kw];
                }
                sharedW[ci][kh][kw] = val;
            }

            GroupMemoryBarrierWithGroupSync();

            if (isActive)
            {
                for (uint ci = 0; ci < cEnd; ci++)
                {
                    for (uint kh = 0; kh < params.KH; kh++)
                    {
                        int ih = int(oh * params.strideY) - int(params.paddingTop) + int(kh * params.dilationY);
                        if (ih < 0 || ih >= int(params.H))
                            continue;
                        for (uint kw = 0; kw < params.KW; kw++)
                        {
                            int iwReal = int(ow * params.strideX) - int(params.paddingLeft) + int(kw * params.dilationX);
                            if (iwReal < 0 || iwReal >= int(params.W))
                                continue;
                            uint tileIwOffset = localId.x * params.strideX + kw * params.dilationX;
                            sum += sharedX[ci][kh][tileIwOffset] * sharedW[ci][kh][kw];
                        }
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }
    else if (isActive)
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
        outputBuf[(no * params.outH + oh) * params.outW + ow] = sum;
    }
}

// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// Generic NumPy/ONNX-style broadcasting for elementwise binary ops.
// `mode` selects add (0) or mul (1).

struct Params
{
    uint count;
    uint rank;
    uint mode;
    uint pad0;
    uint shape0, shape1, shape2, shape3, shape4, shape5, shape6, shape7;
    uint strideA0, strideA1, strideA2, strideA3, strideA4, strideA5, strideA6, strideA7;
    uint strideB0, strideB1, strideB2, strideB3, strideB4, strideB5, strideB6, strideB7;
};

StructuredBuffer<float> aBuf : register(t0);
StructuredBuffer<float> bBuf : register(t1);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;

    uint flat = idx;
    uint aIdx = 0u;
    uint bIdx = 0u;

    if (params.rank > 7u)
    {
        uint coord = flat % params.shape7;
        flat /= params.shape7;
        aIdx += coord * params.strideA7;
        bIdx += coord * params.strideB7;
    }
    if (params.rank > 6u)
    {
        uint coord = flat % params.shape6;
        flat /= params.shape6;
        aIdx += coord * params.strideA6;
        bIdx += coord * params.strideB6;
    }
    if (params.rank > 5u)
    {
        uint coord = flat % params.shape5;
        flat /= params.shape5;
        aIdx += coord * params.strideA5;
        bIdx += coord * params.strideB5;
    }
    if (params.rank > 4u)
    {
        uint coord = flat % params.shape4;
        flat /= params.shape4;
        aIdx += coord * params.strideA4;
        bIdx += coord * params.strideB4;
    }
    if (params.rank > 3u)
    {
        uint coord = flat % params.shape3;
        flat /= params.shape3;
        aIdx += coord * params.strideA3;
        bIdx += coord * params.strideB3;
    }
    if (params.rank > 2u)
    {
        uint coord = flat % params.shape2;
        flat /= params.shape2;
        aIdx += coord * params.strideA2;
        bIdx += coord * params.strideB2;
    }
    if (params.rank > 1u)
    {
        uint coord = flat % params.shape1;
        flat /= params.shape1;
        aIdx += coord * params.strideA1;
        bIdx += coord * params.strideB1;
    }
    if (params.rank > 0u)
    {
        uint coord = flat % params.shape0;
        flat /= params.shape0;
        aIdx += coord * params.strideA0;
        bIdx += coord * params.strideB0;
    }

    float a = aBuf[aIdx];
    float b = bBuf[bIdx];
    outputBuf[idx] = params.mode == 0u ? a + b : a * b;
}

// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// Int8 → Float32 per-tensor dequantization. Input is a ByteAddressBuffer
// holding packed int8 bytes; each thread reads the 32-bit word containing its
// byte, extracts it, sign-extends, and dequantizes.

struct Params
{
    uint count;
    float scale;
    int zeroPoint;
    uint pad0;
};

ByteAddressBuffer inBuf : register(t0);
RWStructuredBuffer<float> outBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;
    uint wordAddr = (idx / 4u) * 4u;
    uint word = inBuf.Load(wordAddr);
    uint byteIdx = idx % 4u;
    int b = int((word >> (byteIdx * 8u)) & 0xFFu);
    if (b > 127)
        b -= 256;
    outBuf[idx] = (float(b) - float(params.zeroPoint)) * params.scale;
}

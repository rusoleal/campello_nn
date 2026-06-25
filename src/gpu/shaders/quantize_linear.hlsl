// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.
// Float32 → Int8 per-tensor quantization. Uses RWByteAddressBuffer so the
// output can be addressed by byte. Each thread ORs its byte into the correct
// position of a 32-bit word; this assumes the output buffer has been zeroed
// beforehand (or is freshly created and cleared) and is safe because each
// byte position is written by exactly one thread and the bit masks are disjoint.

struct Params
{
    uint count;
    float scale;
    int zeroPoint;
    uint pad0;
};

StructuredBuffer<float> inBuf : register(t0);
RWByteAddressBuffer outBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;
    float x = inBuf[idx];
    int q = int(round(x / params.scale)) + params.zeroPoint;
    q = clamp(q, -128, 127);
    uint byteIdx = idx % 4u;
    uint wordAddr = (idx / 4u) * 4u;
    uint mask = (uint)(q) << (byteIdx * 8u);
    uint original;
    outBuf.InterlockedOr(wordAddr, mask, original);
}

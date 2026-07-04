// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<T> aBuf : register(t0);
StructuredBuffer<T> bBuf : register(t1);
RWStructuredBuffer<T> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint base = groupId.x * 4u;
    if (base >= params.count)
        return;

    uint end = min(base + 4u, params.count);
    if (end - base == 4u)
    {
        T4 a = T4(aBuf[base], aBuf[base + 1u], aBuf[base + 2u], aBuf[base + 3u]);
        T4 b = T4(bBuf[base], bBuf[base + 1u], bBuf[base + 2u], bBuf[base + 3u]);
        T4 v = a * b;
        outputBuf[base]      = v.x;
        outputBuf[base + 1u] = v.y;
        outputBuf[base + 2u] = v.z;
        outputBuf[base + 3u] = v.w;
    }
    else
    {
        for (uint i = base; i < end; ++i)
            outputBuf[i] = aBuf[i] * bBuf[i];
    }
}

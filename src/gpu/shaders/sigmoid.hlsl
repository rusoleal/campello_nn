// DirectX (HLSL). Written but unverified — see relu.hlsl's comment.

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<T> inputBuf : register(t0);
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
        T4 v = T4(inputBuf[base], inputBuf[base + 1u],
                  inputBuf[base + 2u], inputBuf[base + 3u]);
        T4 one = T4(T(1.0));
        v = one / (one + exp(-v));
        outputBuf[base]      = v.x;
        outputBuf[base + 1u] = v.y;
        outputBuf[base + 2u] = v.z;
        outputBuf[base + 3u] = v.w;
    }
    else
    {
        for (uint i = base; i < end; ++i)
            outputBuf[i] = 1.0f / (1.0f + exp(-inputBuf[i]));
    }
}

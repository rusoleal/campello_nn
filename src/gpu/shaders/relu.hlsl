// DirectX (HLSL) body — compiled with a dtype preamble by compile_gpu_shaders.py.
// Preamble supplies:
//   T, T2, T4 aliases and dtype_common.hlsl.inc

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
        v = max(v, T4(T(0.0)));
        outputBuf[base]      = v.x;
        outputBuf[base + 1u] = v.y;
        outputBuf[base + 2u] = v.z;
        outputBuf[base + 3u] = v.w;
    }
    else
    {
        for (uint i = base; i < end; ++i)
            outputBuf[i] = max(inputBuf[i], T(0.0));
    }
}

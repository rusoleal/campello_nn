// DirectX (HLSL). Written against real D3D12/HLSL semantics but unverified —
// no Windows toolchain available to compile or run this from the machine
// that wrote it. Same documented-but-unverified treatment already accepted
// for the DirectML backend (see TODO.md). One workgroup per output element,
// same model as relu.comp/relu.metal — numthreads(1,1,1) means genuinely one
// thread per group here, so no thread-0 gate needed (unlike relu.metal).

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float> inputBuf : register(t0);
RWStructuredBuffer<float> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint idx = groupId.x;
    if (idx >= params.count)
        return;
    outputBuf[idx] = max(inputBuf[idx], 0.0f);
}

#include <metal_stdlib>
using namespace metal;

// Metal. Float32 → Int8 per-tensor quantization. `device char*` gives
// byte-addressed output storage. Thread-0 gate for the same reason as
// relu.metal (campello_gpu's dispatchWorkgroups() uses the pipeline's
// threadExecutionWidth as the per-group thread count).

struct Params
{
    uint count;
    float scale;
    int zeroPoint;
    uint pad0;
};

kernel void computeMain(const device float *inBuf [[buffer(0)]],
                         device char *outBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    if (idx >= params.count)
        return;
    float x = inBuf[idx];
    int q = int(round(x / params.scale)) + params.zeroPoint;
    q = clamp(q, -128, 127);
    outBuf[idx] = char(q);
}

#include <metal_stdlib>
using namespace metal;

// Metal. Int8 → Float32 per-tensor dequantization. `device char*` gives
// byte-addressed input storage. Thread-0 gate for the same reason as
// relu.metal.

struct Params
{
    uint count;
    float scale;
    int zeroPoint;
    uint pad0;
};

kernel void computeMain(const device char *inBuf [[buffer(0)]],
                         device float *outBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    if (idx >= params.count)
        return;
    int x = int(inBuf[idx]);
    outBuf[idx] = (float(x) - float(params.zeroPoint)) * params.scale;
}

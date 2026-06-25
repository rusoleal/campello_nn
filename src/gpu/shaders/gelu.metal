#include <metal_stdlib>
using namespace metal;

// Metal. See relu.metal's comment for the thread-0 gate rationale, and
// gelu.comp's comment for why this needs a manual erf approximation
// (metal_stdlib has no native erf — verified directly against the installed
// Metal toolchain headers, not assumed).

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

static inline float erfApprox(float x)
{
    float s = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p = 0.3275911f;
    float t = 1.0f / (1.0f + p * ax);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return s * y;
}

kernel void computeMain(const device float *inputBuf [[buffer(0)]],
                         device float *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    if (idx >= params.count)
        return;
    float v = inputBuf[idx];
    outputBuf[idx] = 0.5f * v * (1.0f + erfApprox(v * 0.70710678118654752f));
}

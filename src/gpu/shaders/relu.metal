#include <metal_stdlib>
using namespace metal;

// Metal. campello_gpu's ComputePassEncoder::dispatchWorkgroups() (see
// src/metal/compute_pass_encoder.cpp) always dispatches
// currentPipeline->threadExecutionWidth() threads per threadgroup — not a
// value this shader declares or controls, and there's no public
// campello_gpu API to query it before dispatch. So every threadgroup here
// will genuinely have multiple threads (typically the GPU's SIMD width),
// unlike relu.comp's Vulkan local_size_x=1. Gate to thread 0 so only one of
// those threads does the work, keeping the "one workgroup = one output
// element" model correct regardless of the actual (unqueryable) group size.

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

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
    outputBuf[idx] = max(inputBuf[idx], 0.0f);
}

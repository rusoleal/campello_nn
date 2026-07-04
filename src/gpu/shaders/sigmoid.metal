// Metal. See relu.metal's comment for the thread-0 gate rationale.

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void computeMain(const device T *inputBuf [[buffer(0)]],
                         device T *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint base = groupId * 4u;
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
        {
            float v = TO_FLOAT(inputBuf[i]);
            outputBuf[i] = TO_T(1.0f / (1.0f + exp(-v)));
        }
    }
}

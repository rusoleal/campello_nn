// Metal. See relu.metal's comment for why the thread-0 gate is needed here
// (campello_gpu's dispatchWorkgroups() always uses multiple threads per
// threadgroup on this backend, not a value this shader controls).

struct Params
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void computeMain(const device T *aBuf [[buffer(0)]],
                         const device T *bBuf [[buffer(1)]],
                         device T *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
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
        T4 a = T4(aBuf[base], aBuf[base + 1u], aBuf[base + 2u], aBuf[base + 3u]);
        T4 b = T4(bBuf[base], bBuf[base + 1u], bBuf[base + 2u], bBuf[base + 3u]);
        T4 v = a + b;
        outputBuf[base]      = v.x;
        outputBuf[base + 1u] = v.y;
        outputBuf[base + 2u] = v.z;
        outputBuf[base + 3u] = v.w;
    }
    else
    {
        for (uint i = base; i < end; ++i)
            outputBuf[i] = aBuf[i] + bBuf[i];
    }
}

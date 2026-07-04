// Metal. Same plain one-workgroup-per-element model as batchnorm.comp —
// see that file's comment. Thread-0 gate for the same reason as relu.metal.

struct Params
{
    uint count;
    uint C;
    uint spatial;
    float eps;
};

kernel void computeMain(const device T *xBuf [[buffer(0)]],
                         const device T *meanBuf [[buffer(1)]],
                         const device T *varBuf [[buffer(2)]],
                         const device T *scaleBuf [[buffer(3)]],
                         const device T *biasBuf [[buffer(4)]],
                         device T *outputBuf [[buffer(5)]],
                         constant Params &params [[buffer(6)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint idx = groupId;
    if (idx >= params.count)
        return;
    uint c = (idx / params.spatial) % params.C;
    float invStd = 1.0f / sqrt(TO_FLOAT(varBuf[c]) + params.eps);
    float x = TO_FLOAT(xBuf[idx]);
    float m = TO_FLOAT(meanBuf[c]);
    float s = TO_FLOAT(scaleBuf[c]);
    float b = TO_FLOAT(biasBuf[c]);
    outputBuf[idx] = TO_T((x - m) * invStd * s + b);
}

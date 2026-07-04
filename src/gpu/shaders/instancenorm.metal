// Metal. Same row(=plane)-per-workgroup two-pass model as
// instancenorm.comp — see that file's comment. Thread-0 gate for the same
// reason as relu.metal.

struct Params
{
    uint spatial;
    uint C;
    float eps;
    uint pad0;
};

kernel void computeMain(const device T *xBuf [[buffer(0)]],
                         const device T *scaleBuf [[buffer(1)]],
                         const device T *biasBuf [[buffer(2)]],
                         device T *outputBuf [[buffer(3)]],
                         constant Params &params [[buffer(4)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint row = groupId;
    uint c = row % params.C;
    uint base = row * params.spatial;

    float mean = 0.0f;
    for (uint k = 0; k < params.spatial; k++)
        mean += TO_FLOAT(xBuf[base + k]);
    mean /= float(params.spatial);

    float var = 0.0f;
    for (uint k = 0; k < params.spatial; k++)
    {
        float d = TO_FLOAT(xBuf[base + k]) - mean;
        var += d * d;
    }
    var /= float(params.spatial);
    float invStd = 1.0f / sqrt(var + params.eps);

    float s = TO_FLOAT(scaleBuf[c]);
    float b = TO_FLOAT(biasBuf[c]);
    for (uint k = 0; k < params.spatial; k++)
    {
        float x = TO_FLOAT(xBuf[base + k]);
        outputBuf[base + k] = TO_T((x - mean) * invStd * s + b);
    }
}

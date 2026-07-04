// Metal. Same row-per-workgroup dispatch model as layernorm.comp. Thread-0
// gate for the same reason as relu.metal.

struct Params
{
    uint lastDim;
    float eps;
    uint pad1;
    uint pad2;
};

inline float4 toFloat4(T4 v)
{
    return float4(TO_FLOAT(v.x), TO_FLOAT(v.y), TO_FLOAT(v.z), TO_FLOAT(v.w));
}

inline T4 toT4(float4 v)
{
    return T4(TO_T(v.x), TO_T(v.y), TO_T(v.z), TO_T(v.w));
}

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
    uint base = row * params.lastDim;
    uint lastDim = params.lastDim;

    float4 acc = float4(0.0f);
    uint k = 0u;
    for (; k + 3u < lastDim; k += 4u)
    {
        T4 v = T4(xBuf[base + k], xBuf[base + k + 1u],
                  xBuf[base + k + 2u], xBuf[base + k + 3u]);
        acc += toFloat4(v);
    }
    float mean = acc.x + acc.y + acc.z + acc.w;
    for (; k < lastDim; ++k)
        mean += TO_FLOAT(xBuf[base + k]);
    mean /= float(lastDim);

    acc = float4(0.0f);
    k = 0u;
    for (; k + 3u < lastDim; k += 4u)
    {
        T4 v = T4(xBuf[base + k], xBuf[base + k + 1u],
                  xBuf[base + k + 2u], xBuf[base + k + 3u]);
        float4 d = toFloat4(v) - mean;
        acc += d * d;
    }
    float var = acc.x + acc.y + acc.z + acc.w;
    for (; k < lastDim; ++k)
    {
        float d = TO_FLOAT(xBuf[base + k]) - mean;
        var += d * d;
    }
    var /= float(lastDim);
    float invStd = 1.0f / sqrt(var + params.eps);

    k = 0u;
    for (; k + 3u < lastDim; k += 4u)
    {
        T4 xv = T4(xBuf[base + k], xBuf[base + k + 1u],
                   xBuf[base + k + 2u], xBuf[base + k + 3u]);
        T4 sv = T4(scaleBuf[k], scaleBuf[k + 1u],
                   scaleBuf[k + 2u], scaleBuf[k + 3u]);
        T4 bv = T4(biasBuf[k], biasBuf[k + 1u],
                   biasBuf[k + 2u], biasBuf[k + 3u]);
        float4 x = toFloat4(xv);
        float4 s = toFloat4(sv);
        float4 b = toFloat4(bv);
        float4 y = (x - mean) * invStd * s + b;
        T4 res = toT4(y);
        outputBuf[base + k]         = res.x;
        outputBuf[base + k + 1u]    = res.y;
        outputBuf[base + k + 2u]    = res.z;
        outputBuf[base + k + 3u]    = res.w;
    }
    for (; k < lastDim; ++k)
    {
        float x = TO_FLOAT(xBuf[base + k]);
        float s = TO_FLOAT(scaleBuf[k]);
        float b = TO_FLOAT(biasBuf[k]);
        outputBuf[base + k] = TO_T((x - mean) * invStd * s + b);
    }
}

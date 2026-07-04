// Metal. Same row-per-workgroup dispatch model as rmsnorm.comp. Thread-0 gate
// for the same reason as relu.metal.

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
                         device T *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
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
        float4 f = toFloat4(v);
        acc += f * f;
    }
    float meanSquare = acc.x + acc.y + acc.z + acc.w;
    for (; k < lastDim; ++k)
    {
        float v = TO_FLOAT(xBuf[base + k]);
        meanSquare += v * v;
    }
    meanSquare /= float(lastDim);
    float invRms = 1.0f / sqrt(meanSquare + params.eps);

    k = 0u;
    for (; k + 3u < lastDim; k += 4u)
    {
        T4 xv = T4(xBuf[base + k], xBuf[base + k + 1u],
                   xBuf[base + k + 2u], xBuf[base + k + 3u]);
        T4 sv = T4(scaleBuf[k], scaleBuf[k + 1u],
                   scaleBuf[k + 2u], scaleBuf[k + 3u]);
        float4 x = toFloat4(xv);
        float4 s = toFloat4(sv);
        float4 y = x * invRms * s;
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
        outputBuf[base + k] = TO_T(x * invRms * s);
    }
}

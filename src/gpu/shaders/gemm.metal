// Metal. See relu.metal's comment for the thread-0 gate rationale, and
// gemm.comp's comment for the alpha/beta/C-broadcast scope.

struct Params
{
    uint m, k, n, cElems;
    float alpha, beta;
};

inline float4 toFloat4(T4 v)
{
    return float4(TO_FLOAT(v.x), TO_FLOAT(v.y), TO_FLOAT(v.z), TO_FLOAT(v.w));
}

kernel void computeMain(const device T *aBuf [[buffer(0)]],
                         const device T *bBuf [[buffer(1)]],
                         const device T *cBuf [[buffer(2)]],
                         device T *outputBuf [[buffer(3)]],
                         constant Params &params [[buffer(4)]],
                         uint2 groupId [[threadgroup_position_in_grid]],
                         uint2 localId [[thread_position_in_threadgroup]])
{
    if (localId.x != 0)
        return;
    uint n = groupId.x;
    uint m = groupId.y;
    if (m >= params.m || n >= params.n)
        return;

    uint aRow = m * params.k;
    uint nVal = n;
    uint kDim = params.k;
    uint nDim = params.n;

    float sum = 0.0f;
    uint i = 0u;
    for (; i + 3u < kDim; i += 4u)
    {
        T4 a = T4(aBuf[aRow + i],
                  aBuf[aRow + i + 1u],
                  aBuf[aRow + i + 2u],
                  aBuf[aRow + i + 3u]);
        float b0 = TO_FLOAT(bBuf[(i + 0u) * nDim + nVal]);
        float b1 = TO_FLOAT(bBuf[(i + 1u) * nDim + nVal]);
        float b2 = TO_FLOAT(bBuf[(i + 2u) * nDim + nVal]);
        float b3 = TO_FLOAT(bBuf[(i + 3u) * nDim + nVal]);
        sum += dot(toFloat4(a), float4(b0, b1, b2, b3));
    }
    for (; i < kDim; ++i)
        sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[i * nDim + nVal]);

    float cv = params.cElems == 1 ? TO_FLOAT(cBuf[0])
               : params.cElems == params.n ? TO_FLOAT(cBuf[nVal])
                                            : TO_FLOAT(cBuf[m * nDim + nVal]);
    outputBuf[m * nDim + nVal] = TO_T(params.alpha * sum + params.beta * cv);
}

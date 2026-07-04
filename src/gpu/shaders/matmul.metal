// Metal. Batched matmul with 1D column tiling. See matmul.comp.
// campello_gpu's Metal dispatch uses the pipeline's threadExecutionWidth as
// the threadgroup size, so we gate to the first tileWidth threads and have
// each compute one column of the tile. tileWidth is passed at runtime.

struct Params
{
    uint m;
    uint k;
    uint n;
    uint batchCount;
    uint tileWidth;
};

inline float4 toFloat4(T4 v)
{
    return float4(TO_FLOAT(v.x), TO_FLOAT(v.y), TO_FLOAT(v.z), TO_FLOAT(v.w));
}

kernel void computeMain(const device T *aBuf [[buffer(0)]],
                         const device T *bBuf [[buffer(1)]],
                         device T *outputBuf [[buffer(2)]],
                         constant Params &params [[buffer(3)]],
                         uint3 groupId [[threadgroup_position_in_grid]],
                         uint3 localId [[thread_position_in_threadgroup]])
{
    if (localId.x >= params.tileWidth)
        return;
    uint n = groupId.x * params.tileWidth + localId.x;
    uint m = groupId.y;
    uint batch = groupId.z;
    if (m >= params.m || n >= params.n || batch >= params.batchCount)
        return;

    uint mn = params.m * params.n;
    uint mk = params.m * params.k;
    uint kn = params.k * params.n;
    uint aOffset = batch * mk;
    uint bOffset = batch * kn;
    uint outOffset = batch * mn;

    uint aRow = aOffset + m * params.k;
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
        float b0 = TO_FLOAT(bBuf[bOffset + (i + 0u) * nDim + nVal]);
        float b1 = TO_FLOAT(bBuf[bOffset + (i + 1u) * nDim + nVal]);
        float b2 = TO_FLOAT(bBuf[bOffset + (i + 2u) * nDim + nVal]);
        float b3 = TO_FLOAT(bBuf[bOffset + (i + 3u) * nDim + nVal]);
        sum += dot(toFloat4(a), float4(b0, b1, b2, b3));
    }
    for (; i < kDim; ++i)
        sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[bOffset + i * nDim + nVal]);

    outputBuf[outOffset + m * nDim + nVal] = TO_T(sum);
}

// Metal. Grouped-query batched matmul. See gqa_matmul.comp for the full
// description; this mirrors matmul.metal's 1D column tiling.
//
// `a` is [batchA, m, k]. `b` is [batchB, n, k] if transposeB (out = a @ b^T,
// each b row read contiguously over k) or [batchB, k, n] otherwise (out = a @ b).
// batchA must be an exact multiple of batchB; groupSize = batchA / batchB, and
// attention head `h` reads shared KV head `h / groupSize` — the whole point of
// this op is that `b` is never physically replicated up to batchA rows.

struct Params
{
    uint m;
    uint k;
    uint n;
    uint batchA;
    uint groupSize;
    uint transposeB;
    uint tileWidth;
};

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
    if (m >= params.m || n >= params.n || batch >= params.batchA)
        return;

    uint kDim = params.k;
    uint nDim = params.n;
    uint kv = batch / params.groupSize;

    uint aRow = batch * params.m * kDim + m * kDim;
    uint bBase = kv * kDim * nDim;

    float sum = 0.0f;
    if (params.transposeB != 0u)
    {
        // b row [n, :] is contiguous over k.
        uint bRow = bBase + n * kDim;
        for (uint i = 0u; i < kDim; ++i)
            sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[bRow + i]);
    }
    else
    {
        for (uint i = 0u; i < kDim; ++i)
            sum += TO_FLOAT(aBuf[aRow + i]) * TO_FLOAT(bBuf[bBase + i * nDim + n]);
    }

    outputBuf[batch * params.m * nDim + m * nDim + n] = TO_T(sum);
}

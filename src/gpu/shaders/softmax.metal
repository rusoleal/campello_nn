// Metal. Same row-per-workgroup dispatch model and generic per-dim decode as
// softmax.comp. Thread-0 gate for the same reason as relu.metal.

struct Params
{
    uint outerRank, axisSize, axisStride, outerTotal;
    uint divisor0, divisor1, divisor2, divisor3, divisor4, divisor5, divisor6, divisor7;
    uint origStride0, origStride1, origStride2, origStride3, origStride4, origStride5, origStride6, origStride7;
};

inline float4 toFloat4(T4 v)
{
    return float4(TO_FLOAT(v.x), TO_FLOAT(v.y), TO_FLOAT(v.z), TO_FLOAT(v.w));
}

inline T4 toT4(float4 v)
{
    return T4(TO_T(v.x), TO_T(v.y), TO_T(v.z), TO_T(v.w));
}

kernel void computeMain(const device T *inputBuf [[buffer(0)]],
                         device T *outputBuf [[buffer(1)]],
                         constant Params &params [[buffer(2)]],
                         uint groupId [[threadgroup_position_in_grid]],
                         uint localId [[thread_position_in_threadgroup]])
{
    if (localId != 0)
        return;
    uint outerIdx = groupId;
    if (outerIdx >= params.outerTotal)
        return;

    uint divisor[8] = {params.divisor0, params.divisor1, params.divisor2, params.divisor3,
                        params.divisor4, params.divisor5, params.divisor6, params.divisor7};
    uint origStride[8] = {params.origStride0, params.origStride1, params.origStride2, params.origStride3,
                           params.origStride4, params.origStride5, params.origStride6, params.origStride7};

    uint remaining = outerIdx;
    uint base = 0;
    for (uint d = 0; d < params.outerRank; d++)
    {
        uint idx = remaining / divisor[d];
        remaining -= idx * divisor[d];
        base += idx * origStride[d];
    }

    uint axisStride = params.axisStride;

    float maxV = TO_FLOAT(inputBuf[base]);
    uint k = 1u;
    for (; k + 3u < params.axisSize; k += 4u)
    {
        T4 v = T4(inputBuf[base + k * axisStride],
                  inputBuf[base + (k + 1u) * axisStride],
                  inputBuf[base + (k + 2u) * axisStride],
                  inputBuf[base + (k + 3u) * axisStride]);
        float4 f = toFloat4(v);
        maxV = max(maxV, max(max(f.x, f.y), max(f.z, f.w)));
    }
    for (; k < params.axisSize; ++k)
        maxV = max(maxV, TO_FLOAT(inputBuf[base + k * axisStride]));

    float sum = 0.0f;
    k = 0u;
    for (; k + 3u < params.axisSize; k += 4u)
    {
        T4 v = T4(inputBuf[base + k * axisStride],
                  inputBuf[base + (k + 1u) * axisStride],
                  inputBuf[base + (k + 2u) * axisStride],
                  inputBuf[base + (k + 3u) * axisStride]);
        float4 e = exp(toFloat4(v) - maxV);
        T4 res = toT4(e);
        outputBuf[base + k * axisStride]             = res.x;
        outputBuf[base + (k + 1u) * axisStride]      = res.y;
        outputBuf[base + (k + 2u) * axisStride]      = res.z;
        outputBuf[base + (k + 3u) * axisStride]      = res.w;
        sum += e.x + e.y + e.z + e.w;
    }
    for (; k < params.axisSize; ++k)
    {
        float e = exp(TO_FLOAT(inputBuf[base + k * axisStride]) - maxV);
        outputBuf[base + k * axisStride] = TO_T(e);
        sum += e;
    }

    k = 0u;
    for (; k + 3u < params.axisSize; k += 4u)
    {
        T4 v = T4(outputBuf[base + k * axisStride],
                  outputBuf[base + (k + 1u) * axisStride],
                  outputBuf[base + (k + 2u) * axisStride],
                  outputBuf[base + (k + 3u) * axisStride]);
        float4 f = toFloat4(v) / sum;
        T4 res = toT4(f);
        outputBuf[base + k * axisStride]             = res.x;
        outputBuf[base + (k + 1u) * axisStride]      = res.y;
        outputBuf[base + (k + 2u) * axisStride]      = res.z;
        outputBuf[base + (k + 3u) * axisStride]      = res.w;
    }
    for (; k < params.axisSize; ++k)
    {
        float v = TO_FLOAT(outputBuf[base + k * axisStride]);
        outputBuf[base + k * axisStride] = TO_T(v / sum);
    }
}

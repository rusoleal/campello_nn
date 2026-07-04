// DirectX (HLSL). Written but unverified — see relu.hlsl's comment. Same
// row-per-workgroup dispatch model as layernorm.comp.

struct Params
{
    uint lastDim;
    float eps;
    uint pad1;
    uint pad2;
};

StructuredBuffer<T> xBuf : register(t0);
StructuredBuffer<T> scaleBuf : register(t1);
StructuredBuffer<T> biasBuf : register(t2);
RWStructuredBuffer<T> outputBuf : register(u0);
cbuffer ParamsCB : register(b0) { Params params; };

float4 toFloat4(T4 v)
{
    return float4(TO_FLOAT(v.x), TO_FLOAT(v.y), TO_FLOAT(v.z), TO_FLOAT(v.w));
}

T4 toT4(float4 v)
{
    return T4(TO_T(v.x), TO_T(v.y), TO_T(v.z), TO_T(v.w));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 groupId : SV_GroupID)
{
    uint row = groupId.x;
    uint base = row * params.lastDim;
    uint lastDim = params.lastDim;

    float4 acc = 0.0f;
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
    mean /= (float)lastDim;

    acc = 0.0f;
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
    var /= (float)lastDim;
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

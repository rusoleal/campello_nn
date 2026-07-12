#pragma once

namespace systems::leal::campello_nn
{

    // Metal compute source for the GgmlQuantizedMatmul custom dispatch inside the
    // MPS backend. Compiled at runtime via -[MTLDevice newLibraryWithSource:].
    inline const char *ggmlQuantizedMatmulMetalSource()
    {
        return R"(
#include <metal_stdlib>
using namespace metal;

static inline float decode_half(uchar2 bits)
{
    return float(as_type<half>(bits));
}

// -----------------------------------------------------------------------------
// Q8_0: 32 elements per block. Bytes: d(2) + qs[32].
// -----------------------------------------------------------------------------
static inline float dequant_q8_0(device const uchar *bytes, uint flatIdx)
{
    const uint kBlockSize = 32u;
    const uint kBlockBytes = 34u;
    uint blockIdx = flatIdx / kBlockSize;
    uint offset = flatIdx % kBlockSize;
    device const uchar *block = bytes + blockIdx * kBlockBytes;
    float d = decode_half(uchar2(block[0], block[1]));
    int q = int(as_type<char>(block[2u + offset]));
    return float(q) * d;
}

// -----------------------------------------------------------------------------
// Q4_0: 32 elements per block. Bytes: d(2) + qs[16] (2 nibbles per byte).
// -----------------------------------------------------------------------------
static inline float dequant_q4_0(device const uchar *bytes, uint flatIdx)
{
    const uint kBlockSize = 32u;
    const uint kBlockBytes = 18u;
    uint blockIdx = flatIdx / kBlockSize;
    uint offset = flatIdx % kBlockSize;
    device const uchar *block = bytes + blockIdx * kBlockBytes;
    float d = decode_half(uchar2(block[0], block[1]));
    uint j = offset % 16u;
    uchar q = block[2u + j];
    uint nibble = (offset < 16u) ? (q & 0x0Fu) : (q >> 4);
    return (float(nibble) - 8.0f) * d;
}

// -----------------------------------------------------------------------------
// Q4_K: 256 elements per super-block. Bytes: d(2) + dmin(2) + scales[12] + qs[128].
// -----------------------------------------------------------------------------
static inline void getScaleMinK4(int j, device const uchar *q, thread uchar *d, thread uchar *m)
{
    if (j < 4)
    {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    }
    else
    {
        *d = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline float dequant_q4_k(device const uchar *bytes, uint flatIdx)
{
    const uint kSuperBlockSize = 256u;
    const uint kSuperBlockBytes = 144u;
    uint sbIdx = flatIdx / kSuperBlockSize;
    uint offset = flatIdx % kSuperBlockSize;
    device const uchar *sb = bytes + sbIdx * kSuperBlockBytes;
    float d = decode_half(uchar2(sb[0], sb[1]));
    float min = decode_half(uchar2(sb[2], sb[3]));
    device const uchar *scales = sb + 4u;
    device const uchar *qs = sb + 16u;
    uint group = offset / 64u;
    uint subGroup = (offset % 64u) / 32u;
    uint l = offset % 32u;
    uchar sc, sm;
    getScaleMinK4(int(group * 2u + subGroup), scales, &sc, &sm);
    float dd = d * float(sc);
    float mm = min * float(sm);
    uchar qv = qs[group * 32u + l];
    uint nibble = (subGroup == 0u) ? (qv & 0x0Fu) : (qv >> 4);
    return dd * float(nibble) - mm;
}

// -----------------------------------------------------------------------------
// Q4_1: 32 elements per block. Bytes: d(2) + m(2) + qs[16] (2 nibbles per byte).
// -----------------------------------------------------------------------------
static inline float dequant_q4_1(device const uchar *bytes, uint flatIdx)
{
    const uint kBlockSize = 32u;
    const uint kBlockBytes = 20u;
    uint blockIdx = flatIdx / kBlockSize;
    uint offset = flatIdx % kBlockSize;
    device const uchar *block = bytes + blockIdx * kBlockBytes;
    float d = decode_half(uchar2(block[0], block[1]));
    float m = decode_half(uchar2(block[2], block[3]));
    uint j = offset % 16u;
    uchar q = block[4u + j];
    uint nibble = (offset < 16u) ? (q & 0x0Fu) : (q >> 4);
    return float(nibble) * d + m;
}

// -----------------------------------------------------------------------------
// Q5_0: 32 elements per block. Bytes: d(2) + qh(4) + qs[16] (2 nibbles + high bit).
// -----------------------------------------------------------------------------
static inline float dequant_q5_0(device const uchar *bytes, uint flatIdx)
{
    const uint kBlockSize = 32u;
    const uint kBlockBytes = 22u;
    uint blockIdx = flatIdx / kBlockSize;
    uint offset = flatIdx % kBlockSize;
    device const uchar *block = bytes + blockIdx * kBlockBytes;
    float d = decode_half(uchar2(block[0], block[1]));
    uint qh = uint(block[2]) | (uint(block[3]) << 8) | (uint(block[4]) << 16) | (uint(block[5]) << 24);
    uint j = offset % 16u;
    uchar q = block[6u + j];
    int x;
    if (offset < 16u)
    {
        uint xh0 = ((qh >> (j + 0u)) << 4u) & 0x10u;
        x = int((uint(q) & 0x0Fu) | xh0) - 16;
    }
    else
    {
        uint xh1 = (qh >> (j + 12u)) & 0x10u;
        x = int((uint(q) >> 4u) | xh1) - 16;
    }
    return float(x) * d;
}

// -----------------------------------------------------------------------------
// Q5_1: 32 elements per block. Bytes: d(2) + m(2) + qh(4) + qs[16].
// -----------------------------------------------------------------------------
static inline float dequant_q5_1(device const uchar *bytes, uint flatIdx)
{
    const uint kBlockSize = 32u;
    const uint kBlockBytes = 24u;
    uint blockIdx = flatIdx / kBlockSize;
    uint offset = flatIdx % kBlockSize;
    device const uchar *block = bytes + blockIdx * kBlockBytes;
    float d = decode_half(uchar2(block[0], block[1]));
    float m = decode_half(uchar2(block[2], block[3]));
    uint qh = uint(block[4]) | (uint(block[5]) << 8) | (uint(block[6]) << 16) | (uint(block[7]) << 24);
    uint j = offset % 16u;
    uchar q = block[8u + j];
    uint v;
    if (offset < 16u)
    {
        uint xh0 = ((qh >> (j + 0u)) << 4u) & 0x10u;
        v = (uint(q) & 0x0Fu) | xh0;
    }
    else
    {
        uint xh1 = (qh >> (j + 12u)) & 0x10u;
        v = (uint(q) >> 4u) | xh1;
    }
    return float(v) * d + m;
}

// -----------------------------------------------------------------------------
// Q8_1: 32 elements per block. Bytes: d(2) + s(2) + qs[32].
// -----------------------------------------------------------------------------
static inline float dequant_q8_1(device const uchar *bytes, uint flatIdx)
{
    const uint kBlockSize = 32u;
    const uint kBlockBytes = 36u;
    uint blockIdx = flatIdx / kBlockSize;
    uint offset = flatIdx % kBlockSize;
    device const uchar *block = bytes + blockIdx * kBlockBytes;
    float d = decode_half(uchar2(block[0], block[1]));
    int q = int(as_type<char>(block[4u + offset]));
    return float(q) * d;
}

// -----------------------------------------------------------------------------
// Q2_K: 256 elements per super-block. Bytes: scales[16] + qs[64] + d(2) + dmin(2).
// -----------------------------------------------------------------------------
static inline float dequant_q2_k(device const uchar *bytes, uint flatIdx)
{
    const uint kSuperBlockSize = 256u;
    const uint kSuperBlockBytes = 16u + 64u + 2u + 2u;
    uint sbIdx = flatIdx / kSuperBlockSize;
    uint offset = flatIdx % kSuperBlockSize;
    device const uchar *sb = bytes + sbIdx * kSuperBlockBytes;
    device const uchar *scales = sb;
    device const uchar *qsBase = sb + 16u;
    float d = decode_half(uchar2(sb[80], sb[81]));
    float dmin = decode_half(uchar2(sb[82], sb[83]));

    uint nBlock = offset / 128u;
    uint withinBlock = offset % 128u;
    uint j = withinBlock / 32u;
    uint withinJ = withinBlock % 32u;
    uint halfSel = withinJ / 16u;
    uint l = withinJ % 16u;

    uint scaleIdx = nBlock * 8u + j * 2u + halfSel;
    uchar sc = scales[scaleIdx];
    float dl = d * float(sc & 0x0Fu);
    float ml = dmin * float(sc >> 4);
    uint shift = j * 2u;
    uint qsIdx = nBlock * 32u + l + halfSel * 16u;
    uint bits = (uint(qsBase[qsIdx]) >> shift) & 3u;
    return dl * float(bits) - ml;
}

// -----------------------------------------------------------------------------
// Q3_K: 256 elements per super-block. Bytes: hmask[32] + qs[64] + scales[12] + d(2).
// -----------------------------------------------------------------------------
static inline int q3kScale(device const uchar *scalesIn, uint idx)
{
    uint word = idx / 4u;
    uint j = idx % 4u;
    uint hi2 = (uint(scalesIn[8u + j]) >> (word * 2u)) & 0x03u;
    uint lo4;
    if (word == 0u)
        lo4 = uint(scalesIn[j]) & 0x0Fu;
    else if (word == 1u)
        lo4 = uint(scalesIn[4u + j]) & 0x0Fu;
    else if (word == 2u)
        lo4 = uint(scalesIn[j]) >> 4u;
    else
        lo4 = uint(scalesIn[4u + j]) >> 4u;
    return int(lo4 | (hi2 << 4u));
}

static inline float dequant_q3_k(device const uchar *bytes, uint flatIdx)
{
    const uint kSuperBlockSize = 256u;
    const uint kSuperBlockBytes = 32u + 64u + 12u + 2u;
    uint sbIdx = flatIdx / kSuperBlockSize;
    uint offset = flatIdx % kSuperBlockSize;
    device const uchar *sb = bytes + sbIdx * kSuperBlockBytes;
    device const uchar *hmask = sb;
    device const uchar *qsBase = sb + 32u;
    device const uchar *scalesIn = sb + 96u;
    float dAll = decode_half(uchar2(sb[108], sb[109]));

    uint nBlock = offset / 128u;
    uint withinBlock = offset % 128u;
    uint j = withinBlock / 32u;
    uint withinJ = withinBlock % 32u;
    uint halfSel = withinJ / 16u;
    uint l = withinJ % 16u;

    uint scaleIdx = nBlock * 8u + j * 2u + halfSel;
    int sc = q3kScale(scalesIn, scaleIdx);
    float dl = dAll * float(sc - 32);

    uint shift = j * 2u;
    uint globalJ = nBlock * 4u + j;
    uchar m = uchar(1u << globalJ);

    uint qsIdx = nBlock * 32u + l + halfSel * 16u;
    uint hmaskIdx = l + halfSel * 16u;

    int q = int((uint(qsBase[qsIdx]) >> shift) & 3u);
    q -= (hmask[hmaskIdx] & m) ? 0 : 4;
    return dl * float(q);
}

// -----------------------------------------------------------------------------
// Q5_K: 256 elements per super-block. Bytes: d(2) + dmin(2) + scales[12] + qh[32] + qs[128].
// -----------------------------------------------------------------------------
static inline float dequant_q5_k(device const uchar *bytes, uint flatIdx)
{
    const uint kSuperBlockSize = 256u;
    const uint kSuperBlockBytes = 2u + 2u + 12u + 32u + 128u;
    uint sbIdx = flatIdx / kSuperBlockSize;
    uint offset = flatIdx % kSuperBlockSize;
    device const uchar *sb = bytes + sbIdx * kSuperBlockBytes;
    float d = decode_half(uchar2(sb[0], sb[1]));
    float dmin = decode_half(uchar2(sb[2], sb[3]));
    device const uchar *scales = sb + 4u;
    device const uchar *qh = sb + 16u;
    device const uchar *qlBase = sb + 48u;

    uint outerIdx = offset / 64u;
    uint withinOuter = offset % 64u;
    uint halfSel = withinOuter / 32u;
    uint l = withinOuter % 32u;

    uint is = outerIdx * 2u;
    uchar sc, m;
    getScaleMinK4(int(is + halfSel), scales, &sc, &m);
    float dd = d * float(sc);
    float mm = dmin * float(m);

    device const uchar *ql = qlBase + outerIdx * 32u;
    uchar highBit = uchar(1u << (2u * outerIdx + halfSel));

    uint v;
    if (halfSel == 0u)
        v = uint(ql[l] & 0x0Fu) + ((qh[l] & highBit) ? 16u : 0u);
    else
        v = uint(ql[l] >> 4) + ((qh[l] & highBit) ? 16u : 0u);

    return dd * float(v) - mm;
}

// -----------------------------------------------------------------------------
// Q6_K: 256 elements per super-block. Bytes: ql[128] + qh[64] + scales[16] + d(2).
// -----------------------------------------------------------------------------
static inline float dequant_q6_k(device const uchar *bytes, uint flatIdx)
{
    const uint kSuperBlockSize = 256u;
    const uint kSuperBlockBytes = 128u + 64u + 16u + 2u;
    uint sbIdx = flatIdx / kSuperBlockSize;
    uint offset = flatIdx % kSuperBlockSize;
    device const uchar *sb = bytes + sbIdx * kSuperBlockBytes;
    device const uchar *qlBase = sb;
    device const uchar *qhBase = sb + 128u;
    device const uchar *scalesBase = sb + 192u;
    float d = decode_half(uchar2(sb[208], sb[209]));

    uint ni = offset / 128u;
    uint withinN = offset % 128u;
    uint l = withinN / 4u;
    uint which = withinN % 4u;

    device const uchar *ql = qlBase + ni * 64u;
    device const uchar *qh = qhBase + ni * 32u;
    device const uchar *scales = scalesBase + ni * 8u;
    uint is = l / 16u;

    int qv;
    int scaleIdx;
    if (which == 0u)
    {
        qv = int((ql[l] & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4u));
        scaleIdx = int(is);
    }
    else if (which == 1u)
    {
        qv = int((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4u));
        scaleIdx = int(is) + 2;
    }
    else if (which == 2u)
    {
        qv = int((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4u));
        scaleIdx = int(is) + 4;
    }
    else
    {
        qv = int((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4u));
        scaleIdx = int(is) + 6;
    }

    int q = qv - 32;
    int scale = int(as_type<char>(scales[scaleIdx]));
    return d * float(scale) * float(q);
}

// -----------------------------------------------------------------------------
// Q8_K: 256 elements per super-block. Bytes: d(4, float32) + qs[256] + bsums[16]*2.
// -----------------------------------------------------------------------------
static inline float dequant_q8_k(device const uchar *bytes, uint flatIdx)
{
    const uint kSuperBlockSize = 256u;
    const uint kSuperBlockBytes = 4u + 256u + 32u;
    uint sbIdx = flatIdx / kSuperBlockSize;
    uint offset = flatIdx % kSuperBlockSize;
    device const uchar *sb = bytes + sbIdx * kSuperBlockBytes;
    uint bits = uint(sb[0]) | (uint(sb[1]) << 8) | (uint(sb[2]) << 16) | (uint(sb[3]) << 24);
    float d = as_type<float>(bits);
    int q = int(as_type<char>(sb[4u + offset]));
    return d * float(q);
}

static inline float dequantizedWeight(device const uchar *bytes, uint ggmlType, uint flatIdx)
{
    switch (ggmlType)
    {
    case 2u:  return dequant_q4_0(bytes, flatIdx);
    case 3u:  return dequant_q4_1(bytes, flatIdx);
    case 6u:  return dequant_q5_0(bytes, flatIdx);
    case 7u:  return dequant_q5_1(bytes, flatIdx);
    case 8u:  return dequant_q8_0(bytes, flatIdx);
    case 9u:  return dequant_q8_1(bytes, flatIdx);
    case 10u: return dequant_q2_k(bytes, flatIdx);
    case 11u: return dequant_q3_k(bytes, flatIdx);
    case 12u: return dequant_q4_k(bytes, flatIdx);
    case 13u: return dequant_q5_k(bytes, flatIdx);
    case 14u: return dequant_q6_k(bytes, flatIdx);
    case 15u: return dequant_q8_k(bytes, flatIdx);
    default:  return 0.0f;
    }
}

kernel void ggmlQuantizedMatmul(
    device const float *activation [[buffer(0)]],
    device const uchar *weightRaw [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint rows = dims.x;   // batchCount * M
    uint K = dims.y;      // inFeatures
    uint N = dims.z;      // outFeatures
    uint ggmlType = dims.w;

    if (gid.x >= rows || gid.y >= N)
        return;

    uint bm = gid.x;
    uint n = gid.y;

    float sum = 0.0f;
    uint weightBase = n * K;
    for (uint k = 0u; k < K; ++k)
    {
        float a = activation[bm * K + k];
        float w = dequantizedWeight(weightRaw, ggmlType, weightBase + k);
        sum += a * w;
    }
    output[bm * N + n] = sum;
}
)";
    }

} // namespace systems::leal::campello_nn

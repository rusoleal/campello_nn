// Compiled with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC) — see
// CMakeLists.txt. Explicit instantiation definitions only; the actual kernel
// bodies live in simd_kernels.hpp (templated on Arch, shared with the
// default-arch instantiations compiled directly into ops.cpp).
#include "simd_kernels.hpp"

#ifdef CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED

namespace systems::leal::campello_nn
{

    template void evalBroadcastBinaryOpImpl<xsimd::fma3<xsimd::avx2>, AddOp>(
        const Node &, std::vector<CpuValue> &, CpuValue &, AddOp);
    template void evalBroadcastBinaryOpImpl<xsimd::fma3<xsimd::avx2>, MulOp>(
        const Node &, std::vector<CpuValue> &, CpuValue &, MulOp);
    template void evalGeluImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalReluImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalSigmoidImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalLayerNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalRmsNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalBatchNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalInstanceNormImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalMatMulImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);
    template void evalGemmImpl<xsimd::fma3<xsimd::avx2>>(
        const Node &, std::vector<CpuValue> &, CpuValue &);

} // namespace systems::leal::campello_nn

#endif // CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED

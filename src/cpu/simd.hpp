#pragma once

#include <cstdint>
#include <xsimd/xsimd.hpp>

namespace systems::leal::campello_nn
{

    // Default architecture: xsimd auto-detects from the compiler's predefined
    // macros, so with no special compile flags this safely targets SSE2 on
    // x86_64 and NEON on AArch64 — both mandatory baseline ISAs for those
    // architectures, so the resulting binary runs on any real device. AVX2
    // would need runtime CPU-feature dispatch to stay safe on non-AVX2 x86_64
    // hardware (xsimd supports this, but it's real added complexity) — left
    // as future work, not done here. See TODO.md.
    using FloatBatch = xsimd::batch<float>;

    constexpr int64_t kSimdWidth = (int64_t)xsimd::simd_type<float>::size;

} // namespace systems::leal::campello_nn

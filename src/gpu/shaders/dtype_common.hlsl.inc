// Shared dtype helpers for HLSL GpuGeneric shaders.
// This file is #include'd after the per-variant preamble has defined T/T2/T4.

#ifndef GPU_DTYPE_COMMON_HLSL
#define GPU_DTYPE_COMMON_HLSL

#define T_LIT(x) T(x)
#define T2_LIT(x, y) T2(x, y)
#define T4_LIT(x, y, z, w) T4(x, y, z, w)

#define TO_FLOAT(v) float(v)
#define TO_T(v) T(v)

#endif // GPU_DTYPE_COMMON_HLSL

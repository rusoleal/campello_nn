// Shared dtype helpers for GLSL/SPIR-V GpuGeneric shaders.
// This file is #include'd after the per-variant preamble has defined T/T2/T4.

#ifndef GPU_DTYPE_COMMON_GLSL
#define GPU_DTYPE_COMMON_GLSL

// Scalar/vector constructors from float literals work for both float and float16_t.
#define T_LIT(x) T(x)
#define T2_LIT(x, y) T2(x, y)
#define T4_LIT(x, y, z, w) T4(x, y, z, w)

// Helper to promote a loaded T value to float for accumulation (norms, matmul, etc.).
#define TO_FLOAT(v) float(v)

// Helper to convert a float accumulator back to T for storage.
#define TO_T(v) T(v)

#endif // GPU_DTYPE_COMMON_GLSL

cmake_minimum_required(VERSION 3.22.1)

project(campello_nn VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build when the user/CI doesn't pick one explicitly
# (single-config generators only — CMAKE_CONFIGURATION_TYPES means a
# multi-config generator like Xcode/Visual Studio, which ignores
# CMAKE_BUILD_TYPE and picks its config per-build instead). Matters more than
# it used to since src/cpu/simd.hpp's xsimd-based kernels (TODO.md's CPU SIMD
# work) rely on inlining to be worth anything at all — measured an unoptimized
# build of the vectorized matmul running *slower* than the plain scalar
# version it replaced (xsimd's abstraction layers don't get inlined away
# without optimization), so silently defaulting to an unoptimized build here
# would be a real regression for anyone following CLAUDE.md's documented
# build command as written (no -DCMAKE_BUILD_TYPE flag).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

include(GNUInstallDirs)
include_directories(inc)

# Threads — backs src/cpu/thread_pool.cpp (CPU backend kernel parallelism). Not
# REQUIRED: Emscripten builds without pthreads enabled compile thread_pool.cpp
# down to nothing (see CAMPELLO_NN_CPU_THREADING_ENABLED in thread_pool.hpp),
# so a missing Threads package there is fine, not a configure error.
find_package(Threads)

# ---------------------------------------------------------------------------
# FlatBuffers — TFLite import (Phase 4b of TODO.md). TFLite's .tflite files
# are FlatBuffers, not protobuf, so the ONNX importer's hand-rolled
# proto_reader.hpp approach doesn't carry over safely: FlatBuffers' vtable +
# byte-offset indirection is much easier to get subtly wrong than protobuf's
# flat tag+value scheme. Fetched the same way GoogleTest/DirectML are
# (FetchContent), but only the header-only `FlatBuffers::FlatBuffers`
# interface target (Table/Vector field-accessor primitives) — no flatc
# codegen step, no generated schema header; src/tflite/tflite_parser.cpp
# hand-writes the field-ID accessors on top of it instead, verified against
# the real schema.fbs (now at google-ai-edge/LiteRT, the old
# tensorflow/tensorflow path moved during the TFLite->LiteRT migration).
# ---------------------------------------------------------------------------
include(FetchContent)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v25.12.19-2026-02-06-03fffb2
)
FetchContent_MakeAvailable(flatbuffers)

# ---------------------------------------------------------------------------
# xsimd — portable SIMD for the CPU backend's hot kernels (src/cpu/simd.hpp).
# Header-only; write each vectorized kernel once against xsimd::batch<float>
# and it compiles to whatever ISA the target's default compiler flags select
# (SSE2 on x86_64, NEON on AArch64 — both mandatory baseline ISAs, so this is
# safe on any real device with zero extra compile flags). Real AVX2+FMA3
# support (8-wide vs. SSE2's 4-wide) is added separately via runtime cpuid
# dispatch — see src/cpu/simd_kernels.hpp/simd_kernels_avx2.cpp and the
# CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED handling below, which compiles the
# AVX2 path into its own translation unit so unconditionally targeting it
# can't crash non-AVX2 x86_64 hardware. xsimd's own
# CMakeLists.txt declares `option(BUILD_TESTS ...)` and `option(BUILD_EXAMPLES
# ...)`, which collide by name with this project's own options of the same
# name (unlike the campello_image BUILD_TESTS collision noted in CLAUDE.md,
# this one isn't harmless: xsimd's own test suite needs `doctest`, which isn't
# set up here, so configuring with -DBUILD_TESTS=ON would otherwise hard-fail
# — confirmed by actually running it). Save/restore both cache variables
# around just this FetchContent call so xsimd always configures with its own
# tests/examples off, without touching this project's own BUILD_TESTS/
# BUILD_EXAMPLES options (declared later in this file) for everything else.
# ---------------------------------------------------------------------------
set(_campello_nn_saved_build_tests "${BUILD_TESTS}")
set(_campello_nn_saved_build_examples "${BUILD_EXAMPLES}")
if(_campello_nn_saved_build_tests STREQUAL "")
    set(_campello_nn_saved_build_tests OFF)
endif()
if(_campello_nn_saved_build_examples STREQUAL "")
    set(_campello_nn_saved_build_examples OFF)
endif()
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG 14.2.0
)
FetchContent_MakeAvailable(xsimd)
set(BUILD_TESTS ${_campello_nn_saved_build_tests} CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES ${_campello_nn_saved_build_examples} CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# campello_gpu — sibling WebGPU-shaped cross-platform GPU library (Metal/
# Vulkan/DirectX12), backing src/gpu/gpu_backend.{hpp,cpp}
# (`DeviceType::GpuGeneric` — see that enum's doc comment). Unlike
# `DeviceType::Gpu` (platform-native: MPSGraph on Apple, DirectML on Windows,
# unimplemented elsewhere), this is the same backend on every platform,
# selected explicitly for benchmarking against the native ones, not as their
# replacement. campello_gpu's own `createComputePipeline`/`dispatchWorkgroups`
# are genuinely implemented across all three of its native backends
# (confirmed by reading its DirectX12 source directly, not just its —
# slightly stale — `shader_module.hpp` doc comment), so this one integration
# reaches macOS/iOS/Windows/Linux/Android. Not built for Emscripten (neither
# campello_gpu's own wasm/WebGPU backend nor this integration is in scope
# here). campello_gpu's own CMakeLists.txt declares the same
# `BUILD_TESTS`/`BUILD_EXAMPLES`/`BUILD_INTEGRATION_TESTS` option names as
# this project — and unlike the campello_image precedent in CLAUDE.md, this
# collision genuinely isn't harmless: campello_gpu's own `tests/CMakeLists.txt`
# references source files via `${CMAKE_SOURCE_DIR}` instead of
# `${CMAKE_CURRENT_SOURCE_DIR}`, which resolves to *this* project's source
# tree under FetchContent, not campello_gpu's own — configuring with
# -DBUILD_TESTS=ON hard-fails with "Cannot find source file" (confirmed by
# actually running it, same discovery process as the xsimd/doctest collision
# above). Same save/restore fix, just for a different root cause.
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(_campello_nn_saved_build_tests2 "${BUILD_TESTS}")
    set(_campello_nn_saved_build_examples2 "${BUILD_EXAMPLES}")
    set(_campello_nn_saved_build_int_tests2 "${BUILD_INTEGRATION_TESTS}")
    if(_campello_nn_saved_build_tests2 STREQUAL "")
        set(_campello_nn_saved_build_tests2 OFF)
    endif()
    if(_campello_nn_saved_build_examples2 STREQUAL "")
        set(_campello_nn_saved_build_examples2 OFF)
    endif()
    if(_campello_nn_saved_build_int_tests2 STREQUAL "")
        set(_campello_nn_saved_build_int_tests2 OFF)
    endif()
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_INTEGRATION_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        campello_gpu
        GIT_REPOSITORY https://github.com/rusoleal/campello_gpu.git
        GIT_TAG v0.13.2
    )
    FetchContent_MakeAvailable(campello_gpu)
    set(BUILD_TESTS ${_campello_nn_saved_build_tests2} CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES ${_campello_nn_saved_build_examples2} CACHE BOOL "" FORCE)
    set(BUILD_INTEGRATION_TESTS ${_campello_nn_saved_build_int_tests2} CACHE BOOL "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Core sources — graph IR, GraphBuilder, Context/Tensor/Graph/Fence, and the
# CPU reference backend. Cross-platform; no accelerator SDK required.
# Each platform .cmake file appends accelerator-backend sources (Phase 3 of
# TODO.md) to this list before calling add_library().
# ---------------------------------------------------------------------------
set(CAMPELLO_NN_CORE_SOURCES
    src/pi/context.cpp
    src/pi/tensor.cpp
    src/pi/graph.cpp
    src/pi/fence.cpp
    src/pi/operand.cpp
    src/pi/graph_builder.cpp
    src/pi/ir_serialization.cpp
    src/pi/graph_cache.cpp
    src/pi/graph_info.cpp
    src/cpu/cpu_backend.cpp
    src/cpu/ops.cpp
    src/cpu/thread_pool.cpp
    src/onnx/onnx_parser.cpp
    src/onnx/onnx_importer.cpp
    src/tflite/tflite_parser.cpp
    src/tflite/tflite_importer.cpp
)

# ---------------------------------------------------------------------------
# AVX2+FMA3 runtime-dispatched kernels (src/cpu/simd_kernels_avx2.cpp) — see
# src/cpu/simd_kernels.hpp for why this needs its own translation unit
# (compiling xsimd::fma3<xsimd::avx2> codegen without -mavx2/-mfma would be
# invalid). x86_64-only: AVX2 is an x86 instruction set, meaningless on
# ARM/other targets, and `-mavx2` would be an unrecognized flag on those
# toolchains. Excluded on Emscripten too (CAMPELLO_NN_CPU_AVX2_DISPATCH_ENABLED
# itself is never defined there either, by simd_kernels.hpp's own check —
# this just avoids compiling the now-unreachable file at all).
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    list(APPEND CAMPELLO_NN_CORE_SOURCES src/cpu/simd_kernels_avx2.cpp)
    if(MSVC)
        set_source_files_properties(src/cpu/simd_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/cpu/simd_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    list(APPEND CAMPELLO_NN_CORE_SOURCES src/gpu/gpu_backend.cpp)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    include(cmake/android.cmake)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    include(cmake/macos.cmake)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    include(cmake/ios.cmake)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    include(cmake/windows.cmake)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    include(cmake/linux.cmake)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Emscripten")
    include(cmake/wasm.cmake)
endif()

target_link_libraries(${PROJECT_NAME} FlatBuffers::FlatBuffers xsimd::xsimd)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND Threads_FOUND)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(${PROJECT_NAME} campello_gpu::campello_gpu)
endif()

add_library(campello_nn::campello_nn ALIAS ${PROJECT_NAME})

# ---------------------------------------------------------------------------
# Tests
#
# BUILD_TESTS=ON             — universal tests (CPU backend, no SDK required).
# BUILD_INTEGRATION_TESTS=ON — accelerator-backend tests (Phase 3+). Implies
#                              BUILD_TESTS=ON.
# ---------------------------------------------------------------------------
option(BUILD_TESTS             "Build unit tests (universal, CPU backend only)" OFF)
option(BUILD_INTEGRATION_TESTS "Build accelerator-backend integration tests"    OFF)

if(BUILD_INTEGRATION_TESTS)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
option(BUILD_EXAMPLES "Build example applications" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples/cpu)
endif()

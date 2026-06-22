cmake_minimum_required(VERSION 3.22.1)

project(campello_nn VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

include(GNUInstallDirs)
include_directories(inc)

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
    src/cpu/cpu_backend.cpp
    src/cpu/ops.cpp
    src/onnx/onnx_parser.cpp
    src/onnx/onnx_importer.cpp
    src/tflite/tflite_parser.cpp
    src/tflite/tflite_importer.cpp
)

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

target_link_libraries(${PROJECT_NAME} FlatBuffers::FlatBuffers)

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

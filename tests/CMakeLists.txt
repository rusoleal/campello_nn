cmake_minimum_required(VERSION 3.22.1)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# Universal tests — exercise the CPU reference backend. No accelerator SDK
# required, so these run on every platform/CI runner.
# ---------------------------------------------------------------------------
add_executable(campello_nn_universal_tests
    universal/test_cpu_ops.cpp
    universal/test_cpu_float16_ops.cpp
    universal/test_cpu_quantization_ops.cpp
    universal/test_cpu_threading.cpp
    universal/test_cpu_simd.cpp
    universal/test_graph_builder_validation.cpp
    universal/test_transformer_block.cpp
    universal/test_onnx_importer.cpp
    universal/test_tflite_importer.cpp
    universal/test_graph_cache.cpp
)
target_include_directories(campello_nn_universal_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
)
target_compile_definitions(campello_nn_universal_tests PRIVATE
    CAMPELLO_NN_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(campello_nn_universal_tests
    campello_nn
    GTest::gtest_main
)
set_target_properties(campello_nn_universal_tests PROPERTIES
    CXX_STANDARD 20
)

# ---------------------------------------------------------------------------
# Integration tests — accelerator backends (TODO.md Phase 3+). Currently just
# the MPSGraph backend on macOS/iOS (DeviceType::Gpu); other platforms have
# nothing to build here yet.
# ---------------------------------------------------------------------------
if(BUILD_INTEGRATION_TESTS)
    if(APPLE)
        add_executable(campello_nn_integration_tests
            platform/test_mps_ops.cpp
            platform/test_mps_float16_ops.cpp
            platform/test_mps_quantization_ops.cpp
            platform/test_mps_graph_cache.cpp
        )
        target_include_directories(campello_nn_integration_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/inc
        )
        target_link_libraries(campello_nn_integration_tests
            campello_nn
            GTest::gtest_main
        )
        set_target_properties(campello_nn_integration_tests PROPERTIES
            CXX_STANDARD 20
        )
        include(GoogleTest)
        gtest_discover_tests(campello_nn_integration_tests)
    elseif(WIN32)
        add_executable(campello_nn_integration_tests
            platform/test_directml_ops.cpp
            platform/test_directml_float16_ops.cpp
            platform/test_directml_quantization_ops.cpp
            platform/test_directml_graph_cache.cpp
        )
        target_include_directories(campello_nn_integration_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/inc
        )
        target_link_libraries(campello_nn_integration_tests
            campello_nn
            GTest::gtest_main
        )
        set_target_properties(campello_nn_integration_tests PROPERTIES
            CXX_STANDARD 20
        )
        # DirectML.dll (the redistributable NuGet version, see cmake/windows.cmake)
        # must sit next to the test executable at runtime.
        add_custom_command(TARGET campello_nn_integration_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CAMPELLO_NN_DIRECTML_DLL}"
                "$<TARGET_FILE_DIR:campello_nn_integration_tests>"
        )
        include(GoogleTest)
        gtest_discover_tests(campello_nn_integration_tests)
    else()
        message(STATUS "campello_nn: BUILD_INTEGRATION_TESTS requested, but no accelerator backend "
                        "is implemented yet for this platform (see TODO.md Phase 3).")
    endif()
endif()

# ---------------------------------------------------------------------------
# GPU generic backend tests — DeviceType::GpuGeneric (src/gpu/gpu_backend.cpp).
# Kept separate from BUILD_INTEGRATION_TESTS's if(APPLE)/elseif(WIN32) shape
# above: unlike MPSGraph/DirectML, this backend is meant to build on every
# platform campello_gpu supports, not gated by OS.
# ---------------------------------------------------------------------------
option(BUILD_GPU_GENERIC_TESTS "Build DeviceType::GpuGeneric (campello_gpu-based) backend tests" OFF)

if(BUILD_GPU_GENERIC_TESTS)
    set(GPU_GENERIC_TEST_SOURCES platform/test_gpu_generic_ops.cpp)
    set(GPU_GENERIC_TEST_LIBS campello_nn GTest::gtest_main)

    # Real-model end-to-end tests need campello_image (same dependency as
    # BUILD_MODEL_TESTS). Share the already-fetched target when available.
    if(BUILD_MODEL_TESTS)
        list(APPEND GPU_GENERIC_TEST_SOURCES platform/test_gpu_generic_models.cpp)
        list(APPEND GPU_GENERIC_TEST_LIBS campello_image)
    endif()

    add_executable(campello_nn_gpu_generic_tests ${GPU_GENERIC_TEST_SOURCES})
    target_include_directories(campello_nn_gpu_generic_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/inc
    )
    target_compile_definitions(campello_nn_gpu_generic_tests PRIVATE
        CAMPELLO_NN_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    )
    target_link_libraries(campello_nn_gpu_generic_tests
        ${GPU_GENERIC_TEST_LIBS}
    )
    set_target_properties(campello_nn_gpu_generic_tests PROPERTIES
        CXX_STANDARD 20
    )
    include(GoogleTest)
    gtest_discover_tests(campello_nn_gpu_generic_tests)
endif()

# ---------------------------------------------------------------------------
# Model tests — exercise real third-party models (YuNet/ONNX and BlazeFace/
# TFLite, both face detectors) end-to-end via their respective importers, on
# real decoded images. Pulls in campello_image (FetchContent) to decode the
# test images, which itself pulls in basis_universal + libwebp — opt-in via
# its own flag so a normal `-DBUILD_TESTS=ON` build stays light.
# ---------------------------------------------------------------------------
option(BUILD_MODEL_TESTS "Build the YuNet/BlazeFace face-detection model import tests (pulls in campello_image)" OFF)

if(BUILD_MODEL_TESTS)
    FetchContent_Declare(
        campello_image
        GIT_REPOSITORY https://github.com/rusoleal/campello_image.git
        GIT_TAG v0.5.0
    )
    FetchContent_MakeAvailable(campello_image)

    add_executable(campello_nn_model_tests
        universal/test_yunet_face_detection.cpp
        universal/test_blazeface_face_detection.cpp
    )
    target_include_directories(campello_nn_model_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/inc
    )
    target_compile_definitions(campello_nn_model_tests PRIVATE
        CAMPELLO_NN_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    )
    target_link_libraries(campello_nn_model_tests
        campello_nn
        campello_image
        GTest::gtest_main
    )
    set_target_properties(campello_nn_model_tests PROPERTIES
        CXX_STANDARD 20
    )
    if(WIN32)
        # campello_image is a shared library on Windows (no rpath/install_name
        # trick like macOS/Linux) — it must sit next to the test executable at
        # runtime, same reasoning as DirectML.dll's copy step above.
        add_custom_command(TARGET campello_nn_model_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:campello_image>"
                "$<TARGET_FILE_DIR:campello_nn_model_tests>"
        )
    endif()
    include(GoogleTest)
    gtest_discover_tests(campello_nn_model_tests)
endif()

include(GoogleTest)
gtest_discover_tests(campello_nn_universal_tests)

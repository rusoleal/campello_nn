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
    universal/test_graph_builder_validation.cpp
    universal/test_transformer_block.cpp
    universal/test_onnx_importer.cpp
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
    else()
        message(STATUS "campello_nn: BUILD_INTEGRATION_TESTS requested, but no accelerator backend "
                        "is implemented yet for this platform (see TODO.md Phase 3).")
    endif()
endif()

# ---------------------------------------------------------------------------
# Model tests — exercise a real third-party model (YuNet, an ONNX face
# detector) end-to-end via the ONNX importer, on real decoded images. Pulls in
# campello_image (FetchContent) to decode the test images, which itself pulls
# in basis_universal + libwebp — opt-in via its own flag so a normal
# `-DBUILD_TESTS=ON` build stays light.
# ---------------------------------------------------------------------------
option(BUILD_MODEL_TESTS "Build the YuNet face-detection ONNX import test (pulls in campello_image)" OFF)

if(BUILD_MODEL_TESTS)
    FetchContent_Declare(
        campello_image
        GIT_REPOSITORY https://github.com/rusoleal/campello_image.git
        GIT_TAG v0.5.0
    )
    FetchContent_MakeAvailable(campello_image)

    add_executable(campello_nn_model_tests
        universal/test_yunet_face_detection.cpp
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
    include(GoogleTest)
    gtest_discover_tests(campello_nn_model_tests)
endif()

include(GoogleTest)
gtest_discover_tests(campello_nn_universal_tests)

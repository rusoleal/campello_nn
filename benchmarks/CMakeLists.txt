include(FetchContent)

# The YuNet workload needs campello_image to decode/resize test fixtures.
FetchContent_Declare(
    campello_image
    GIT_REPOSITORY https://github.com/rusoleal/campello_image.git
    GIT_TAG v0.5.0
)
FetchContent_MakeAvailable(campello_image)

# ResNet-50 v1 from the ONNX Model Zoo (~98 MB). Kept out of git (see .gitignore);
# downloaded once at build time when the benchmark is enabled.
set(CAMPELLO_NN_RESNET50_URL
    "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx")
set(CAMPELLO_NN_RESNET50_PATH "${CMAKE_SOURCE_DIR}/tests/fixtures/resnet50-v1-7.onnx")
if(NOT EXISTS "${CAMPELLO_NN_RESNET50_PATH}")
    find_program(CURL_EXECUTABLE curl REQUIRED)
    add_custom_command(
        OUTPUT "${CAMPELLO_NN_RESNET50_PATH}"
        COMMAND "${CURL_EXECUTABLE}" -L -o "${CAMPELLO_NN_RESNET50_PATH}" "${CAMPELLO_NN_RESNET50_URL}"
        COMMENT "Downloading ResNet-50 benchmark fixture (~98 MB)..."
        VERBATIM)
    add_custom_target(campello_nn_resnet50_fixture DEPENDS "${CAMPELLO_NN_RESNET50_PATH}")
endif()

add_executable(campello_nn_benchmark benchmark_backends.cpp)
target_include_directories(campello_nn_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_compile_definitions(campello_nn_benchmark PRIVATE
    CAMPELLO_NN_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(campello_nn_benchmark campello_nn campello_image)
set_target_properties(campello_nn_benchmark PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
)
if(TARGET campello_nn_resnet50_fixture)
    add_dependencies(campello_nn_benchmark campello_nn_resnet50_fixture)
endif()

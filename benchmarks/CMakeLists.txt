include(FetchContent)

# The YuNet workload needs campello_image to decode/resize test fixtures.
FetchContent_Declare(
    campello_image
    GIT_REPOSITORY https://github.com/rusoleal/campello_image.git
    GIT_TAG v0.5.0
)
FetchContent_MakeAvailable(campello_image)

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

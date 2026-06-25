add_executable(campello_nn_benchmark benchmark_backends.cpp)
target_link_libraries(campello_nn_benchmark campello_nn)
set_target_properties(campello_nn_benchmark PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
)

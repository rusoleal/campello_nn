# iOS — mirrors macos.cmake. MPSGraph backend (TODO.md Phase 3a).
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME} STATIC
    ${CAMPELLO_NN_CORE_SOURCES}
    src/metal/mps_backend.mm
)

set_source_files_properties(src/metal/mps_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_link_libraries(${PROJECT_NAME}
    "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
)

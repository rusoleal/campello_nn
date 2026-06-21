# Web/Emscripten — passthrough to native browser WebNN not implemented yet
# (TODO.md Phase 3e). CPU reference backend covers wasm builds for now.

add_library(${PROJECT_NAME} STATIC
    ${CAMPELLO_NN_CORE_SOURCES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
)

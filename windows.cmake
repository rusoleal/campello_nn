# Windows — DirectML backend not implemented yet (TODO.md Phase 3b).
# CPU reference backend covers Windows for now.

add_library(${PROJECT_NAME} STATIC
    ${CAMPELLO_NN_CORE_SOURCES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
)

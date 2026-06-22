# Linux — no official accelerator graph API exists (oneDNN/Vulkan backend
# not implemented yet, TODO.md Phase 3c). CPU reference backend covers Linux
# for now; unlike campello_gpu's Vulkan dependency, this needs no SDK.

add_library(${PROJECT_NAME} STATIC
    ${CAMPELLO_NN_CORE_SOURCES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
)

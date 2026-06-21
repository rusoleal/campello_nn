# Windows — DirectML backend (TODO.md Phase 3b).
#
# DirectML ships as a redistributable NuGet package (headers + import lib +
# DLL), not part of the Windows SDK proper — fetched the same way GoogleTest/
# campello_image are fetched in tests/CMakeLists.txt. The package is a plain
# zip (NuGet's .nupkg is just a renamed zip); DOWNLOAD_NAME forces CMake to
# recognize and auto-extract it despite the extensionless URL.
include(FetchContent)
FetchContent_Declare(
    directml
    URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4
    DOWNLOAD_NAME directml.zip
)
FetchContent_MakeAvailable(directml)

add_library(${PROJECT_NAME} STATIC
    ${CAMPELLO_NN_CORE_SOURCES}
    src/directml/directml_backend.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>"
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "${directml_SOURCE_DIR}/include"
)

target_link_libraries(${PROJECT_NAME}
    d3d12.lib
    dxgi.lib
    "${directml_SOURCE_DIR}/bin/x64-win/DirectML.lib"
)

# DirectML.dll (the redistributable version from the NuGet package, not
# whatever in-box version Windows happens to ship) must sit next to any
# executable that links this library — exposed here so tests/CMakeLists.txt
# and examples/CMakeLists.txt can add a post-build copy step.
set(CAMPELLO_NN_DIRECTML_DLL "${directml_SOURCE_DIR}/bin/x64-win/DirectML.dll")

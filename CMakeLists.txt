cmake_minimum_required(VERSION 3.16)
project(xtpms LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ──────────────────────────────────────────────────────────
# Dependencies
#   conda:  conda env create -f environment.yml && conda activate xtpms
#   vcpkg:  vcpkg install cgal openmesh eigen3 gtest libigl
# ──────────────────────────────────────────────────────────
include(FetchContent)

find_package(Eigen3 CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# OpenMesh: try find_package, fallback to FetchContent
find_package(OpenMesh CONFIG QUIET)
if(NOT TARGET OpenMeshCore)
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        set(_vp "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
        if(WIN32 AND EXISTS "${_vp}/lib/OpenMeshCore.lib")
            add_library(OpenMeshCore SHARED IMPORTED GLOBAL)
            set_target_properties(OpenMeshCore PROPERTIES
                IMPORTED_IMPLIB "${_vp}/lib/OpenMeshCore.lib"
                IMPORTED_LOCATION "${_vp}/bin/OpenMeshCore.dll"
                INTERFACE_INCLUDE_DIRECTORIES "${_vp}/include"
            )
        endif()
    endif()
endif()
if(NOT TARGET OpenMeshCore AND TARGET OpenMesh::OpenMeshCore)
    add_library(OpenMeshCore ALIAS OpenMesh::OpenMeshCore)
endif()
if(NOT TARGET OpenMeshCore)
    message(STATUS "OpenMesh not found, fetching via FetchContent...")
    set(BUILD_APPS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(openmesh
        GIT_REPOSITORY https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
        GIT_TAG        OpenMesh-8.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(openmesh)
endif()

# libigl: try find_package, fallback to local third_party, then FetchContent
find_package(libigl CONFIG QUIET)
if(NOT TARGET igl::core)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/igl/cotmatrix.h")
        message(STATUS "Using local libigl from third_party/")
        add_library(igl_core INTERFACE)
        add_library(igl::core ALIAS igl_core)
        target_include_directories(igl_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    else()
        message(STATUS "libigl not found, fetching via FetchContent...")
        set(LIBIGL_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(libigl
            GIT_REPOSITORY https://github.com/libigl/libigl.git
            GIT_TAG        v2.5.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(libigl)
    endif()
endif()

# ──────────────────────────────────────────────────────────
# xtpms library
# ──────────────────────────────────────────────────────────
add_library(xtpms_headers INTERFACE)
target_include_directories(xtpms_headers INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/xtpms/src"
)
if(WIN32)
    target_compile_definitions(xtpms_headers INTERFACE _USE_MATH_DEFINES)
endif()
if(MSVC)
    target_compile_options(xtpms_headers INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>)
endif()
target_link_libraries(xtpms_headers INTERFACE OpenMeshCore)

add_library(xtpms_core STATIC
    xtpms/src/PeriodicMesh.cpp
    xtpms/src/MarchingCubes.cpp
    xtpms/src/PeriodicRemesh.cpp
    xtpms/src/AsymptoticConductivity.cpp
)
target_link_libraries(xtpms_core PUBLIC xtpms_headers Eigen3::Eigen PRIVATE CGAL::CGAL igl::core)
if(MSVC)
    target_compile_options(xtpms_core PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>)
else()
    target_compile_options(xtpms_core PRIVATE -Wno-deprecated-declarations)
endif()

# ──────────────────────────────────────────────────────────
# CLI (command-line tool)
# ──────────────────────────────────────────────────────────
# CLI11 (header-only argument parser)
find_package(CLI11 CONFIG QUIET)
if(NOT TARGET CLI11::CLI11)
    FetchContent_Declare(cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG        v2.4.2
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(cli11)
endif()

# tinyexpr (expression parser, single file)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyexpr.c")
    set(tinyexpr_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
else()
    FetchContent_Declare(tinyexpr
        GIT_REPOSITORY https://github.com/codeplea/tinyexpr.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(tinyexpr)
    if(NOT tinyexpr_POPULATED)
        FetchContent_Populate(tinyexpr)
    endif()
endif()
add_library(tinyexpr_lib OBJECT "${tinyexpr_SOURCE_DIR}/tinyexpr.c")
target_include_directories(tinyexpr_lib PUBLIC "${tinyexpr_SOURCE_DIR}")
set_source_files_properties("${tinyexpr_SOURCE_DIR}/tinyexpr.c" PROPERTIES LANGUAGE C)

add_executable(xtpms_app xtpms/main.cpp)
set_target_properties(xtpms_app PROPERTIES OUTPUT_NAME "xtpms")
target_link_libraries(xtpms_app PRIVATE xtpms_core CLI11::CLI11 tinyexpr_lib)
if(MSVC)
    target_compile_options(xtpms_app PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>)
endif()

# ──────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────
enable_testing()
include(GoogleTest)

add_executable(test_periodic_spatial_hash tests/test_periodic_spatial_hash.cpp)
target_link_libraries(test_periodic_spatial_hash PRIVATE xtpms_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_periodic_spatial_hash)

add_executable(test_geometry_voxel tests/test_geometry_voxel.cpp)
target_link_libraries(test_geometry_voxel PRIVATE xtpms_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_geometry_voxel)

add_executable(test_periodic_mesh_periodize tests/test_periodic_mesh_periodize.cpp)
target_link_libraries(test_periodic_mesh_periodize PRIVATE xtpms_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_periodic_mesh_periodize)

add_executable(test_aabb_tree tests/test_aabb_tree.cpp)
target_link_libraries(test_aabb_tree PRIVATE CGAL::CGAL GTest::gtest GTest::gtest_main)
target_include_directories(test_aabb_tree PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/xtpms/src")
if(WIN32)
    target_compile_definitions(test_aabb_tree PRIVATE _USE_MATH_DEFINES)
endif()
if(MSVC)
    target_compile_options(test_aabb_tree PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8 /bigobj /FS>)
endif()

add_executable(test_marching_cubes tests/test_marching_cubes.cpp)
target_link_libraries(test_marching_cubes PRIVATE xtpms_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_marching_cubes)

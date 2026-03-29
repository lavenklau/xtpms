cmake_minimum_required(VERSION 3.16)
project(xtpms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Quick configure: .\configure.ps1   (or configure.cmd)
#
# vcpkg + Visual Studio 2019 (Release only, x64), PowerShell:
#   $env:VCPKG_ROOT = 'C:\vcpkg'
#   cmake --preset vs2019-vcpkg
#   cmake --build --preset release
#   ctest -C Release --output-on-failure
# Manual:
#   cmake -B build -S . -G "Visual Studio 16 2019" -A x64 ^
#     -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
#     -DCMAKE_CONFIGURATION_TYPES=Release
#
# vcpkg OpenMesh: shipped OpenMeshConfig.cmake can reference a wrong debug lib path (share/debug/...).
# Define IMPORTED targets from known layout under VCPKG_INSTALLED_DIR when present.
set(_xtpms_vp "")
if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
	set(_xtpms_vp "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
endif()
if(WIN32 AND EXISTS "${_xtpms_vp}/lib/OpenMeshCore.lib" AND NOT TARGET OpenMeshCore)
	add_library(OpenMeshCore SHARED IMPORTED GLOBAL)
	set_target_properties(OpenMeshCore PROPERTIES
		IMPORTED_CONFIGURATIONS "RELEASE"
		IMPORTED_IMPLIB_RELEASE "${_xtpms_vp}/lib/OpenMeshCore.lib"
		IMPORTED_LOCATION_RELEASE "${_xtpms_vp}/bin/OpenMeshCore.dll"
		INTERFACE_INCLUDE_DIRECTORIES "${_xtpms_vp}/include"
	)
endif()
if(NOT TARGET OpenMeshCore)
	find_package(OpenMesh CONFIG REQUIRED)
endif()
if(NOT TARGET OpenMeshCore)
	message(FATAL_ERROR "OpenMesh: expected OpenMeshCore target (vcpkg prefix: ${_xtpms_vp})")
endif()
find_package(GTest CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)

# Header-only library surface (sources under xtpms/src).
add_library(xtpms_headers INTERFACE)
target_include_directories(xtpms_headers INTERFACE
	"${CMAKE_CURRENT_SOURCE_DIR}/xtpms/src"
)
# OpenMesh on Windows expects M_PI from <cmath> via MSVC math defs.
if(WIN32)
	target_compile_definitions(xtpms_headers INTERFACE _USE_MATH_DEFINES)
endif()
# UTF-8 sources + MSVC default code page 936 -> C4819 and broken parsing of DefaultTriMesh etc.
if(MSVC)
	target_compile_options(xtpms_headers INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>)
endif()
# vcpkg OpenMesh exports OpenMeshCore / OpenMeshTools (names may be OpenMesh::OpenMeshCore).
if(TARGET OpenMesh::OpenMeshCore)
	target_link_libraries(xtpms_headers INTERFACE OpenMesh::OpenMeshCore)
elseif(TARGET OpenMeshCore)
	target_link_libraries(xtpms_headers INTERFACE OpenMeshCore)
else()
	message(FATAL_ERROR "OpenMesh CMake targets not found (expected OpenMesh::OpenMeshCore or OpenMeshCore).")
endif()

add_library(xtpms_core STATIC
	xtpms/src/PeriodicMesh.cpp
	xtpms/src/MarchingCubes.cpp
	xtpms/src/PeriodicRemesh.cpp
	xtpms/src/AsymptoticConductivity.cpp
)
target_link_libraries(xtpms_core PUBLIC xtpms_headers Eigen3::Eigen PRIVATE CGAL::CGAL igl::core)
if(MSVC)
	target_compile_options(xtpms_core PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>)
endif()

# CGAL AABB 模块（AABBTree.h）：不依赖 xtpms_headers/OpenMesh，避免测试目标多链一份 DLL 与错误配置库。
add_library(xtpms_aabb_tree INTERFACE)
target_include_directories(xtpms_aabb_tree INTERFACE
	"${CMAKE_CURRENT_SOURCE_DIR}/xtpms/src"
)
if(WIN32)
	target_compile_definitions(xtpms_aabb_tree INTERFACE _USE_MATH_DEFINES)
endif()
if(MSVC)
	target_compile_options(xtpms_aabb_tree INTERFACE
		$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>
		$<$<COMPILE_LANGUAGE:CXX>:/bigobj>
	)
endif()
target_link_libraries(xtpms_aabb_tree INTERFACE CGAL::CGAL)

add_executable(xtpms_cli xtpms/main.cpp)
target_link_libraries(xtpms_cli PRIVATE xtpms_core)

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
# Optional: enforce periodic partner ratio, e.g. target_compile_definitions(... XTPMS_TEST_STL_MIN_PARTNER_RATIO=0.85)
gtest_discover_tests(test_periodic_mesh_periodize)

add_executable(test_aabb_tree tests/test_aabb_tree.cpp)
target_link_libraries(test_aabb_tree PRIVATE xtpms_aabb_tree GTest::gtest GTest::gtest_main)
if(MSVC)
	# 减轻多项目并行生成时 cl 写 PDB / 中间文件竞争导致的 LNK1104（找不到 .obj）
	target_compile_options(test_aabb_tree PRIVATE /FS)
endif()
gtest_discover_tests(test_aabb_tree)

add_executable(test_marching_cubes tests/test_marching_cubes.cpp)
target_link_libraries(test_marching_cubes PRIVATE xtpms_core GTest::gtest GTest::gtest_main)
# Gyroid mesh: <repo>/results/xtpms_marching_cubes_gyroid.obj (independent of process cwd).
target_compile_definitions(test_marching_cubes PRIVATE
	"XTPMS_MC_RESULTS_DIR=R\"(${CMAKE_SOURCE_DIR}/results)\"")
gtest_discover_tests(test_marching_cubes)

cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # minimum CMake version to support Kokkos

# put these before project(...)
include(cmake/util.cmake)
include(cmake/machine.cmake)

set_environment()

project(triton CXX)
set(CMAKE_CXX_STANDARD 17) # minimum C++ standard to support Kokkos

# check if MPI is available
if(MACHINE STREQUAL Windows)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# Find GDAL
find_package(GDAL REQUIRED)
if(GDAL_FOUND)
    message(STATUS "Found GDAL: ${GDAL_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "GDAL not found. Please install libgdal-dev or load it as a module")
endif()


# add external packages
add_subdirectory(${TRITON_SOURCE_DIR}/external)

# build triton
process_target(triton.exe ./src/main.cpp)
target_include_directories(triton.exe PUBLIC ${Kokkos_INCLUDE_DIRS_RET})
target_include_directories(triton.exe PUBLIC ${GDAL_INCLUDE_DIR})
target_link_libraries(triton.exe PUBLIC ${GDAL_LIBRARY})
add_build_and_run_scripts()

# add tools
if(${BUILD_TOOLS})
  add_subdirectory(${TRITON_SOURCE_DIR}/tools)
endif()

# add tests
if(${BUILD_TESTS})
  enable_testing()
  add_subdirectory(${TRITON_SOURCE_DIR}/test)
  add_test_script()
endif()

add_clean_script()

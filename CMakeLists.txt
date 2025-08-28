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

# add external packages
add_subdirectory(${TRITON_SOURCE_DIR}/external)

# build triton
process_target(triton.exe ./src/main.cpp)
target_include_directories(triton.exe PUBLIC ${Kokkos_INCLUDE_DIRS_RET})

if (GDAL_FOUND)
target_include_directories(triton.exe PUBLIC ${GDAL_INCLUDE_DIR})
target_link_libraries(triton.exe PUBLIC ${GDAL_LIBRARY})
endif()

add_build_and_run_scripts()

# add tests
if(${BUILD_TESTS})
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  enable_testing()
  add_subdirectory(${TRITON_SOURCE_DIR}/test)
  add_test_script()
endif()

add_clean_script()

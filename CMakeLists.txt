cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # minimum CMake version to support Kokkos

set(TRITON_EXECUTABLE "triton.exe")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# put these before project(...)
include(cmake/util.cmake)
include(cmake/machine.cmake)

set_environment()

project(triton CXX)
set(CMAKE_CXX_STANDARD 17) # minimum C++ standard to support Kokkos
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")

# check if MPI is available
if(MACHINE STREQUAL Windows)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# SWMM coupling option
option(TRITON_ENABLE_SWMM "Enable SWMM coupling for urban drainage" OFF)

if(TRITON_ENABLE_SWMM)
  # Find SWMM library
  set(SWMM_ROOT_DIR "$ENV{SWMM_ROOT_DIR}" CACHE PATH "SWMM installation directory")
  set(SWMM_INCLUDE_DIR "${SWMM_ROOT_DIR}/include" CACHE PATH "SWMM include directory")
  set(SWMM_LIBRARY_DIR "${SWMM_ROOT_DIR}/lib" CACHE PATH "SWMM library directory")

  # Try to find SWMM headers
  find_path(SWMM_INCLUDE_PATH
    NAMES swmm5.h
    PATHS
      ${SWMM_INCLUDE_DIR}
      ${CMAKE_SOURCE_DIR}/external/swmm/include
      /usr/local/include/swmm
      /usr/include/swmm
      ${PROJECT_SOURCE_DIR}/../Stormwater-Management-Model/src/solver/include
    DOC "SWMM include directory"
  )

  # Try to find SWMM library
  find_library(SWMM_LIBRARY
    NAMES swmm5 swmm
    PATHS
      ${SWMM_LIBRARY_DIR}
      ${CMAKE_SOURCE_DIR}/external/swmm/lib
      /usr/local/lib
      /usr/lib
      ${PROJECT_SOURCE_DIR}/../Stormwater-Management-Model/build/bin
    DOC "SWMM library"
  )

  if(SWMM_INCLUDE_PATH AND SWMM_LIBRARY)
    message(STATUS "SWMM include directory: ${SWMM_INCLUDE_PATH}")
    message(STATUS "SWMM library: ${SWMM_LIBRARY}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRITON_SWMM")
  else()
    message(FATAL_ERROR "SWMM coupling requested but SWMM library/headers not found. "
                       "Please set SWMM_ROOT_DIR or SWMM_INCLUDE_DIR and SWMM_LIBRARY_DIR")
  endif()
endif()

# add external packages
add_subdirectory(${TRITON_SOURCE_DIR}/external)

# build triton
add_executable(${TRITON_EXECUTABLE} ./src/main.cpp)
target_link_libraries(${TRITON_EXECUTABLE} PUBLIC kokkos)

if (ENSEMBLE_BUILD)
  target_link_libraries(${TRITON_EXECUTABLE} PUBLIC yaml-cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENSEMBLE_BUILD")
endif()

if (ENABLE_GDAL)
  target_include_directories(${TRITON_EXECUTABLE} PUBLIC ${GDAL_INCLUDE_DIR})
  target_link_libraries(${TRITON_EXECUTABLE} PUBLIC ${GDAL_LIBRARY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRITON_GDAL")
endif()

if(TRITON_ENABLE_SWMM)
  target_include_directories(${TRITON_EXECUTABLE} PUBLIC ${SWMM_INCLUDE_PATH})
  target_link_libraries(${TRITON_EXECUTABLE} PUBLIC ${SWMM_LIBRARY})
endif()

add_build_and_run_scripts()

# add tests
if(${BUILD_TESTS})
  enable_testing()
  add_subdirectory(${TRITON_SOURCE_DIR}/test)
  add_test_script()
endif()

add_clean_script()

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")


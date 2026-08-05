cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # minimum CMake version to support Kokkos

set(TRITON_EXECUTABLE "triton.exe")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Option to ignore Machine files and use manual configuration only
option(TRITON_IGNORE_MACHINE_FILES "Ignore Machine files and use manual CMake/environment configuration only" OFF)

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

# Get TRITON git version info
find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE TRITON_GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE TRITON_GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} diff-index --quiet HEAD --
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE TRITON_GIT_DIRTY
    ERROR_QUIET
  )
  if(TRITON_GIT_DIRTY)
    set(TRITON_GIT_VERSION "${TRITON_GIT_BRANCH}-${TRITON_GIT_COMMIT_HASH}-dirty")
  else()
    set(TRITON_GIT_VERSION "${TRITON_GIT_BRANCH}-${TRITON_GIT_COMMIT_HASH}")
  endif()
else()
  set(TRITON_GIT_VERSION "unknown")
endif()
message(STATUS "TRITON_GIT_VERSION = ${TRITON_GIT_VERSION}")

# SWMM coupling option
option(TRITON_ENABLE_SWMM "Enable SWMM coupling for urban drainage" OFF)
option(TRITON_SWMM_FLOODING_DEBUG "Enable SWMM flooding consistency debug check" OFF)

# Diagnostic probe for the hotstart-resume external-boundary-condition investigation.
# DEFAULTS OFF: with it off, every instrumented line is inside #ifdef TRITON_EXTBC_PROBE and the
# preprocessed sources are identical to the un-instrumented tree, so this branch cannot silently
# become a production build. Turn it on either explicitly
#     cmake -DTRITON_EXTBC_PROBE=ON ...
# or, for build drivers that hard-code their cmake line and cannot inject a flag, by exporting
#     TRITON_EXTBC_PROBE=ON
# in the environment of the configure step. Runtime behaviour is then controlled by
# TRITON_EXTBC_PROBE_T0 / _STEPS / _DIR -- see src/extbc_probe.h.
if(DEFINED ENV{TRITON_EXTBC_PROBE})
  set(_TRITON_EXTBC_PROBE_DEFAULT $ENV{TRITON_EXTBC_PROBE})
else()
  set(_TRITON_EXTBC_PROBE_DEFAULT OFF)
endif()
option(TRITON_EXTBC_PROBE "Enable the external-BC / hotstart-resume diagnostic probe" ${_TRITON_EXTBC_PROBE_DEFAULT})

if(TRITON_EXTBC_PROBE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRITON_EXTBC_PROBE")
  message(STATUS "TRITON_EXTBC_PROBE enabled -- diagnostic build, NOT for production runs")
endif()

if(TRITON_ENABLE_SWMM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRITON_SWMM")
  message(STATUS "SWMM coupling enabled - SWMM will be built from bundled source")
endif()

# add external packages
add_subdirectory(${TRITON_SOURCE_DIR}/external)

# build triton
add_executable(${TRITON_EXECUTABLE} ./src/main.cpp)
target_link_libraries(${TRITON_EXECUTABLE} PUBLIC kokkos)
target_compile_definitions(${TRITON_EXECUTABLE} PRIVATE TRITON_GIT_VERSION="${TRITON_GIT_VERSION}")

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
  target_include_directories(${TRITON_EXECUTABLE} PUBLIC ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include)
  target_include_directories(${TRITON_EXECUTABLE} PUBLIC ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)
  target_link_libraries(${TRITON_EXECUTABLE} PUBLIC swmm5)

  if(TRITON_SWMM_FLOODING_DEBUG)
    target_compile_definitions(swmm5 PRIVATE SWMM_FLOODING_DEBUG)
    message(STATUS "SWMM flooding debug check enabled")
  endif()
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


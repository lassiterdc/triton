set(TRITON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TRITON_BUILD_DIR ${CMAKE_BINARY_DIR})

if(NOT DEFINED MACHINE)
  if(DEFINED ENV{TRITON_MACHINE})
    set(MACHINE "$ENV{TRITON_MACHINE}")
  else()
    set(MACHINE "${CMAKE_HOST_SYSTEM_NAME}")
  endif()
endif()

if(NOT DEFINED COMPILER)
  set(COMPILER "default")
endif()

if(NOT DEFINED BACKEND)
  set(BACKEND "default")
endif()

macro (set_environment)

set(machinefile_path "")

if(EXISTS "${MACHINE}")
  set(machinefile_path "${MACHINE}")

else()
  file(GLOB_RECURSE ALL_FILES "${TRITON_SOURCE_DIR}/cmake/machines/${MACHINE}/*")
  set(machinefile_name "${COMPILER}_${BACKEND}")
  
  foreach(f ${ALL_FILES})
    # Get the filename without the directory
    get_filename_component(fname "${f}" NAME_WLE)
  
    # Check if osname starts with "Linux"
    if("${fname}" STREQUAL "${machinefile_name}")
      set(machinefile_path "${f}")
      #message(STATUS "fname : ${fname}")
      break()
    endif()
  endforeach()
endif()

if (DEBUG)
  message(STATUS "MACHINE = '${MACHINE}'")
  message(STATUS "COMPILER = '${COMPILER}'")
  message(STATUS "BACKEND = '${BACKEND}'")
endif()

if(NOT EXISTS "${machinefile_path}")
  message(FATAL_ERROR "No maching machine file: ${machinefile_path}")
else()
  get_filename_component(FILE_EXT "${machinefile_path}" EXT)
  set(ENVFILE "triton_env${FILE_EXT}")
  configure_file("${machinefile_path}" "${TRITON_BUILD_DIR}/${ENVFILE}" COPYONLY)
  message(STATUS "machine file: ${machinefile_path}")
endif()

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
	run_win_command("${machinefile_path} && set" ENV_OUTPUT)
else()
	run_bash_command("source ${machinefile_path} && env" ENV_OUTPUT)
endif()

string(REPLACE "\n" ";" lines ${ENV_OUTPUT})

set(_REQUIRED_ENVS
  COMPILER
  BACKEND
  RUN_COMMAND
)

set(_TRITON_BACKENDS
  CUDA
  HIP
  SYCL
  OPENMP
  OPENMPTARGET
  THREADS
  SERIAL
  default
)

# set env. variables
foreach(line ${lines})
  string(REGEX MATCH "([A-Za-z_][A-Za-z0-9_]*)=(.*)" ENV_LINE ${line})
  if (ENV_LINE)
    set(ENV_VAR "${CMAKE_MATCH_1}")
    set(ENV_VAL "${CMAKE_MATCH_2}")
  
    string(REGEX MATCH "^TRITON_(.*)" _matched "${ENV_VAR}")
	if(_matched)
      set(SUFFIX "${CMAKE_MATCH_1}")
      if(NOT DEFINED ${SUFFIX} OR ${SUFFIX} STREQUAL "default" OR ${SUFFIX} STREQUAL "COMPILER")
        if(DEFINED ENV{TRITON_${SUFFIX}})
          set(${SUFFIX} "$ENV{TRITON_${SUFFIX}}")
        else()
          set(${SUFFIX} "${ENV_VAL}")
        endif()
      endif()
    else()
      set(ENV{${ENV_VAR}} "${ENV_VAL}")
    endif()
  endif()

endforeach()

set(_MISSING_ENVS "")

foreach(VAR IN LISTS _REQUIRED_ENVS)
  if(NOT DEFINED ${VAR})
    list(APPEND _MISSING_ENVS "${VAR}")
  endif()
endforeach()

if(_MISSING_ENVS)
  message(FATAL_ERROR "The following triton variables are not defined:\n  ${_MISSING_ENVS}")
endif()

find_program(CMAKE_CXX_COMPILER "${COMPILER}")
set(CMAKE_CXX_FLAGS "${COMPILER_FLAGS} ${COMPILER_FLAGS_APPEND}")
set(CMAKE_EXE_LINKER_FLAGS "${LINKER_FLAGS} ${LINKER_FLAGS_APPEND}")

if (DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRITON_DEBUG")
endif()

message(STATUS "TRITON_MACHINE=${MACHINE}")

list(FIND _TRITON_BACKENDS "${BACKEND}" VAR_INDEX)
if(VAR_INDEX EQUAL -1)
  message(FATAL_ERROR "Not supported BACKEND: ${BACKEND}")
else()
  message(STATUS "TRITON_BACKEND=${BACKEND}")
endif()

set(_Kokkos_Enable_Var "Kokkos_ENABLE_${BACKEND}")
set(${_Kokkos_Enable_Var} ON CACHE BOOL "Enable Kokkos backend: ${BACKEND}")

if("${BACKEND}" STREQUAL "CUDA")
  set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Enable CUDA CONSTEXPR")
endif()

if (ARCH)
  set(_Kokkos_Arch_Var "Kokkos_ARCH_${ARCH}")
  set(${_Kokkos_Arch_Var} ON CACHE BOOL "Set Kokkos archtecture: ${ARCH}")
  message(STATUS "TRITON_ARCH=${ARCH}")
endif()

message(STATUS "TRITON_RUN_COMMAND=${RUN_COMMAND}")

message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")

endmacro()

function(add_triton_test test_name cfg_name)
  file(COPY ${TRITON_SOURCE_DIR}/test/${cfg_name}
       DESTINATION ${TRITON_BUILD_DIR}/test)
  set(_TEST_CMD "${RUN_COMMAND} ../${TRITON_EXECUTABLE} ./${cfg_name}")
  string (STRIP "${_TEST_CMD}" _TEST_CMD_STRIPPED)
  string (REPLACE " " ";" _CMD_LIST "${_TEST_CMD_STRIPPED}")
  add_test(
    NAME ${test_name}
    COMMAND ${_CMD_LIST}
  )
  set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test")
endfunction()

if (ENSEMBLE_BUILD)

  add_test(
    NAME ENSEMBLE_PARABOLOID
    COMMAND bash;-c;./triton_ensrun.sh &&
      [ -s ./output_ens1/asc/QY_12_00.out ] &&
      [ -s ./output_ens2/asc/QY_12_00.out ] &&
      cmp -s ./output_ens1/asc/QY_12_00.out ./output_ens2/asc/QY_12_00.out
  )
  set_tests_properties(ENSEMBLE_PARABOLOID PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

else()

  add_triton_test(
      PARABOLOID
      paraboloid.cfg
  )

  add_test(
      NAME BIT4BIT-PARABOLOID
      COMMAND ${Python3_EXECUTABLE} 
                  ${CMAKE_BINARY_DIR}/compare_runs_simple.py
                  --ref ${CMAKE_SOURCE_DIR}/test/reference/paraboloid
                  --test ${CMAKE_BINARY_DIR}/test/output_paraboloid/asc
  )
  add_triton_test(
      CIRCULAR-DAMBREAK
      circular_dambreak.cfg
  )

  add_test(
      NAME BIT4BIT-CIRCULAR-DAMBREAK
      COMMAND ${Python3_EXECUTABLE} 
                  ${CMAKE_BINARY_DIR}/compare_runs_simple.py
                  --ref ${CMAKE_SOURCE_DIR}/test/reference/circular
                  --test ${CMAKE_BINARY_DIR}/test/output_circular_dambreak/asc
  )

  add_triton_test(
      ALLATOONA
      allatoona.cfg
  )

  add_test(
      NAME BIT4BIT-ALLATOONA
      COMMAND ${Python3_EXECUTABLE} 
                  ${CMAKE_BINARY_DIR}/compare_runs_simple.py
                  --ref ${CMAKE_SOURCE_DIR}/test/reference/allatoona
                  --test ${CMAKE_BINARY_DIR}/test/output_allatoona/asc
  )

  # Order each comparator after its own simulation, from an empty output
  # directory. CTest schedules by cost, so under `ctest -jN` a cheap BIT4BIT
  # comparator starts beside its simulation and finishes first: red on a fresh
  # build, and on a reused build it compares the previous run's rasters and
  # passes. The clean node also stops a leftover raster from a longer earlier
  # run being picked by the comparator's lexicographically-last file selection.
  foreach(_case PARABOLOID CIRCULAR-DAMBREAK ALLATOONA)
    if(_case STREQUAL "PARABOLOID")
      set(_out output_paraboloid)
    elseif(_case STREQUAL "CIRCULAR-DAMBREAK")
      set(_out output_circular_dambreak)
    else()
      set(_out output_allatoona)
    endif()
    add_test(NAME CLEAN-OUTPUT-${_case}
             COMMAND ${CMAKE_COMMAND} -E remove_directory ${_out})
    set_tests_properties(CLEAN-OUTPUT-${_case} PROPERTIES
                         WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test"
                         FIXTURES_SETUP ${_case}_CLEAN)
    set_tests_properties(${_case} PROPERTIES
                         FIXTURES_REQUIRED ${_case}_CLEAN
                         FIXTURES_SETUP ${_case}_OUTPUT)
    set_tests_properties(BIT4BIT-${_case} PROPERTIES
                         FIXTURES_REQUIRED ${_case}_OUTPUT)
  endforeach()

endif()

# Tier-1 unit tests and the snapshot-inventory tooling. The unit tests include
# swmm_triton.h directly, so they exist only in a coupled configuration.
if(TRITON_ENABLE_SWMM)
  add_subdirectory(${CMAKE_SOURCE_DIR}/test/unit ${CMAKE_BINARY_DIR}/test/unit)
  add_subdirectory(${CMAKE_SOURCE_DIR}/test/snapshot ${CMAKE_BINARY_DIR}/test/snapshot)
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/test/inventory ${CMAKE_BINARY_DIR}/test/inventory)

# Runtime tests for the SWMM timer column split: they drive the real SuperTimer
# and Output::write_times and assert that SWMM_STEP is exactly 0.0 on every rank
# but rank 0 and that the SWMM child columns sum to SWMM. Registered in every
# configuration, because the columns are emitted in every configuration.
add_subdirectory(${CMAKE_SOURCE_DIR}/test/perf ${CMAKE_BINARY_DIR}/test/perf)



# Source checks: pure Python over the tree, no build and no solver run, so they
# are registered unconditionally.

# No swmm_open, swmm_start or swmm_step call in src/ discards its return code.
# SWMM reports a rejected input only through those codes; discarded, the coupled
# run completes against a sewer that never moves. The test also runs its
# predicate on a discarding and a capturing snippet, so a predicate that can no
# longer go red fails it.
add_test(
    NAME SOURCE-SWMM-RETURN-CODES
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_swmm_return_codes_source.py
                ${CMAKE_SOURCE_DIR}
)

# Every C++ translation unit must PARSE under `mpic++ -fsyntax-only` with the
# Kokkos config shim in test/syntax/kokkos-shim/: no CMake configure of Kokkos,
# no link, no solver run. It asserts each TU's exit status, not the absence of
# warnings (their count varies by site). It does not use CMAKE_CXX_COMPILER,
# which may be a site wrapper; pass --compiler to override.
add_test(
    NAME SOURCE-SYNTAX-PARSE-ALL-TUS
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/syntax/run_syntax_check.py
                ${CMAKE_SOURCE_DIR}
)

# Source properties of the SWMM timer column split that the runtime tests cannot
# tell from their violations, among them that SWMM_STEP's bracket in triton.h
# sits inside the rank-0 guard.
add_test(
    NAME SOURCE-SWMM-COLUMN-SPLIT
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_swmm_column_split_source.py
                ${CMAKE_SOURCE_DIR}
)

# Tier-1 unit tests for the coupled side-file header (no solver run).
# Placed at the top of this file so an appended registration elsewhere cannot
# conflict with it on a shared branch.
if(TRITON_ENABLE_SWMM)
  add_subdirectory(unit)
  add_subdirectory(snapshot)
endif()

# Slow-tier inventory-regeneration check (WP-1B). Registered unconditionally:
# it reads the vendored EPA source with Python and links nothing, so it is
# meaningful in a non-coupled configuration too -- the inventory describes the
# source tree, not the built binary.
add_subdirectory(inventory)

# Source-structural guards (WP-0B). Pure Python over src/triton.h: no build, no
# GPU, no solver run and no SWMM, so they are registered UNCONDITIONALLY --
# gating them on TRITON_ENABLE_SWMM would make the fence guard unreachable in
# exactly the configuration a reviewer can run fastest, and the property it
# holds is a property of the SOURCE rather than of any build.  The source path
# is passed explicitly so a node id does not depend on the working directory.
#
# Two node ids, not one.  The first asserts the property; the second asserts
# that the first still HOLDS that property against every conditional form.  A
# guard with no guard of its own can be weakened by a later edit and stay green,
# which is the condition this package was dispatched to repair.
add_test(
    NAME SOURCE-FENCE-COMPUTE-TIMER
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_compute_timer_fence.py
                ${CMAKE_SOURCE_DIR}/src/triton.h
)

add_test(
    NAME SOURCE-FENCE-COMPUTE-TIMER-CONDITIONAL-FORMS
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_compute_timer_fence_conditional_forms.py
                ${CMAKE_SOURCE_DIR}/src/triton.h
)

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

endif()



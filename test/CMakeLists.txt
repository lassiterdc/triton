# Tier-1 unit tests for the coupled side-file header (no solver run).
# Placed at the top of this file so an appended registration elsewhere cannot
# conflict with it on a shared branch.
if(TRITON_ENABLE_SWMM)
  add_subdirectory(unit)
  add_subdirectory(snapshot)
endif()

# FAST-TIER syntax check. Every C++ translation unit must PARSE under
# `mpic++ -fsyntax-only` with the six-file Kokkos config shim checked into
# test/syntax/kokkos-shim/. Measured on this tree: 5 TUs, ~6.6 s total.
#
# Registered UNCONDITIONALLY and deliberately FIRST among the Python nodes. It
# needs no CMake configure of Kokkos, no link, no GPU and no solver run, so it
# is the cheapest instrument in the suite and the one most likely to be run --
# which matters, because the failure class it catches (a broken declaration in a
# header) otherwise surfaces only after a full Kokkos build. It already caught a
# real symbol redefinition on this branch, fixed at d648106.
#
# It asserts each TU's compiler EXIT STATUS and NOT the absence of warnings.
# Measured at this commit with -Wall -Wextra there are 4 project-source warnings
# on every test TU and 11 on src/main.cpp, plus 2 more per TU from the installed
# OpenMPI's own headers -- a count that varies by site. A zero-diagnostics gate
# would fail immediately and would fail differently on every machine. See
# test/syntax/run_syntax_check.py's docstring for the full attribution and for
# the extern "C" signature-drift blind spot that no syntax-only check can close.
#
# It does NOT use ${CMAKE_CXX_COMPILER}: that may be a site wrapper (CC on Cray)
# whose include defaults differ from the plain MPI wrapper this shim was measured
# against. Pass --compiler to override.
add_test(
    NAME SOURCE-SYNTAX-PARSE-ALL-TUS
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/syntax/run_syntax_check.py
                ${CMAKE_SOURCE_DIR}
)

# Slow-tier inventory-regeneration check (WP-1B). Registered unconditionally:
# it reads the vendored EPA source with Python and links nothing, so it is
# meaningful in a non-coupled configuration too -- the inventory describes the
# source tree, not the built binary.
add_subdirectory(inventory)

# Tier-1 runtime tests for the SWMM timer column split (WP-1A). Registered
# UNCONDITIONALLY: they include only supertimer.h and output.h, and the four new
# columns are emitted in EVERY configuration -- get_custom_time REGISTERS an
# absent category and returns 0.0, so the identity must close in a non-coupled
# build too. Gating them on TRITON_ENABLE_SWMM would make them unreachable in
# the configuration a reviewer can run fastest.
add_subdirectory(perf)

# Source-structural guards for the same split, plus the guard-of-the-guard.
# Pure Python over src/: no build, no GPU, no solver run, so registered
# unconditionally like the fence guards below. Two node ids, not one, and each
# holds a different line. The first asserts the five source properties the
# RUNTIME tests cannot distinguish from their violations -- a child bracket that
# escaped its parent still emits a plausible number, and a SWMM_OTHER macro
# still compiles and still returns 0.0. The second applies eight named mutations
# to a throwaway copy of the tree and asserts the first goes RED on each,
# because a guard that cannot go red is indistinguishable from one that can and
# only the mutation run tells them apart.
add_test(
    NAME SOURCE-SWMM-COLUMN-SPLIT
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_swmm_column_split_source.py
                ${CMAKE_SOURCE_DIR}
)

add_test(
    NAME SOURCE-SWMM-COLUMN-SPLIT-GUARD-FORMS
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_swmm_column_split_guard_forms.py
                ${CMAKE_SOURCE_DIR}
)

# check_performance_identity.py is the artifact an operator points at a REAL
# coupled run's performance.txt -- the §8.3 smoke test's, or any member's. It is
# not exercised by any other node id, so it gets one: this feeds it four
# known-good and seven known-bad files and asserts it separates them AND reaches
# each verdict for the stated reason. Without it, "the checker passed" is
# indistinguishable from "the checker never reached its comparison".
add_test(
    NAME SOURCE-PERF-IDENTITY-CHECKER
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_performance_identity_checker.py
                ${CMAKE_SOURCE_DIR}
)

# WHY THE RESIDUAL'S SIGN IS THE FALSIFIER, AND THE SUM IS NOT.
# SWMM_OTHER is DERIVED by subtraction, so XFER + MPI + STEP + OTHER == SWMM is
# an algebraic tautology over the emitted row: it holds for ANY values the three
# measured children hold, including values no correct bracket placement could
# produce. What falsifies a bracket fault is the residual's SIGN -- a child that
# escaped the parent, or two children that overlap, drive SWMM_OTHER NEGATIVE
# while leaving the sum exact. test_perf_identity.cpp P4 asserts other >= 0, but
# only against its own correctly-nested in-process harness, which is why the two
# artifacts that meet REAL data need the assertion of their own: the
# operator-facing checker, and the source guard over the bracket placement.
#
# This node covers exactly those two. C1 mutates a run's emitted row into an
# escaped child and into overlapping children and requires the operator-facing
# checker to reject both; C2 mutates the bracket placement in source and
# requires the source guard to reject it.
#
# IT IS REGISTERED WITHOUT CTEST'S EXPECTED-FAILURE PROPERTY AND IS NOT
# EXPECTED TO FAIL. (The property's name is deliberately not spelled here:
# `grep -c` for it over this file is how a reader establishes that no node
# carries one, and a mention in prose would answer 1 where the true count of
# set properties is 0.) A red at this node is a regression in one of those two
# artifacts, not a known-open item. It was registered red by the WP-1A review,
# when neither artifact carried the
# assertion and the node was the review's way of leaving the gap visible; both
# were subsequently repaired and the node passes. That history is recorded
# because the node id still reads like a reviewer's placeholder, and because a
# status claim left standing after its repair trains a reader to discount a
# genuine red at exactly this node.
add_test(
    NAME SOURCE-SWMM-COLUMN-SPLIT-RESIDUAL-SIGN
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_swmm_column_split_residual_sign.py
                ${CMAKE_SOURCE_DIR}
)

# Source-structural guards (WP-0B). Pure Python over src/triton.h: no build, no
# GPU, no solver run and no SWMM, so they are registered UNCONDITIONALLY --
# gating them on TRITON_ENABLE_SWMM would make the fence guard unreachable in
# exactly the configuration a reviewer can run fastest, and the property it
# holds is a property of the SOURCE rather than of any build.  The source path
# is passed explicitly so a node id does not depend on the working directory.
#
# Three node ids, not one, and each holds a different line.  The first asserts
# the property.  The second asserts that the first still HOLDS that property
# against every conditional form it claims to catch.  The third pins the forms
# it is DOCUMENTED NOT to catch, so the guard's blind-spot list and the guard's
# behaviour cannot drift apart in either direction.  A guard with no guard of
# its own can be weakened by a later edit and stay green, which is the condition
# this package was dispatched to repair -- and an unregistered guard is the same
# defect wearing a different hat, which is why all three are registered here
# rather than left to be run by hand.
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

# Characterization tripwire: PASSES today because the three forms it carries are
# known, disclosed gaps in CHECK 2b.  It goes RED the day someone closes one
# without updating the guard's blind-spot list, and its failure output names the
# form and both files to update.  A red here is news that the guard improved,
# not that it broke.
add_test(
    NAME SOURCE-FENCE-COMPUTE-TIMER-UNDETECTED-FORMS
    COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/test/test_compute_timer_fence_undetected_forms.py
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



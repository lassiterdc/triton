# KEEP THIS FILE. The upstream tree deleted it when it integrated this branch's
# work, and it is kept here deliberately. This note is also the edit that makes
# a merge with upstream report a modify/delete conflict on this path rather than
# drop the file with no conflict at all, because a path one side deleted and the
# other side never changed merges as deleted. The conflict puts the deletion to
# whoever resolves the merge instead of letting it be inherited.
#
# Why it is worth keeping: test_perf_identity drives the real SuperTimer and the
# real Output::write_times in the order src/triton.h uses them, on one rank and,
# where CMake finds an MPI launcher, on four, and asserts from the emitted
# performance.txt that SWMM_STEP is exactly 0.0 on every rank but rank 0 and that
# SWMM_XFER + SWMM_MPI + SWMM_STEP + SWMM_OTHER == SWMM on every per-rank row.
# The upstream tree carries no test that SWMM_STEP is 0.0 off rank 0 (its only
# SWMM_STEP check, section G of test/coupled_resume_suite.sh, tests the column
# identity and the residual's sign on a serial run), and this test asserts it on
# the emitted file through the real timer and emitter. It re-enacts triton.h's
# guarded bracket rather than reading triton.h, so it does not detect the
# bracket moving; the placement itself is read by
# test/test_swmm_column_split_source.py's S3, re-incorporated in the same set.

# Tier-1 runtime tests for the SWMM timer column split.
#
# No solver run, no cluster, no SWMM call, no GPU. They drive the real
# SuperTimer and the real Output::output<T>::write_times, then parse the file
# write_times actually emitted -- so the property is held against the emit path
# rather than against a re-implementation of its arithmetic.
#
# Registered UNCONDITIONALLY, unlike test/unit and test/snapshot. Those include
# swmm_triton.h, which compiles to nothing outside a coupled build; this one
# includes only supertimer.h and output.h, and the columns it checks are emitted
# in EVERY configuration -- get_custom_time registers an absent category and
# returns 0.0, so a non-coupled build still writes all four with value zero. The
# identity must hold there too, and gating this on TRITON_ENABLE_SWMM would make
# it unreachable in the configuration a reviewer can run fastest.

add_executable(test_perf_identity test_perf_identity.cpp)

target_include_directories(test_perf_identity PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(test_perf_identity PRIVATE kokkos)

if(TRITON_GDAL)
  target_link_libraries(test_perf_identity PRIVATE ${GDAL_LIBRARY})
  target_include_directories(test_perf_identity PRIVATE ${GDAL_INCLUDE_DIR})
endif()

# One CTest node id per property, so a result is accounted by name rather than by
# a pass/fail total that cannot say which property it contains.
set(TRITON_PERF_IDENTITY_TESTS
  P1_header_carries_four_child_columns
  P2_identity_closes_on_every_per_rank_row
  P3_average_row_present_with_full_arity
  P4_step_is_exactly_zero_off_rank0_in_the_timer
  P5_step_prints_as_zero_off_rank0
  P6_parent_equals_an_unsplit_measurement)

# SINGLE rank: the identity, the header and the arity hold at any rank count, and
# this arm needs no launcher.
foreach(_t IN LISTS TRITON_PERF_IDENTITY_TESTS)
  add_test(NAME PERF-IDENTITY-1RANK-${_t}
           COMMAND test_perf_identity ${_t})
endforeach()
# P6 compares two WALL-CLOCK readings of the same sleep-driven work at a 10 %
# tolerance -- about 0.65 ms on ranks 1-3. A concurrent ctest neighbour (any
# co-scheduled work; oversubscription is not required) delays a wakeup by more than that, so the node must
# run alone. Measured: 13 of 15 `ctest -j4` repeats red without this, 0 of 15
# with it; the other PERF nodes assert exact arithmetic and need no isolation.
set_tests_properties(PERF-IDENTITY-1RANK-P6_parent_equals_an_unsplit_measurement
                     PROPERTIES RUN_SERIAL TRUE)

# MULTI rank: P4 and P5 are specifically about the ranks that are NOT rank 0, and
# P3's Average-is-rank0/N assertion needs size_ > 1. A single-rank run cannot
# distinguish "the bracket is inside the guard" from "the bracket is outside it"
# -- on one rank the guard is always taken -- so the multi-rank arm is the one
# that actually holds the placement decision.
find_program(TRITON_MPIEXEC NAMES mpiexec mpirun srun)
if(TRITON_MPIEXEC)
  foreach(_t IN LISTS TRITON_PERF_IDENTITY_TESTS)
    add_test(NAME PERF-IDENTITY-4RANK-${_t}
             COMMAND ${TRITON_MPIEXEC} -n 4 $<TARGET_FILE:test_perf_identity> ${_t})
  endforeach()
  set_tests_properties(PERF-IDENTITY-4RANK-P6_parent_equals_an_unsplit_measurement
                       PROPERTIES RUN_SERIAL TRUE)
else()
  message(STATUS "No mpiexec/mpirun/srun found: the multi-rank PERF-IDENTITY "
                 "arm is NOT registered. The single-rank arm cannot hold the "
                 "SWMM_STEP bracket-placement property.")
endif()

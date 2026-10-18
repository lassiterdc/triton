# Tier-1 unit tests for the coupled-resume state snapshot.
#
# No solver run, no cluster, no simulation. They include swmm_triton.h and call
# snapshot.c's C entry points directly, so they build only in a coupled
# (TRITON_ENABLE_SWMM=ON) configuration.

add_executable(test_state_snapshot test_state_snapshot.cpp)

target_include_directories(test_state_snapshot PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

# MPI IS A DECLARED DEPENDENCY OF THIS TARGET, not an inherited accident.
#
# test_state_snapshot.cpp includes "mpi.h" directly (it reproduces main.cpp's
# include order, because swmm_triton.h is not independently includable at this
# commit). The top-level CMakeLists requires MPI only on Windows, so on every
# other platform this target compiled and linked ONLY because the configured
# CXX compiler happened to be an MPI wrapper -- a configuration the build
# nowhere states and nowhere checks. A configure that selects a bare g++ then
# fails at the END of the build on a header this file names on line 29.
#
# REQUIRED rather than optional, and the choice is the point: the dependency is
# unconditional in the source, so a missing MPI is a configure-time error with
# the cause named rather than a link-time failure at 98%.
find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(test_state_snapshot PRIVATE kokkos swmm5 MPI::MPI_CXX)

# One CTest node id per property, so a result is accounted by name rather than
# by a pass/fail total that cannot say which property it contains.
set(TRITON_SNAPSHOT_TESTS
  T3_snapshot_roundtrips_bit_exactly
  T3b_stored_width_is_double_not_float
  T3c_routing_fields_roundtrip_bit_exactly
  T4_bad_magic_is_refused
  T4b_missing_file_is_refused_by_name
  T5_shape_mismatch_is_refused_and_named
  T6_manifest_covers_committed_inventory
  T7_sizeof_guard_boundary_is_documented
  T8_single_writer_is_structural
  # Full-window .out continuity. One node per property, as above.
  T3d_nperiods_roundtrips_bit_exactly
  T12_resume_refuses_more_periods_than_the_file_holds
  T13_snapshot_load_return_is_graduated
  T14_output_end_leaves_the_trailer_last
  # PAYLOAD LENGTH. The only property in this tier that can detect a SYMMETRIC
  # write/read defect: the repaired guard sat above a traversal shared across
  # manifest, write and read modes, so the round-trip nodes above pass on a
  # tree that writes every snapshot short. One node per property, as above.
  T15_payload_length_does_not_vary_with_routeto
  T16_payload_length_tracks_the_pollutant_axis
  T17_a_refused_save_publishes_nothing
  # THE RESUME-SIDE KILLED-WRITE CONSULTATION. One node per DoD-10 command
  # (a)-(d), so a result names which of the four properties it contains rather
  # than reporting a pass/fail total over all of them. T17 above covers the
  # WRITE-side half of the same mechanism and is deliberately a separate node.
  T18_killed_part_beside_a_complete_prior_snapshot_is_refused
  T19_the_killed_write_refusal_names_cause_fallback_and_replay_cost
  T20_the_part_is_renamed_killed_and_the_classifier_returns_an_existing_token
  T21_a_clean_start_disposes_both_markers
  # THE STEP-BACK LINES, THE FAILED-WRITE STOP AND THE CLEAN-START DISPOSAL.
  # One node per DoD-10 command (e)-(l) and (k1)-(k4), for the same reason as
  # above: a result names which property it carries.
  T22_a_killed_write_with_a_published_snapshot_below_emits_the_step_back_line
  T23_no_published_snapshot_below_emits_the_clean_start_line_with_its_reason
  T24_a_failed_write_with_a_published_snapshot_below_emits_the_stop_line
  T25_the_step_back_line_is_identical_on_a_second_resume
  T26_the_step_back_confirmation_scan_reads_only_live_markers
  T27_a_clean_start_keeps_a_failed_marker_under_one_directory_wide_k
  T28_a_clean_start_without_keep_previous_removes_the_earlier_run
  T29_a_clean_start_with_keep_previous_keeps_every_artifact
  T30_an_undisposable_published_snapshot_refuses_the_clean_start
  T31_an_undisposable_marker_warns_and_the_clean_start_continues
  T32_a_failed_snapshot_write_keeps_its_marker_and_says_the_resume_will_stop
  # DoD 10 (l2) and (m): the count-pass refusal's marker, and the resume stop
  # it causes.
  T33_a_count_pass_refusal_leaves_a_one_line_failed_marker
  T34_a_count_pass_failed_marker_stops_the_resume_with_the_stop_line)

foreach(_t IN LISTS TRITON_SNAPSHOT_TESTS)
  add_test(NAME UNIT-SNAPSHOT-${_t}
           COMMAND test_state_snapshot ${_t})
  # T6 compares the emitted manifest against the COMMITTED inventory, so it
  # needs to find it. Passed by environment rather than compiled in, so the
  # test binary carries no build-time copy of a path that can go stale.
  set_tests_properties(UNIT-SNAPSHOT-${_t} PROPERTIES
    ENVIRONMENT
      "TRITON_SNAPSHOT_INVENTORY=${CMAKE_SOURCE_DIR}/test/inventory/snapshot_inventory.txt")
endforeach()

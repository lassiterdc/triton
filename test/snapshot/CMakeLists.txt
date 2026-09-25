# Tier-1 unit tests for the coupled-resume state snapshot (WP-1B).
#
# No solver run, no cluster, no simulation. They include swmm_triton.h and call
# snapshot.c's C entry points directly, so they build only in a coupled
# (TRITON_ENABLE_SWMM=ON) configuration.

add_executable(test_state_snapshot test_state_snapshot.cpp)

target_include_directories(test_state_snapshot PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_state_snapshot PRIVATE kokkos swmm5)

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
  T8_single_writer_is_structural)

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

# Slow-tier invocation of the snapshot inventory-regeneration check.
#
# This is ONE of the check's two declared firing points; the other is a named
# step in the upstream-bump procedure (test/inventory/UPSTREAM_BUMP.md).  Two
# firing points rather than one because they catch the change at different
# moments: the bump step catches it before the vendored tree lands, and this
# test catches it if the bump step was skipped.
#
# It is excluded from the fast suite by the `slow` label -- it re-parses the
# EPA solver tree and closes two call graphs over it, which is cheap in
# absolute terms but has no business running on every edit.  Run it with:
#
#     ctest -L slow
#
# and exclude it from a fast pass with:
#
#     ctest -LE slow

if(Python3_EXECUTABLE)
  add_test(
    NAME SLOW-SNAPSHOT-INVENTORY-REGENERATION
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/regenerate_snapshot_inventory.py
            --solver-dir ${CMAKE_SOURCE_DIR}/external/swmm/src/solver
            --inventory  ${CMAKE_CURRENT_SOURCE_DIR}/snapshot_inventory.txt
            --check)
  set_tests_properties(SLOW-SNAPSHOT-INVENTORY-REGENERATION
                       PROPERTIES LABELS "slow")

  # Criterion P's own assertions (WP-1B chunk 9).  FAST tier, deliberately:
  # ~3 s, and it carries the ORDER-SENSITIVITY regression, which is the
  # assertion a builder editing the kill pass most needs to see immediately.
  # A regression that only runs under `ctest -L slow` is one nobody runs.
  add_test(
    NAME SOURCE-SNAPSHOT-INVENTORY-CRITERION-P
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/test/test_snapshot_inventory_criterion_p.py
            ${CMAKE_SOURCE_DIR})

  # The guard of that guard.  Both nodes above return 0 on the landed tree, and
  # so would a check whose pre-filter verified itself, whose triage tables
  # swallowed every name, or whose parser truncated the walk -- none of which is
  # visible from a green run.  This one mutates a throwaway copy of the solver
  # tree eight times and asserts the check goes RED on each, naming the mode.
  # SLOW tier: it copies the solver tree once per mutation (~14 s).
  add_test(
    NAME SLOW-SNAPSHOT-INVENTORY-CLOSURE-MODES
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/test/test_snapshot_inventory_closure_modes.py
            ${CMAKE_SOURCE_DIR})
  set_tests_properties(SLOW-SNAPSHOT-INVENTORY-CLOSURE-MODES
                       PROPERTIES LABELS "slow")
endif()

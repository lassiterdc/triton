# Slow-tier invocation of the snapshot inventory-regeneration check.
#
# This is ONE of the check's two declared firing points; the other is a named
# step in the upstream-bump procedure (test/inventory/UPSTREAM_BUMP.md).  Two
# firing points rather than one because they catch the change at different
# moments: the bump step catches it before the vendored tree lands, and this
# test catches it if the bump step was skipped.
#
# It is excluded from the fast suite by the `slow` label -- it re-parses four
# EPA translation units and closes a call graph over them, which is cheap in
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
endif()

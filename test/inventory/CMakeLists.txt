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

# IT REDDENS FOR TWO INDEPENDENT REASONS AND THE OUTPUT NAMES WHICH.  Exit 1 is
# DRIFT -- the vendored EPA tree moved underneath the committed artifact, and
# the remedy is to regenerate and re-review.  Exit 3 is OPEN TRIAGE DECISIONS --
# the artifact is current and some quantity the closure surfaced has no verdict
# yet, and the remedy is to give it one.  Do not read the second as the first:
# on an exit-3 red the vendored tree has not moved at all.
#
# The open-decision half is closure-check mode (i), and it is
# armed here deliberately.  It was disarmed for a while by an honest accident:
# a round correctly surfaced a large set of untriaged quantities and correctly
# committed them to the artifact, which emptied the diff -- so a diff-only
# check had nothing left to say and reported a tree full of open decisions as
# clean.  An empty diff over an un-triaged artifact is not a passing state.

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

  # Criterion P's own assertions.  FAST tier, deliberately:
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

  # The GUARD-PREDICATE closure check.  Pure Python
  # over src/triton.h -- no build, no solver run.  FAST tier: it is a grep and a
  # set comparison, and its subject is a one-token divergence between a local
  # and a global node count that reads identically at a glance and fails
  # SILENTLY on a decomposition where rank 0 owns no manhole.  It returned 3
  # lines with 2 distinct predicates and FAILED before chunk (12); it is kept so
  # a future edit cannot reintroduce the divergence unnoticed.
  add_test(
    NAME SOURCE-SWMM-GUARD-PREDICATE-CLOSURE
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/test/test_swmm_guard_predicate_closure.py
            ${CMAKE_SOURCE_DIR})

  # The snapshot TRAVERSAL-GUARD source check.  Pure Python over
  # external/swmm/src/solver/snapshot.c -- no build, no solver run, so it sits
  # in the fast tier like its SOURCE-* siblings and carries no `slow` label.
  #
  # It exists because NOTHING in the suite could detect a revert of the
  # traversal-guard repair.  Measured: the two nearest inventory checks carry
  # zero references to wRouted, routeTo, SNAP_MANIFEST or Nobjects[POLLUT],
  # and their verdict output is byte-identical on the pre-fix and post-fix
  # trees -- a test that never names a predicate cannot discriminate a change
  # to it.
  #
  # Three assertions, and they are NOT all the same kind: two are detectors
  # keyed on what the repair changed (the per-element guard's routeTo conjunct;
  # SNAPSHOT_VERSION 6 where pre-fix carries 5), and one is an anti-narrowing
  # invariant keyed on what the repair deliberately did NOT change (the
  # pre-loop guard still admits the counting mode).  Expect the third to be
  # green on a pre-fix tree -- that is correct, not a broken assertion.
  #
  # Narrowing the pre-loop guard is what it exists to stop: the per-element
  # s->routeTo load is safe only because that guard already fired, so the
  # observable of removing it is a null dereference forming an element address
  # at (nil), exit 139.
  #
  # Measured on the tree as landed: exit 0.  On a `git archive 031ea42b`
  # pre-fix export: exit 1 on the two detectors.  On an out-of-tree mutant with
  # the pre-loop guard narrowed: exit 1 on the invariant.
  add_test(
    NAME SOURCE-SNAPSHOT-TRAVERSAL-GUARD
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/test/test_snapshot_traversal_guard_source.py
            ${CMAKE_SOURCE_DIR})
endif()

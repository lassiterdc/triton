#!/usr/bin/env python3
"""Falsifiability tests for check_performance_identity.py (WP-1A chunk 4).

WHY THIS EXISTS
---------------
``check_performance_identity.py`` is the artifact a reviewer or a smoke-test
operator points at a REAL coupled run's ``performance.txt``.  It returns 0 on a
healthy file -- and so would a checker whose comparison was inverted, whose
tolerance was infinite, or which never reached the comparison at all.  Nothing in
a green run distinguishes those.  This file feeds the checker known-GOOD and
known-BAD files and asserts it separates them, so "the checker passed" means
something.

The BAD cases are the ones a real defect would produce, not arbitrary corruption:

  B1  the four columns sum HIGH                  -- corruption, or a write_times
  B2  the four columns sum LOW                      derivation edit; NOT a
                                                    bracket fault, which leaves
                                                    the sum exact
  B3  SWMM_STEP nonzero on a rank other than 0   -- the bracket left its guard
  B7  a child LARGER than its whole parent       -- an escaped bracket: the sum
  B8  two children overlapping                      still closes exactly, and
                                                    only the residual's SIGN
                                                    rejects either
  B9  a negative measured child                  -- a parse or format fault
  B4  the Average row missing                    -- arity broken
  B5  a row with the wrong arity                 -- a column dropped
  B6  a missing child column                     -- the split half-landed

and the GOOD cases pin the boundary rather than only the middle:

  G1  exact closure
  G2  error just INSIDE the derived print bound  -- must PASS
  G3  all-zero SWMM fields (a non-coupled run)   -- must PASS, and be REPORTED
                                                    as degenerate rather than
                                                    counted as meaningful
  G4  single-rank run, SWMM_STEP nonzero on rank 0 -- must PASS
  G5  a child exactly EQUAL to its parent        -- legal, and the boundary the
                                                    [0, parent] band must not
                                                    false-fail
  G6  a residual of exactly zero                 -- legal; the band is >= 0

and three flag cases pin the option surface, because a silently-dropped
``--require-ranks`` is a PASS without the check the operator asked for:

  F1  the space-separated form is honoured
  F2  a mistyped flag exits 2 rather than being ignored
  F3  --require-ranks with no value exits 2

USAGE
-----
    test_performance_identity_checker.py <repo-root>
"""

import os
import subprocess
import sys
import tempfile

COLS = ["%Rank", "Compute", "MPI", "IO", "Resize", "SWMM", "SWMM_XFER",
        "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER", "Other", "Simulation",
        "Init", "Total"]


def row(rank, swmm, xfer, mpi, step, other):
    """One well-formed row; the non-SWMM columns are filler this checker ignores."""
    return [str(rank), "1", "0.5", "0.1", "0", str(swmm), str(xfer), str(mpi),
            str(step), str(other), "0.2", "3", "0.4", "3.4"]


def write(path, rows, cols=None, average=True):
    cols = cols or COLS
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(", ".join(cols) + "\n")
        for r in rows:
            fh.write(", ".join(r) + "\n")
        if average:
            n = float(len(rows)) or 1.0
            avg = ["Average"]
            for c in range(1, len(cols)):
                try:
                    avg.append("%.4g" % (sum(float(r[c]) for r in rows) / n))
                except ValueError:
                    avg.append("0")
            fh.write(", ".join(avg) + "\n")


def run_checker(root, path, extra=()):
    checker = os.path.join(root, "test", "check_performance_identity.py")
    cmd = [sys.executable, checker, path] + list(extra)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate()
    return p.returncode, out.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# cases: (name, builder, expect_rc, extra_flags, substring_required_in_output)
# ---------------------------------------------------------------------------

def _good_exact(p):
    write(p, [row(0, 10.0, 3.0, 2.0, 4.0, 1.0),
              row(1, 6.0, 3.0, 2.0, 0, 1.0)])


def _good_within_bound(p):
    # Perturb one child by 0.9 of the derived bound. Bound is
    # 5e-4 * (sum|children| + |parent|) = 5e-4 * 20 = 0.01 for parent 10.
    write(p, [row(0, 10.0, 3.009, 2.0, 4.0, 1.0)])


def _good_all_zero(p):
    write(p, [row(0, 0, 0, 0, 0, 0), row(1, 0, 0, 0, 0, 0)])


def _good_single_rank(p):
    write(p, [row(0, 10.0, 3.0, 2.0, 4.0, 1.0)])


def _bad_escaped_child(p):
    """A child 40% LARGER than its whole parent. The four columns still sum to
    the parent exactly, because SWMM_OTHER is derived by subtraction -- this is
    the reviewer's row, and the identity alone passes it."""
    write(p, [row(0, 10.0, 14.0, 1.0, 2.0, -7.0)])


def _bad_overlapping_children(p):
    """Two children overlapping, so the shared region is counted twice. NO single
    child exceeds the parent (6, 6, 1 against 10), so the [0, parent] band does
    not catch it either -- only the residual's SIGN does."""
    write(p, [row(0, 10.0, 6.0, 6.0, 1.0, -3.0)])


def _bad_negative_measured_child(p):
    """A measured accumulator cannot go below zero: a parse or format fault."""
    write(p, [row(0, 10.0, -2.0, 5.0, 4.0, 3.0)])


def _good_child_equals_parent(p):
    """The boundary the [0, parent] band must NOT false-fail: all the parent's
    time in one child, with the child and the parent rounded independently."""
    write(p, [row(0, 10.0, 10.0, 0, 0, 0)])


def _good_zero_residual(p):
    """SWMM_OTHER exactly 0 is legal -- the band is >= 0, not > 0."""
    write(p, [row(0, 10.0, 5.0, 3.0, 2.0, 0)])


def _bad_sum_broken_high(p):
    """The four columns do NOT sum to the parent. Only a corrupted or mis-parsed
    row, or an edit to write_times' derivation, can produce this -- a bracket
    fault cannot, because the derivation makes the sum exact whatever the
    brackets do. Named for what it is, not for what the old wording implied."""
    write(p, [row(0, 10.0, 3.5, 2.0, 4.0, 1.0)])


def _bad_sum_broken_low(p):
    write(p, [row(0, 10.0, 1.0, 2.0, 4.0, 1.0)])


def _bad_step_off_rank0(p):
    write(p, [row(0, 10.0, 3.0, 2.0, 4.0, 1.0),
              row(1, 10.0, 3.0, 2.0, 4.0, 1.0)])


def _bad_no_average(p):
    write(p, [row(0, 10.0, 3.0, 2.0, 4.0, 1.0)], average=False)


def _bad_short_row(p):
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(", ".join(COLS) + "\n")
        fh.write("0, 1, 0.5, 0.1, 0, 10, 3, 2, 4\n")


def _bad_missing_column(p):
    cols = [c for c in COLS if c != "SWMM_OTHER"]
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(", ".join(cols) + "\n")
        fh.write(", ".join(["0", "1", "0.5", "0.1", "0", "10", "3", "2", "4",
                            "0.2", "3", "0.4", "3.4"]) + "\n")


CASES = (
    ("G1_exact_closure",              _good_exact,          0, (), None),
    ("G2_error_inside_print_bound",   _good_within_bound,   0, (), None),
    ("G3_all_zero_is_degenerate",     _good_all_zero,       0, (),
     "0 NON-degenerate"),
    ("G3b_all_zero_rejected_when_nondegenerate_required",
     _good_all_zero, 1, ("--require-nondegenerate",), "degenerate"),
    ("G4_single_rank",                _good_single_rank,    0, (), None),
    ("B1_sum_broken_high_is_not_a_bracket_fault",
     _bad_sum_broken_high, 1, (), "does not close"),
    ("B2_sum_broken_low_is_not_a_bracket_fault",
     _bad_sum_broken_low,  1, (), "does not close"),
    ("B3_step_nonzero_off_rank0",     _bad_step_off_rank0,  1, (),
     "nonzero on rank"),
    ("B4_average_row_missing",        _bad_no_average,      1, (), "Average row"),
    ("B5_row_arity_wrong",            _bad_short_row,       1, (), "arity"),
    ("B6_child_column_missing",       _bad_missing_column,  1, (), "SWMM_OTHER"),
    # The reviewer's two rows. Each PASSED the identity-only form of this
    # checker, because SWMM_OTHER is derived by subtraction and the sum is a
    # tautology over the emitted row. They are the reason (B) and (C) exist.
    ("B7_escaped_child_negative_residual", _bad_escaped_child, 1, (), "NEGATIVE"),
    ("B8_overlapping_children_negative_residual",
     _bad_overlapping_children, 1, (), "NEGATIVE"),
    ("B9_negative_measured_child",    _bad_negative_measured_child, 1, (),
     "cannot go below zero"),
    ("G5_child_equal_to_parent_is_legal", _good_child_equals_parent, 0, (), None),
    ("G6_zero_residual_is_legal",     _good_zero_residual,  0, (), None),
    # The flag surface: a mistyped option must not be silently dropped, because a
    # dropped --require-ranks is a PASS without the check the operator asked for.
    ("F1_space_separated_require_ranks_is_honoured",
     _good_exact, 1, ("--require-ranks", "7"), "expected 7 per-rank row"),
    ("F2_unrecognised_flag_is_rejected",
     _good_exact, 2, ("--require-rank=2",), "unrecognised option"),
    ("F3_require_ranks_without_a_value_is_rejected",
     _good_exact, 2, ("--require-ranks",), "needs a value"),
)


def main(argv):
    if len(argv) != 2:
        print("usage: %s <repo-root>" % os.path.basename(argv[0]))
        return 2
    root = os.path.abspath(argv[1])
    if not os.path.exists(os.path.join(root, "test",
                                       "check_performance_identity.py")):
        print("ERROR: no check_performance_identity.py under %s/test" % root)
        return 2

    failures = []
    tmp = tempfile.mkdtemp(prefix="wp1a-perfcheck-")
    try:
        for name, build, want_rc, extra, needle in CASES:
            path = os.path.join(tmp, name + ".txt")
            build(path)
            rc, out = run_checker(root, path, extra)
            ok = (rc == want_rc)
            if ok and needle is not None and needle not in out:
                ok = False
                failures.append("%s: exit %d as expected, but the output does "
                                "not mention %r -- the checker may be passing or "
                                "failing for the wrong reason.\n%s"
                                % (name, rc, needle, out))
            elif not ok:
                failures.append("%s: expected exit %d, got %d.\n%s"
                                % (name, want_rc, rc, out))
            print("%-5s %-52s exit %d (wanted %d)"
                  % ("ok" if ok else "FAIL", name, rc, want_rc))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: the checker separates %d known-good from %d known-bad files, "
          "and each verdict is reached for the stated reason."
          % (sum(1 for c in CASES if c[2] == 0),
             sum(1 for c in CASES if c[2] != 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

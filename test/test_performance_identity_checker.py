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

  B1  a child inflated past the print bound      -- a bracket double-counted
  B2  a child deflated past the print bound      -- a bracket escaped the parent
  B3  SWMM_STEP nonzero on a rank other than 0   -- the bracket left its guard
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


def _bad_child_inflated(p):
    write(p, [row(0, 10.0, 3.5, 2.0, 4.0, 1.0)])


def _bad_child_deflated(p):
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
    ("B1_child_inflated",             _bad_child_inflated,  1, (), "closure FAILS"),
    ("B2_child_deflated",             _bad_child_deflated,  1, (), "closure FAILS"),
    ("B3_step_nonzero_off_rank0",     _bad_step_off_rank0,  1, (),
     "nonzero on rank"),
    ("B4_average_row_missing",        _bad_no_average,      1, (), "Average row"),
    ("B5_row_arity_wrong",            _bad_short_row,       1, (), "arity"),
    ("B6_child_column_missing",       _bad_missing_column,  1, (), "SWMM_OTHER"),
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

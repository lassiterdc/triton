#!/usr/bin/env python3
"""REVIEWER test for the SWMM column split (WP-1A): the residual's SIGN.

WHY THIS EXISTS
---------------
WP-1A's stated acceptance property is

    SWMM_XFER + SWMM_MPI + SWMM_STEP + SWMM_OTHER == SWMM

on every emitted per-rank row, and both landed instruments assert it.  But
``SWMM_OTHER`` is DEFINED in ``output.h::write_times`` as

    swmm_other_time = swmm_time - swmm_xfer_time - swmm_mpi_time - swmm_step_time

so the sum of the four columns is the parent BY CONSTRUCTION, for ANY values the
three measured children happen to hold -- including values no correct bracket
placement could produce.  The identity is an ALGEBRAIC TAUTOLOGY over the emitted
row.  It can detect a print, parse or arity fault, and it can detect an edit to
the derivation itself.  It cannot detect a bracket fault, because a bracket fault
does not change the sum; it changes the SIGN and MAGNITUDE of the residual term.

That distinction is not academic here.  ``check_performance_identity.py``'s own
failure text tells its reader the opposite:

    "The residual column is computed by SUBTRACTION from the parent, so a miss
     larger than the print bound means a child bracket escaped the parent or was
     double-counted."

Those two causes are exactly the ones that produce NO miss.  A reviewer or an
operator who trusts that sentence will read a green run as evidence about
bracket placement, which it is not.

WHAT ACTUALLY FALSIFIES A BRACKET FAULT is the residual's SIGN.  Every child
bracket is nested inside the parent, so the measured children cannot sum to more
than the parent and ``SWMM_OTHER`` cannot be negative.  A child that escaped the
parent, or two children that overlap and double-count the region between them,
drive the residual NEGATIVE while leaving the identity exact.  ``SWMM_OTHER < 0``
is therefore the assertion that carries the weight the identity is credited with.

The landed C++ test DOES make that assertion -- ``test_perf_identity.cpp`` P4
checks ``other >= 0.0``.  But it makes it against its own in-process harness,
whose brackets are correctly nested by construction, so it cannot observe a fault
in ``triton.h`` or in a real ``performance.txt``.  The two artifacts that meet
real data -- the operator-facing checker and the source guard -- are the two that
do not make it.

THE TWO CASES
-------------
C1  the operator-facing checker, pointed at a per-rank row carrying the
    signature of an escaped or double-counted child (a child exceeding its
    parent, residual negative), must FAIL that row.  It currently PASSES it.

C2  the source guard must go RED when two child brackets OVERLAP inside the
    parent -- the general form of the defect its M3 mutation covers only in the
    one special case of SWMM_MPI collapsing to a single pair.  S2 tests each
    child's bracket lines for containment in the parent and never tests the
    children against EACH OTHER, so an overlap passes.

Both cases are expected RED against the tree as landed.  This test is the
reviewer's finding in executable form; the repair belongs to the package author.

USAGE
-----
    test_swmm_column_split_residual_sign.py <repo-root>
"""

import os
import shutil
import subprocess
import sys
import tempfile

CHECKER = "check_performance_identity.py"
GUARD = "test_swmm_column_split_source.py"

HEADER = ("%Rank, Compute, MPI, IO, Resize, SWMM, SWMM_XFER, SWMM_MPI, "
          "SWMM_STEP, SWMM_OTHER, Other, Simulation, Init, Total")


def _write_perf(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HEADER + "\n")
        for r in rows:
            fh.write(", ".join(str(c) for c in r) + "\n")


def C1_checker_rejects_a_negative_residual(root, failures, report):
    """A child larger than its parent must not pass the construction check."""
    checker = os.path.join(root, "test", CHECKER)
    if not os.path.exists(checker):
        failures.append("no %s at %s" % (CHECKER, checker))
        return

    tmp = tempfile.mkdtemp(prefix="wp1a-residual-sign-")
    try:
        # SWMM_XFER = 14 against a parent SWMM = 10.  The identity is exact --
        # 14 + 1 + 2 + (-7) == 10 -- and the row is impossible: a bracket nested
        # inside the parent cannot outrun it.  This is what an escaped XFER
        # bracket (one that started before st.start(SWMM_TIME), swallowing the
        # hydro drain the WP-0B fence moved into COMPUTE) emits.
        escaped = os.path.join(tmp, "performance_escaped.txt")
        _write_perf(escaped, [
            [0, 50, 1, 1, 0, 10, 14, 1, 2, -7, 1, 63, 2, 65],
            [1, 50, 1, 1, 0, 10, 14, 1, 0, -5, 1, 63, 2, 65],
            ["Average", 50, 1, 1, 0, 10, 14, 1, 1, -6, 1, 63, 2, 65],
        ])

        # Two children that OVERLAP: each is individually smaller than the
        # parent, so a per-child sanity bound would miss it, but together they
        # exceed it because the region between them is counted twice.
        overlap = os.path.join(tmp, "performance_overlap.txt")
        _write_perf(overlap, [
            [0, 50, 1, 1, 0, 10, 6, 6, 1, -3, 1, 63, 2, 65],
            [1, 50, 1, 1, 0, 10, 6, 6, 0, -2, 1, 63, 2, 65],
            ["Average", 50, 1, 1, 0, 10, 6, 6, 0.5, -2.5, 1, 63, 2, 65],
        ])

        for label, path in (("escaped child", escaped),
                            ("overlapping children", overlap)):
            proc = subprocess.run([sys.executable, checker, path],
                                  capture_output=True, text=True)
            report.append("%-22s : checker exit %d" % (label, proc.returncode))
            if proc.returncode == 0:
                failures.append(
                    "%s: %s PASSED a row whose residual is NEGATIVE. The four "
                    "columns sum to the parent because SWMM_OTHER is derived by "
                    "subtraction, so the identity holds for any children "
                    "whatsoever. The assertion that carries the bracket-placement "
                    "claim is SWMM_OTHER >= 0, and the checker does not make it. "
                    "Its own failure text credits the identity with detecting "
                    "exactly this cause."
                    % (label, CHECKER))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def C2_source_guard_catches_overlapping_child_brackets(root, failures, report):
    """Two children may not overlap; S2 tests containment but not disjointness."""
    guard = os.path.join(root, "test", GUARD)
    if not os.path.exists(guard):
        failures.append("no %s at %s" % (GUARD, guard))
        return

    tmp = tempfile.mkdtemp(prefix="wp1a-overlap-mutation-")
    try:
        for sub in ("src", "test"):
            shutil.copytree(os.path.join(root, sub), os.path.join(tmp, sub))

        path = os.path.join(tmp, "src", "triton.h")
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()

        # THE MUTATION. Move st.stop(SWMM_XFER) to after st.start(SWMM_MPI).
        # Both brackets remain strictly inside the parent and SWMM_MPI keeps its
        # required two pairs, so every predicate S2 evaluates still holds -- but
        # the MPI_Gatherv region is now inside BOTH children and is counted
        # twice, which is the defect S2's own comment says a collapsed SWMM_MPI
        # bracket would cause.
        stop_xfer = "      st.stop(SWMM_XFER);\n"
        anchor = "      st.start(SWMM_MPI);\n      // Gather exchange_q"
        if text.count(stop_xfer) != 1 or text.count(anchor) != 1:
            failures.append(
                "the overlap mutation could not be anchored in triton.h; the "
                "bracket layout moved and this reviewer test needs re-aiming "
                "(stop_xfer=%d anchor=%d)"
                % (text.count(stop_xfer), text.count(anchor)))
            return
        text = text.replace(stop_xfer, "", 1)
        text = text.replace(
            anchor,
            "      st.start(SWMM_MPI);\n" + stop_xfer + "      // Gather exchange_q",
            1)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

        proc = subprocess.run(
            [sys.executable, os.path.join(tmp, "test", GUARD), tmp],
            capture_output=True, text=True)
        report.append("overlap mutation       : source guard exit %d"
                      % proc.returncode)
        if proc.returncode == 0:
            failures.append(
                "the source guard PASSED a tree in which SWMM_XFER and SWMM_MPI "
                "overlap. S2 checks that each child's bracket lines lie between "
                "the parent's, and pins SWMM_MPI at exactly two start brackets; "
                "it never checks the children against EACH OTHER. The general "
                "predicate is that the three child spans are pairwise DISJOINT, "
                "of which M3's collapsed-SWMM_MPI case is one instance. The "
                "runtime identity cannot cover the gap -- the overlap leaves the "
                "sum exact and drives the residual negative.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


CASES = (
    ("C1_checker_rejects_a_negative_residual",
     C1_checker_rejects_a_negative_residual),
    ("C2_source_guard_catches_overlapping_child_brackets",
     C2_source_guard_catches_overlapping_child_brackets),
)


def main(argv):
    if len(argv) != 2:
        print(__doc__.split("USAGE")[-1].strip())
        return 2
    root = argv[1]
    if not os.path.isdir(os.path.join(root, "src")):
        print("ERROR: %s is not a repo root (no src/)" % root)
        return 2

    failures = []
    for name, fn in CASES:
        report = []
        print("RUN  %s" % name)
        before = len(failures)
        fn(root, failures, report)
        for line in report:
            print("    %s" % line)
        print("%s %s\n" % ("PASS" if len(failures) == before else "FAIL", name))

    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        print("\n%d finding(s). The residual's SIGN, not the identity, is what "
              "falsifies a bracket fault." % len(failures))
        return 1
    print("PASS: a negative SWMM_OTHER is rejected by the operator-facing "
          "checker and overlapping child brackets are rejected by the source "
          "guard.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

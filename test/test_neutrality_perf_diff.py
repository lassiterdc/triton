#!/usr/bin/env python3
"""Unit test of the per-rank restatement of Invariant B in neutrality_compare.py perf-diff.

The problem this guards, INFERRED and not tested (no run varied the capture set): a rank that
carries less DIAG may wait in MPI for a rank that carries more; on the A100 np2 runs the DIAG gap
explains 81 to 93 percent of the rank 1 MPI shift (12 of 12 per-run comparisons: the delta exceeds
the gap by +0.0265 to +0.0839 s, gap 0.324 to 0.356 s, the rest inside the band). The restated check lets
the MPI column of rank r rise by max_k(DIAG_k) - DIAG_r (the DIAG of the capture-on run) beyond
its band, and nothing else moves: Compute, Other and IO stay in band on every rank, and an MPI
rise above the gap plus the band still fails, so MPI cost that is not capture waiting is caught.

Each case writes synthetic performance.txt files (three capture-off runs, one capture-on run) into
a temporary directory and runs the comparator's perf-diff as a subprocess, reading its exit status.
The A100 rows use the numbers of the six-run tables of the repeat-run round: rank 1 MPI off mean
0.4138 (gd0) and 0.6162 (gd1), on 0.7898 and 1.007, DIAG on 1.454/1.122 and 1.467/1.128.

Usage:  python3 test/test_neutrality_perf_diff.py      (exit 0 only when every case holds)
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(HERE, "neutrality_compare.py")
HEADER = "%Rank, Compute, MPI, IO, Other, DIAG, SWMM_MPI"


def write_run(d, ranks):
    """ranks: {rank: (Compute, MPI, IO, Other, DIAG[, SWMM_MPI])}; SWMM_MPI defaults to 0."""
    os.makedirs(d)
    with open(os.path.join(d, "performance.txt"), "w") as f:
        f.write(HEADER + "\n")
        for r in sorted(ranks):
            row = tuple(ranks[r]) + (0.0,) * (6 - len(ranks[r]))
            f.write("%d, %s\n" % (r, ", ".join("%g" % v for v in row)))


def run_case(off_rows, on_row, cols="Compute,MPI,IO,Other"):
    """Return the perf-diff exit status for three capture-off runs and one capture-on run."""
    tmp = tempfile.mkdtemp(prefix="perf_diff_test_")
    try:
        offs = []
        for i, rows in enumerate(off_rows):
            d = os.path.join(tmp, "off%d" % i)
            write_run(d, rows)
            offs.append(d)
        on = os.path.join(tmp, "on")
        write_run(on, on_row)
        p = subprocess.Popen([sys.executable, "-I", COMPARE, "perf-diff", "--on", on, "--off"] + offs + ["--cols", cols],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.communicate()[0].decode()
        return p.returncode, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def spread3(mean, spread):
    """Three values with the given mean and max-min spread."""
    return [mean - spread / 2.0, mean, mean + spread / 2.0]


def a100_case(off_mpi1, spr_mpi1, on_mpi1, diag0, diag1):
    """The A100 np2 numbers of the repeat-run tables (rank 1 MPI off mean, off spread, on mean, DIAG on)."""
    offs = []
    for v in spread3(off_mpi1, spr_mpi1):
        offs.append({0: (0.63, 0.05, 0.25, 0.026, 0.0), 1: (0.58, v, 0.17, 0.10, 0.0)})
    on = {0: (0.65, 0.07, 0.33, 0.029, diag0), 1: (0.60, on_mpi1, 0.24, 0.113, diag1)}
    return offs, on


def synthetic_base():
    offs = [{0: (3.0, 0.5, 1.0, 0.05, 0.0), 1: (3.0, 0.5, 1.0, 0.05, 0.0)} for _ in range(3)]
    return offs


def main():
    fails = []

    def expect(name, got, want_rc):
        rc, out = got
        status = "ok" if rc == want_rc else "WRONG"
        print("%-62s exit %d (wanted %d) %s" % (name, rc, want_rc, status))
        if rc != want_rc:
            fails.append(name)
            print(out)

    # Real-data rows: the two A100 np2 cases that failed under the unrestated check pass.
    expect("A100 np2 gd0 rank 1 MPI +0.376, gap 0.332: passes", run_case(*a100_case(0.4138, 0.0313, 0.7898, 1.454, 1.122)), 0)
    expect("A100 np2 gd1 rank 1 MPI +0.391, gap 0.339: passes", run_case(*a100_case(0.6162, 0.0346, 1.007, 1.467, 1.128)), 0)

    # Synthetic: the busiest rank is rank 0 (DIAG 1.0), rank 1 has DIAG 0.4, gap 0.6; MPI band here is 0.25 (0.5 x 0.5).
    # Delta 0.8 <= 0.6 + 0.25: passes. Delta 1.0 > 0.85: fails.
    offs = synthetic_base()
    expect("MPI delta within gap + band passes",
           run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 1.3, 1.0, 0.05, 0.4)}), 0)
    expect("MPI delta above gap + band FAILS",
           run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 1.5, 1.0, 0.05, 0.4)}), 1)
    # The busiest rank has no allowance: an MPI rise there above its band fails.
    expect("MPI rise on the busiest rank above its band FAILS",
           run_case(offs, {0: (3.0, 0.9, 1.0, 0.05, 1.0), 1: (3.0, 0.5, 1.0, 0.05, 0.4)}), 1)
    # With no DIAG gap (equal DIAG) the unrestated band applies.
    expect("MPI rise above band with equal DIAG FAILS",
           run_case(offs, {0: (3.0, 0.9, 1.0, 0.05, 1.0), 1: (3.0, 0.5, 1.0, 0.05, 1.0)}), 1)
    expect("MPI fall below the band FAILS despite a gap",
           run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 0.1, 1.0, 0.05, 0.4)}), 1)

    # Compute, Other and IO excursions still fail on any rank, gap or no gap.
    expect("Compute excursion on a rank FAILS", run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.8, 0.5, 1.0, 0.05, 0.4)}), 1)
    expect("Other excursion on a rank FAILS", run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 0.5, 1.0, 0.25, 0.4)}), 1)
    expect("IO excursion on a rank FAILS", run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 0.5, 1.8, 0.05, 0.4)}), 1)
    # SWMM_MPI is a different column from MPI: an allowance that leaked to it would pass this delta
    # of 0.4 (band 0.05, gap 0.6), so the case must FAIL.
    expect("SWMM_MPI excursion on a rank FAILS (no allowance)",
           run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0, 0.0), 1: (3.0, 0.5, 1.0, 0.05, 0.4, 0.4)}, cols="SWMM_MPI"), 1)
    # Control that must pass: the same rows with every excursion inside its band.
    expect("all columns inside their bands passes", run_case(offs, {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 0.6, 1.1, 0.06, 0.4)}), 0)

    print("PASS" if not fails else "FAIL: %s" % fails)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

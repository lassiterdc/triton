#!/usr/bin/env python3
"""Construction check for the SWMM column split (WP-1A chunk 4).

THE PROPERTY
------------
    SWMM_XFER + SWMM_MPI + SWMM_STEP + SWMM_OTHER == SWMM
    on EVERY emitted per-rank row.

Scoped to the per-rank rows.  The ``Average`` row is deliberately OUTSIDE the
claim and is checked only for presence and arity -- see THE AVERAGE ROW below.

WHY THE THRESHOLD IS A RATIO AND WHERE IT COMES FROM
----------------------------------------------------
``write_times`` emits through ``std::setprecision(4)`` in the default
floating-point format, which is FOUR SIGNIFICANT DIGITS -- not four decimals.
A printed value ``v_hat`` of a true value ``v`` therefore satisfies

    |v_hat - v| <= 0.5 * 10 ** (floor(log10|v|) - 3) <= 5e-4 * |v|

so the relative print error of any single field is at most ``5e-4``.  The
identity holds exactly in the emitting process's doubles, because SWMM_OTHER is
DERIVED as ``swmm - xfer - mpi - step``.  What the file can show is therefore the
identity plus five roundings, and the bound on their sum is

    |sum(children_hat) - parent_hat|  <=  5e-4 * (sum|children_hat| + |parent_hat|)

which, when the children are non-negative and sum to the parent, is exactly
``1e-3 * parent`` -- i.e. 0.1 %.  The bound is DERIVED from the emit precision
rather than chosen, and it is written in the measurable form above so it stays
correct if a residual ever prints negative.

An absolute epsilon is deliberately NOT used: at four SIGNIFICANT digits the
representable error scales with the value, so any fixed epsilon is either vacuous
on a long production run or false-failing on a short synthetic one.

THE AVERAGE ROW
---------------
``write_times`` emits N per-rank rows and ONE ``Average`` row whose every column
is an arithmetic mean over ranks.  ``SWMM_STEP`` is exactly 0.0 on the N-1 ranks
that do not run the serial solve, so ``Average(SWMM_STEP)`` is ``rank0/N`` -- a
number that halves as rank count doubles while the serial solve is constant.
That is not a defect to repair here and three repairs were rejected by name in
the design: emitting ``max`` for one column (a per-column reduction asymmetry no
reader of the row can see), suppressing that column's Average (the row's arity is
fixed and the downstream parser's Average-presence detector depends on it), and
renaming the column ``SWMM_STEP_RANK0`` (rank-0-onlyness is a property of the
current coupling architecture, not of the measured quantity, so the name would go
SILENTLY stale if the step were ever distributed).  This checker therefore
asserts the Average row is PRESENT with full arity and asserts NOTHING about its
values.

DEGENERATE ROWS ARE REPORTED, NOT SILENTLY PASSED
--------------------------------------------------
In a non-coupled build -- or a coupled build whose rank owns no SWMM node -- all
five SWMM fields are 0.0 and the identity holds as ``0 == 0``.  That is a true
pass and a weak one.  The report names how many rows were NON-DEGENERATE, so a
reader cannot mistake "examined N, all trivial" for "examined N, all meaningful".

USAGE
-----
    check_performance_identity.py <performance.txt> [--require-nondegenerate]
                                                    [--require-ranks N]
"""

import math
import os
import re
import sys

PARENT = "SWMM"
CHILDREN = ("SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER")

# Half a unit in the 4th significant digit, as a relative bound. See module
# docstring: this is the direct consequence of std::setprecision(4) in the
# default float format and is not a tuned constant.
REL_PRINT_ERROR = 5.0e-4


def parse(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        raise ValueError("%s is empty" % path)

    header = lines[0]
    if not header.lstrip().startswith("%Rank"):
        raise ValueError("first line is not the %%Rank header: %r" % header[:80])
    cols = [c.strip() for c in header.lstrip("%").split(",")]

    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) != len(cols):
            raise ValueError(
                "row arity %d does not match header arity %d: %r"
                % (len(cells), len(cols), ln[:100]))
        rows.append(dict(zip(cols, cells)))
    return cols, rows


def check(path, require_nondegenerate=False, require_ranks=None):
    failures = []
    cols, rows = parse(path)

    for want in (PARENT,) + CHILDREN:
        if want not in cols:
            failures.append("header has no %r column (has %s)" % (want, cols))
    if failures:
        return failures, []

    report = ["file              : %s" % path,
              "columns           : %d -> %s" % (len(cols), ", ".join(cols))]

    per_rank = [r for r in rows if r["Rank"] != "Average"]
    average = [r for r in rows if r["Rank"] == "Average"]

    # Arity, not values. See THE AVERAGE ROW in the module docstring.
    if len(average) != 1:
        failures.append(
            "expected exactly 1 Average row, found %d. The row's arity is fixed "
            "and the downstream parser's Average-presence detector keys on it."
            % len(average))
    else:
        missing = [c for c in cols if not average[0].get(c, "").strip()]
        if missing:
            failures.append("the Average row has empty cell(s) %s; every column "
                            "must be emitted" % missing)

    if not per_rank:
        failures.append("no per-rank rows found; the identity is scoped to them")
        return failures, report

    if require_ranks is not None and len(per_rank) != require_ranks:
        failures.append("expected %d per-rank row(s), found %d"
                        % (require_ranks, len(per_rank)))

    nondegenerate = 0
    step_nonzero_ranks = []
    worst = None

    for r in per_rank:
        try:
            parent = float(r[PARENT])
            kids = [float(r[c]) for c in CHILDREN]
        except ValueError as exc:
            failures.append("rank %s: unparseable cell (%s)" % (r["Rank"], exc))
            continue

        total = sum(kids)
        residual = total - parent
        bound = REL_PRINT_ERROR * (sum(abs(k) for k in kids) + abs(parent))

        if abs(parent) > 0.0 or any(k != 0.0 for k in kids):
            nondegenerate += 1
        if float(r["SWMM_STEP"]) != 0.0:
            step_nonzero_ranks.append(r["Rank"])

        ratio = (abs(residual) / bound) if bound > 0 else (
            0.0 if residual == 0.0 else math.inf)
        if worst is None or ratio > worst[0]:
            worst = (ratio, r["Rank"], parent, kids, residual, bound)

        if abs(residual) > bound:
            failures.append(
                "rank %s: closure FAILS. %s = %r but %s sum to %r "
                "(residual %+.6g, derived bound %.6g at 4 significant digits). "
                "The residual column is computed by SUBTRACTION from the parent, "
                "so a miss larger than the print bound means a child bracket "
                "escaped the parent or was double-counted."
                % (r["Rank"], PARENT, parent, "+".join(CHILDREN), total,
                   residual, bound))

    report.append("per-rank rows     : %d examined, %d NON-degenerate "
                  "(a row whose SWMM fields are all 0.0 passes trivially)"
                  % (len(per_rank), nondegenerate))
    if worst is not None:
        ratio, rk, parent, kids, residual, bound = worst
        report.append("worst row         : rank %s  SWMM=%.6g  children=%s  "
                      "residual=%+.6g  bound=%.6g  (%.3f of bound)"
                      % (rk, parent, ["%.6g" % k for k in kids], residual,
                         bound, ratio))
    report.append("SWMM_STEP nonzero : rank(s) %s"
                  % (step_nonzero_ranks if step_nonzero_ranks else "none"))

    # The rank-0-only property. Not an error on its own -- a single-rank run has
    # rank 0 nonzero and nothing else, and a fully idle coupling has none -- but a
    # nonzero on a rank OTHER than 0 means the bracket escaped its guard, which is
    # exactly the placement decision this split made deliberately.
    stray = [rk for rk in step_nonzero_ranks if rk != "0"]
    if stray:
        failures.append(
            "SWMM_STEP is nonzero on rank(s) %s. Its bracket must sit INSIDE the "
            "`if (rank == 0)` guard, so every other rank reports exactly 0.0 "
            "rather than a small time for a branch test it evaluated and a solve "
            "it never performed." % stray)

    if require_nondegenerate and nondegenerate == 0:
        failures.append(
            "every per-rank row is degenerate (all SWMM fields 0.0). The "
            "identity held, but on nothing. Run this against a COUPLED run whose "
            "ranks own SWMM nodes to exercise it.")

    return failures, report


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = [a for a in argv[1:] if a.startswith("--")]
    if len(args) != 1:
        print(__doc__.split("USAGE")[-1].strip())
        return 2
    path = args[0]
    if not os.path.exists(path):
        print("ERROR: no such file: %s" % path)
        return 2

    require_ranks = None
    for f in flags:
        m = re.match(r'^--require-ranks=(\d+)$', f)
        if m:
            require_ranks = int(m.group(1))

    try:
        failures, report = check(path,
                                 require_nondegenerate="--require-nondegenerate" in flags,
                                 require_ranks=require_ranks)
    except ValueError as exc:
        print("FAIL: %s" % exc)
        return 1

    for line in report:
        print("    %s" % line)
    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: XFER + MPI + STEP + OTHER == SWMM on every per-rank row, "
          "within the bound derived from setprecision(4).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

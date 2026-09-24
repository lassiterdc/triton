#!/usr/bin/env python3
"""Construction check for the SWMM column split (WP-1A chunk 4).

WHAT THIS CHECKS, AND WHAT THE IDENTITY IS WORTH
-------------------------------------------------
Three properties, on EVERY emitted per-rank row.  They are listed in order of
what they can falsify, because the first one used to be presented as the whole
check and it is the weakest of the three:

    (A) SWMM_XFER + SWMM_MPI + SWMM_STEP + SWMM_OTHER == SWMM
    (B) SWMM_OTHER >= 0
    (C) each measured child is >= 0 and <= SWMM

**(A) IS AN ALGEBRAIC TAUTOLOGY OVER THE EMITTED ROW AND MUST NOT BE CREDITED
WITH MORE.**  ``output.h::write_times`` DERIVES the residual as

    swmm_other_time = swmm_time - swmm_xfer_time - swmm_mpi_time - swmm_step_time

so the four columns sum to the parent BY CONSTRUCTION, for ANY values the three
measured children hold -- including values no correct bracket placement could
produce.  (A) detects a print fault, a parse fault, an arity fault, and an edit
to the derivation itself.  **It cannot detect a bracket fault**, because a
bracket fault does not change the sum; it changes the SIGN of the residual.

**(B) IS THE ASSERTION THAT CARRIES THE BRACKET-PLACEMENT CLAIM.**  Every child
bracket is nested inside the parent, and every accumulator is a sum of
non-negative durations, so the measured children cannot sum to more than the
parent and the residual cannot be negative.  A child that ESCAPED the parent, or
two children that OVERLAP and double-count the region between them, drive
SWMM_OTHER negative while leaving (A) exact to the last digit.

**(C) names the offending child**, which (B) cannot.  It is implied by (B)
whenever the measured children are non-negative, and it is kept separately
because a failure report that says which column is wrong is worth more than one
that says only that something is.

This ordering was not the original design and the correction came from review:
the first version of this file asserted (A) alone, and its failure text told the
reader that a miss meant "a child bracket escaped the parent or was
double-counted" -- naming the two causes that produce NO miss.  Two constructed
rows passed it: a child 40% larger than its whole parent, and an overlapping
double-counted pair.  Both are rejected now, by (B) and (C).

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

**THE SIGN CHECK (B) TAKES NO TOLERANCE AT ALL, AND THAT IS THE POINT.**  The
true residual is non-negative EXACTLY -- the accumulators are unsigned sums of
non-negative durations and the children's intervals are subsets of the parent's
-- and printing is a faithful rounding, so a non-negative true value can never
print negative.  A negative cell therefore means a negative true value, with no
rounding story available.

That matters because the bound above is the SUM over ``|children| + |parent|``,
which a negative residual INFLATES: on the reviewer's escaped-child row it rose
from 0.010 to 0.017, i.e. the tolerance grew by 70% in exactly the case the
instrument most needed to be strict.  The form is still the right one -- it stays
correct if a term ever prints negative -- but it was doing the wrong job while
(B) was missing.  With (B) in place, every row that reaches the (A) comparison
has already been shown to have non-negative terms, so ``sum|children|`` equals
``sum children`` equals the parent and the bound collapses to the intended
``1e-3 * parent``.

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
                                                    [--require-ranks=N]

``--require-ranks`` takes an ``=``-joined value.  The space-separated form is
also accepted, because an operator will type it; what is NOT accepted is an
unrecognised or malformed flag, which now exits 2 with the offending token named
rather than being silently ignored.  A silently-ignored ``--require-ranks 4`` is
a run that reports PASS without having made the check the operator asked for --
the same class of defect as the tautology this file was reviewed for, in a
smaller place.
"""

import math
import os
import re
import sys

PARENT = "SWMM"
CHILDREN = ("SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER")
# The three MEASURED children. SWMM_OTHER is excluded because it is the
# DERIVED residual: it is legitimately allowed to be small, and its own
# admissible range is the sign check rather than the [0, parent] band.
MEASURED_CHILDREN = ("SWMM_XFER", "SWMM_MPI", "SWMM_STEP")

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
    negative_residual_ranks = []
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

        # ---- (B) the residual's SIGN. No tolerance, on the ground in the
        # module docstring: the true residual is non-negative exactly, and a
        # faithful rounding of a non-negative value is never negative. This is
        # the assertion that carries the bracket-placement claim; (A) below
        # cannot make it, because a bracket fault leaves the sum exact.
        other = float(r["SWMM_OTHER"])
        if other < 0.0:
            negative_residual_ranks.append(r["Rank"])
            failures.append(
                "rank %s: SWMM_OTHER = %.6g, which is NEGATIVE. Every child "
                "bracket nests inside %s and every accumulator is a sum of "
                "non-negative durations, so the measured children cannot sum to "
                "more than the parent. A negative residual is the signature of a "
                "child bracket that ESCAPED the parent, or of two children that "
                "OVERLAP and double-count the region between them. Note that the "
                "four columns still sum to %s exactly in this row -- SWMM_OTHER "
                "is derived by subtraction, so the sum is a tautology and cannot "
                "detect this. The sign can."
                % (r["Rank"], other, PARENT, PARENT))

        # ---- (C) each MEASURED child within [0, parent]. Implied by (B) when
        # the children are non-negative, kept because it names the offender.
        # The upper comparison carries the print bound: a child that legitimately
        # equals the parent is rounded independently of it.
        for name, value in zip(MEASURED_CHILDREN, kids[:len(MEASURED_CHILDREN)]):
            if value < 0.0:
                failures.append(
                    "rank %s: %s = %.6g, which is NEGATIVE. A measured timer "
                    "accumulates unsigned microsecond deltas and cannot go below "
                    "zero; this is a parse or format fault, not a timing result."
                    % (r["Rank"], name, value))
            slack = REL_PRINT_ERROR * (abs(value) + abs(parent))
            if value > parent + slack:
                failures.append(
                    "rank %s: %s = %.6g EXCEEDS its parent %s = %.6g by %.6g "
                    "(print slack %.6g). A child bracket is nested inside the "
                    "parent's, so it cannot measure more time than the parent "
                    "does. Either its bracket escaped the parent bracket, or it "
                    "overlaps another child and the shared region is counted "
                    "twice."
                    % (r["Rank"], name, value, PARENT, parent,
                       value - parent, slack))

        # ---- (A) the identity. Real but WEAK: it is a tautology over the
        # emitted row, so it detects a print, parse or arity fault and an edit to
        # the derivation in write_times -- NOT a bracket fault. The failure text
        # says so, because the previous wording named the two causes that
        # produce no miss and a reader who trusted it would read a green run as
        # evidence about bracket placement.
        if abs(residual) > bound:
            failures.append(
                "rank %s: the emitted row does not close. %s = %r but %s sum to "
                "%r (residual %+.6g, derived bound %.6g at 4 significant "
                "digits). Because SWMM_OTHER is DERIVED as the parent minus the "
                "three measured children, this sum is exact by construction -- "
                "so a miss here is NOT a bracket fault. It means the emitted "
                "text has been corrupted, mis-parsed or mis-rounded, or that the "
                "derivation in write_times has been edited. A bracket fault "
                "shows up as a NEGATIVE SWMM_OTHER, which is checked separately."
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
    report.append("SWMM_OTHER < 0    : rank(s) %s  <- the bracket-fault signal"
                  % (negative_residual_ranks if negative_residual_ranks
                     else "none (required)"))

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
    raw = argv[1:]
    flags = [a for a in raw if a.startswith("--")]
    args = [a for a in raw if not a.startswith("--")]
    # Recover the value of a space-separated --require-ranks so it is not
    # mistaken for the performance.txt path.
    pending_rank_value = None
    if "--require-ranks" in raw:
        k = raw.index("--require-ranks")
        if k + 1 < len(raw) and raw[k + 1].isdigit():
            pending_rank_value = int(raw[k + 1])
            args = [a for a in args if a != raw[k + 1]]
    if len(args) != 1:
        print(__doc__.split("USAGE")[-1].strip())
        return 2
    path = args[0]
    if not os.path.exists(path):
        print("ERROR: no such file: %s" % path)
        return 2

    require_ranks = None
    i = 0
    while i < len(flags):
        f = flags[i]
        m = re.match(r'^--require-ranks=(\d+)$', f)
        if m:
            require_ranks = int(m.group(1))
        elif f == "--require-ranks":
            # The space-separated form an operator will type. Its value was
            # parsed out of argv as a positional, so it is recovered from there.
            if pending_rank_value is None:
                print("ERROR: --require-ranks needs a value, e.g. "
                      "--require-ranks=4")
                return 2
            require_ranks = pending_rank_value
        elif f == "--require-nondegenerate":
            pass
        else:
            # NOT silently ignored. A mistyped flag that is dropped on the floor
            # produces a PASS without the check the operator asked for, which is
            # the same class of defect this file was reviewed for.
            print("ERROR: unrecognised option %r. Valid options are "
                  "--require-nondegenerate and --require-ranks=N." % f)
            return 2
        i += 1

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
    print("PASS: on every per-rank row, SWMM_OTHER >= 0 and each measured child "
          "lies within [0, SWMM] -- so no child bracket escaped the parent or "
          "overlaps another; and the four columns sum to SWMM within the bound "
          "derived from setprecision(4), which confirms the row was emitted and "
          "parsed intact. The sum is a tautology over the emitted row and is "
          "reported as the weaker of the two.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

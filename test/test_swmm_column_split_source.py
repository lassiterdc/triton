#!/usr/bin/env python3
"""Structural guards for the SWMM timer column split (WP-1A).

WHAT THIS FALSIFIES
-------------------
``SWMM_TIME`` had exactly one start/stop pair spanning a block that contains a
Kokkos kernel, two host-device transfers and several device-wide
synchronizations, so the column reported as serial-SWMM time was a coupled-BLOCK
timer containing GPU work.  The split adds three MEASURED children inside the
UNCHANGED parent bracket plus one DERIVED residual, so the level closes exactly:

    SWMM_XFER + SWMM_MPI + SWMM_STEP + SWMM_OTHER == SWMM

The runtime identity is held by the companion test ``test_perf_identity.cpp``,
which exercises the real ``output<T>::write_times`` emit path.  THIS file holds
the properties that are properties of the SOURCE rather than of any run, and
that a passing runtime test cannot distinguish from their violations:

  S1  Exactly three timer-name macros are declared, and ``SWMM_OTHER`` is NOT
      one of them.  This is the whole ground of the three-not-four decision, and
      it is a compiler rather than a comment: ``super_timer::get_cat_index_``
      REGISTERS an absent category instead of failing, so
      ``get_custom_time(SWMM_OTHER)`` would compile and silently return 0.0.
      With no symbol declared, that call cannot be written.
  S2  The parent bracket is unchanged and every child bracket lies strictly
      inside it.  A child that escaped the parent would still emit a number and
      would still look plausible; only the closure would break.
  S3  ``SWMM_STEP``'s bracket is INSIDE the ``if (rank == 0)`` guard.  Both
      placements are correct accounting.  INSIDE is the one that reports exactly
      0.0 on the N-1 ranks that never perform the solve, rather than a small
      nonzero time for a branch test.
  S4  ``write_times`` derives ``SWMM_OTHER`` by subtraction, leaves the
      pre-existing ``other_time`` residual untouched, and emits all four columns
      in the header, in every per-rank row, and in the Average row.
  S6  The three child spans are pairwise DISJOINT. S2 tests containment in the
      parent and never tests the children against each other, so an overlap
      passes it — and an overlap is invisible to every instrument that reads the
      emitted sum, because the derived residual keeps that sum exact and goes
      negative instead. Added on review.
  S5  No timer API call takes a bare string literal.  This is the residual hole
      S1 cannot close: ``start``/``stop``/``get_custom_time`` take
      ``std::string``, so ``get_custom_time("swmm_other")`` compiles even with no
      macro declared, registers a new empty category and returns 0.0.  S1 removes
      the symbol; S5 removes the literal; together they close the hole.

USAGE
-----
    test_swmm_column_split_source.py <subtest-name> <repo-root>

With no subtest name every subtest runs.  One CTest node id per subtest, so a
result is accounted by name rather than by a pass/fail total that cannot say
which property it contains.
"""

import os
import re
import sys

# The three MEASURED children.  SWMM_OTHER is deliberately absent: it is the
# derived residual and takes no macro, exactly as `Other` and `Init` take none.
CHILD_MACROS = ("SWMM_XFER", "SWMM_MPI", "SWMM_STEP")

# Every timer-category string value that must stay case-insensitively distinct.
# super_timer::timecats_ is keyed with a `ci_less` comparator, so two categories
# differing only in case would silently collide into one accumulator.
EXISTING_TIMER_MACROS = (
    "TOTAL_TIME", "SIMULATION_TIME", "COMPUTE_TIME", "MPI_TIME",
    "IO_TIME", "RESIZE_TIME", "BALANCING_MPI_TIME", "SWMM_TIME",
)

DEFINE_RE = re.compile(r'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+"([^"]*)"')

# A timer API call whose argument is a bare string literal rather than a macro.
BARE_LITERAL_CALL_RE = re.compile(
    r'\bst\s*\.\s*(?:start|stop|restart)\s*\(\s*"'
    r'|\bget_custom_time\s*\(\s*"'
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def strip_comments(lines):
    """Blank out // comments and /* */ blocks, preserving line numbering.

    A guard that matched inside a comment would be satisfied by prose describing
    the construct rather than by the construct, which is the failure mode the
    whole family of source guards exists to avoid.
    """
    out = []
    in_block = False
    for ln in lines:
        buf = []
        i = 0
        while i < len(ln):
            two = ln[i:i + 2]
            if in_block:
                if two == "*/":
                    in_block = False
                    i += 2
                    continue
                i += 1
                continue
            if two == "/*":
                in_block = True
                i += 2
                continue
            if two == "//":
                break
            buf.append(ln[i])
            i += 1
        out.append("".join(buf))
    return out


def find_one(code, pattern, what, failures):
    """Index of the single line matching `pattern`, or None with a failure."""
    hits = [i for i, ln in enumerate(code) if re.search(pattern, ln)]
    if len(hits) != 1:
        failures.append(
            "%s: expected exactly 1 occurrence, found %d%s"
            % (what, len(hits),
               "" if not hits else " at lines %s" % [h + 1 for h in hits]))
        return None
    return hits[0]


def brace_depth_between(code, lo, hi):
    """Running brace depth relative to `lo`, evaluated after each line."""
    depth = 0
    for i in range(lo, hi + 1):
        depth += code[i].count("{") - code[i].count("}")
    return depth


# ---------------------------------------------------------------------------
# S1 -- the define count, and the absent fourth symbol
# ---------------------------------------------------------------------------

def S1_three_child_macros_and_no_swmm_other(root, failures, report):
    path = os.path.join(root, "src", "constants.h")
    code = strip_comments(read(path))

    values = {}
    for ln in code:
        m = DEFINE_RE.match(ln)
        if m:
            values[m.group(1)] = m.group(2)

    for name in CHILD_MACROS:
        if name not in values:
            failures.append(
                "constants.h declares no string macro %s; the split needs all "
                "three of %s" % (name, ", ".join(CHILD_MACROS)))
    report.append("declared children : %s"
                  % {n: values.get(n) for n in CHILD_MACROS})

    # The load-bearing negative.  Searched over the WHOLE source tree, not just
    # constants.h: the property is that the symbol does not exist anywhere, so a
    # macro added in triton.h or output.h would violate it just as squarely.
    offenders = []
    for dirpath, _dirs, files in os.walk(os.path.join(root, "src")):
        for fn in files:
            if not fn.endswith((".h", ".cpp", ".hpp")):
                continue
            p = os.path.join(dirpath, fn)
            for i, ln in enumerate(strip_comments(read(p))):
                m = re.match(r'^\s*#\s*define\s+SWMM_OTHER\b', ln)
                if m:
                    offenders.append("%s:%d" % (os.path.relpath(p, root), i + 1))
    if offenders:
        failures.append(
            "SWMM_OTHER is declared as a macro at %s. It must NOT be: it is a "
            "DERIVED residual computed in write_times, and declaring it makes "
            "get_custom_time(SWMM_OTHER) compile -- which returns 0.0 silently "
            "because get_cat_index_ registers an absent category rather than "
            "failing. The absence of the symbol IS the guard."
            % ", ".join(offenders))
    report.append("SWMM_OTHER macro  : %s"
                  % ("ABSENT (required)" if not offenders else offenders))

    # Case-insensitive distinctness -- a real requirement, not hygiene, because
    # timecats_ is keyed with ci_less.
    seen = {}
    for name in list(EXISTING_TIMER_MACROS) + list(CHILD_MACROS):
        v = values.get(name)
        if v is None:
            continue
        key = v.lower()
        if key in seen:
            failures.append(
                "timer category values %s and %s collide case-insensitively "
                "(%r vs %r); super_timer::timecats_ uses a ci_less comparator, "
                "so they would share one accumulator"
                % (seen[key], name, values[seen[key]], v))
        seen[key] = name
    report.append("ci-distinct values: %d categories, %d distinct keys"
                  % (len(seen), len(set(seen))))


# ---------------------------------------------------------------------------
# S2 -- children strictly inside an unchanged parent
# ---------------------------------------------------------------------------

def _locate_brackets(root, failures):
    path = os.path.join(root, "src", "triton.h")
    code = strip_comments(read(path))
    idx = {}
    idx["parent_start"] = find_one(code, r'\bst\.start\(\s*SWMM_TIME\s*\)',
                                   "triton.h st.start(SWMM_TIME)", failures)
    idx["parent_stop"] = find_one(code, r'\bst\.stop\(\s*SWMM_TIME\s*\)',
                                  "triton.h st.stop(SWMM_TIME)", failures)
    return path, code, idx


def S2_children_inside_unchanged_parent(root, failures, report):
    path, code, idx = _locate_brackets(root, failures)
    ps, pe = idx["parent_start"], idx["parent_stop"]
    if ps is None or pe is None:
        return
    if not ps < pe:
        failures.append("the SWMM_TIME parent bracket is inverted")
        return
    report.append("parent bracket    : triton.h:%d -> :%d (must be ONE pair, "
                  "unchanged)" % (ps + 1, pe + 1))

    # The parent must remain a single pair.  R4's whole architectural claim is
    # that SWMM's emitted value is the same reading it was before the split, and
    # a second parent pair anywhere would silently change it.
    for kind in ("start", "stop"):
        n = sum(1 for ln in code
                if re.search(r'\bst\.%s\(\s*SWMM_TIME\s*\)' % kind, ln))
        if n != 1:
            failures.append("st.%s(SWMM_TIME) occurs %d times; the parent must "
                            "stay exactly one pair" % (kind, n))

    for child in CHILD_MACROS:
        starts = [i for i, ln in enumerate(code)
                  if re.search(r'\bst\.start\(\s*%s\s*\)' % child, ln)]
        stops = [i for i, ln in enumerate(code)
                 if re.search(r'\bst\.stop\(\s*%s\s*\)' % child, ln)]
        if not starts or not stops:
            failures.append("%s has no start/stop pair in triton.h" % child)
            continue
        if len(starts) != len(stops):
            failures.append("%s has %d start(s) and %d stop(s); every start "
                            "needs its stop or the accumulator runs away"
                            % (child, len(starts), len(stops)))
        outside = [i + 1 for i in starts + stops if not (ps < i < pe)]
        if outside:
            failures.append(
                "%s bracket line(s) %s lie OUTSIDE the SWMM_TIME parent "
                "(:%d..:%d). A child outside the parent still emits a plausible "
                "number and breaks only the closure."
                % (child, outside, ps + 1, pe + 1))
        report.append("%-10s pairs : %d, at lines %s"
                      % (child, len(starts),
                         sorted([i + 1 for i in starts + stops])))

    # SWMM_MPI legitimately takes TWO pairs because its span is disjoint: the
    # rank-0 solve sits between the gather and the scatter.  Pinned here so a
    # later "simplification" to one bracket -- which would swallow SWMM_STEP and
    # double-count it -- is a red test rather than a silent regression.
    n_mpi = sum(1 for ln in code if re.search(r'\bst\.start\(\s*SWMM_MPI\s*\)', ln))
    if n_mpi != 2:
        failures.append(
            "SWMM_MPI has %d start bracket(s); it must have exactly 2. Its span "
            "is DISJOINT -- MPI_Gatherv before the rank-0 solve, MPI_Scatterv "
            "after it -- and a single bracket spanning both would contain "
            "SWMM_STEP and double-count it." % n_mpi)


# ---------------------------------------------------------------------------
# S3 -- SWMM_STEP inside the rank-0 guard
# ---------------------------------------------------------------------------

def S3_swmm_step_inside_rank0_guard(root, failures, report):
    path, code, idx = _locate_brackets(root, failures)
    ps, pe = idx["parent_start"], idx["parent_stop"]
    if ps is None or pe is None:
        return

    guard = None
    for i in range(ps, pe + 1):
        if re.search(r'\bif\s*\(\s*rank\s*==\s*0\s*\)', code[i]):
            guard = i
            break
    if guard is None:
        failures.append("no `if (rank == 0)` guard found inside the SWMM_TIME "
                        "parent bracket")
        return

    start_i = find_one(code, r'\bst\.start\(\s*SWMM_STEP\s*\)',
                       "st.start(SWMM_STEP)", failures)
    stop_i = find_one(code, r'\bst\.stop\(\s*SWMM_STEP\s*\)',
                      "st.stop(SWMM_STEP)", failures)
    if start_i is None or stop_i is None:
        return

    if not guard < start_i < stop_i:
        failures.append(
            "SWMM_STEP's bracket is not inside the rank-0 guard: guard at "
            ":%d, start at :%d, stop at :%d"
            % (guard + 1, start_i + 1, stop_i + 1))
        return

    # Inside means nesting, not merely ordering: the bracket must sit at a brace
    # depth strictly greater than the guard's, and must not leave that block.
    depth_at_start = brace_depth_between(code, guard, start_i)
    depth_at_stop = brace_depth_between(code, guard, stop_i)
    if depth_at_start < 1 or depth_at_stop < 1:
        failures.append(
            "SWMM_STEP's bracket sits at brace depth %d/%d relative to the "
            "rank-0 guard; it must be strictly inside the guarded block. "
            "Placed OUTSIDE, every non-rank-0 rank reports a nonzero SWMM_STEP "
            "for a branch test it evaluated and a solve it never performed."
            % (depth_at_start, depth_at_stop))

    report.append("rank-0 guard      : triton.h:%d  %s"
                  % (guard + 1, code[guard].strip()[:60]))
    report.append("SWMM_STEP bracket : :%d -> :%d, depth %+d/%+d inside guard "
                  "(>=1 required)"
                  % (start_i + 1, stop_i + 1, depth_at_start, depth_at_stop))


# ---------------------------------------------------------------------------
# S4 -- write_times: the derivation, the untouched residual, the four columns
# ---------------------------------------------------------------------------

def S4_write_times_derivation_and_columns(root, failures, report):
    path = os.path.join(root, "src", "output.h")
    raw = read(path)
    code = strip_comments(raw)

    # The PRE-EXISTING simulation-level residual must be untouched.  It subtracts
    # the PARENT swmm_time; the three children are already inside it, so
    # subtracting them here too would double-count and stop Simulation closing.
    expected_other = ("T other_time = simulation_time - compute_time - mpi_time"
                      " - io_time - resize_time - swmm_time;")
    hits = [i for i, ln in enumerate(code)
            if " ".join(ln.split()) == expected_other]
    if len(hits) != 1:
        failures.append(
            "the pre-existing other_time residual is not present verbatim and "
            "exactly once (found %d). R4 explicitly does NOT edit it: it "
            "subtracts the PARENT swmm_time, which already contains the three "
            "children." % len(hits))
    else:
        report.append("other_time (:%d)  : UNCHANGED (required)" % (hits[0] + 1))

    # The derived residual, and the shape of the derivation.
    derived = [i for i, ln in enumerate(code)
               if re.search(r'\bswmm_other_time\s*=', ln)]
    if len(derived) != 1:
        failures.append("expected exactly one swmm_other_time derivation, "
                        "found %d" % len(derived))
    else:
        expr = " ".join(code[derived[0]].split())
        want = ("T swmm_other_time = swmm_time - swmm_xfer_time - "
                "swmm_mpi_time - swmm_step_time;")
        if expr != want:
            failures.append(
                "swmm_other_time is derived as %r; it must be exactly %r -- "
                "the subtraction from the PARENT is what makes the level close."
                % (expr, want))
        report.append("swmm_other (:%d)  : %s" % (derived[0] + 1, expr))

    # No get_custom_time for the residual, under any spelling.
    if re.search(r'get_custom_time\s*\(\s*(SWMM_OTHER|"swmm_other")',
                 "\n".join(code)):
        failures.append(
            "write_times reads the residual through get_custom_time. It must be "
            "DERIVED by subtraction: an unregistered category read returns 0.0 "
            "and the level would appear to close while carrying nothing.")

    # Three child reads.
    for child, local in (("SWMM_XFER", "swmm_xfer_time"),
                         ("SWMM_MPI", "swmm_mpi_time"),
                         ("SWMM_STEP", "swmm_step_time")):
        if not re.search(r'%s\s*=\s*st\.get_custom_time\(\s*%s\s*\)'
                         % (local, child), "\n".join(code)):
            failures.append("write_times does not read %s into %s"
                            % (child, local))

    # The emitted header, and the four new columns in it.  Anchored on
    # "%Rank, Compute" rather than "%Rank," because output.h emits a SECOND,
    # unrelated "%Rank,Rows,Cols" header for the domain-decomposition log.
    hdr = [i for i, ln in enumerate(code) if '"%Rank, Compute' in ln]
    if len(hdr) != 1:
        failures.append("expected exactly one performance-file header literal, "
                        "found %d" % len(hdr))
    else:
        header = re.search(r'"(%Rank,[^"]*)"', code[hdr[0]]).group(1)
        cols = [c.strip() for c in header.split(",")]
        for want in ("SWMM", "SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER"):
            if want not in cols:
                failures.append("header literal lacks column %r (has %s)"
                                % (want, cols))
        # Parent immediately followed by its decomposition, so a reader meets
        # the whole before the parts.
        if "SWMM" in cols:
            k = cols.index("SWMM")
            if cols[k + 1:k + 5] != ["SWMM_XFER", "SWMM_MPI", "SWMM_STEP",
                                     "SWMM_OTHER"]:
                failures.append(
                    "the four child columns must immediately follow SWMM in the "
                    "header; got %s" % cols[k + 1:k + 5])
        report.append("header columns    : %d -> %s" % (len(cols), header))

    # Per-rank row and Average row must each carry all four.
    for local in ("swmm_xfer_time_all", "swmm_mpi_time_all",
                  "swmm_step_time_all", "swmm_other_time_all"):
        if not re.search(r'new T\[size_\]', "\n".join(
                [ln for ln in code if local in ln])):
            failures.append("%s is not allocated as new T[size_]" % local)
        if not re.search(r'MPI_Gather\(\s*&%s\s*,' % local[:-4], "\n".join(code)):
            failures.append("no MPI_Gather for %s" % local[:-4])
        if not re.search(r'\b%s\[j\]' % local, "\n".join(code)):
            failures.append("%s does not appear in the per-rank row writer"
                            % local)
        if not re.search(r'average\(\s*%s\s*,\s*size_\s*\)' % local,
                         "\n".join(code)):
            failures.append(
                "%s has no average(...) term. The Average row's arity is FIXED: "
                "the downstream parser's Average-presence detector depends on "
                "it, so a suppressed column is not an option even though the "
                "Average of SWMM_STEP is rank0/N and is outside the closure "
                "claim." % local)
    report.append("row writers       : 4 arrays, 4 gathers, 4 row terms, "
                  "4 average terms")


# ---------------------------------------------------------------------------
# S5 -- no bare string literal reaches the timer API
# ---------------------------------------------------------------------------

def S5_no_bare_timer_string_literals(root, failures, report):
    """The residual hole S1 cannot close.

    ``super_timer::start``/``stop``/``get_custom_time`` take ``std::string``, so
    removing the ``SWMM_OTHER`` macro does not stop anyone writing
    ``get_custom_time("swmm_other")``.  That call compiles, registers a new empty
    category through ``get_cat_index_`` -- which is a WRITE, not a read -- and
    returns 0.0.  The closure check would then pass while the residual carried
    nothing.  S1 removes the symbol; this removes the literal.
    """
    offenders = []
    for rel in ("src/triton.h", "src/output.h", "src/supertimer.h"):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        for i, ln in enumerate(strip_comments(read(p))):
            # supertimer.h's own definitions name the parameter, not a literal;
            # only CALL sites with a literal argument are offenders.
            if BARE_LITERAL_CALL_RE.search(ln):
                offenders.append("%s:%d  %s" % (rel, i + 1, ln.strip()[:70]))
    if offenders:
        failures.append(
            "timer API called with a bare string literal at:\n    %s\n"
            "Every category must be named by a macro. A literal bypasses the "
            "define-count guard entirely: get_cat_index_ REGISTERS the unknown "
            "category and returns 0.0 rather than failing."
            % "\n    ".join(offenders))
    report.append("bare literal calls: %s"
                  % ("none (required)" if not offenders else offenders))


# ---------------------------------------------------------------------------
# S6 -- the child spans are pairwise DISJOINT
# ---------------------------------------------------------------------------

def S6_child_spans_are_pairwise_disjoint(root, failures, report):
    """No two child timers may be open at the same line.

    S2 tests each child for CONTAINMENT in the parent and never tests the
    children against EACH OTHER, so two children that OVERLAP pass it: both
    brackets are inside the parent, and SWMM_MPI keeps its required two pairs.
    The region between them is then inside BOTH children and counted twice.

    That defect is invisible to every runtime instrument that reads the emitted
    sum, because SWMM_OTHER is DERIVED by subtraction: an overlap leaves
    XFER + MPI + STEP + OTHER == SWMM exact and drives the residual NEGATIVE.
    check_performance_identity.py now rejects the negative residual, but only
    once a run has happened; this rejects the source that would produce it.

    The guard-forms M3 mutation covers ONE instance of this -- SWMM_MPI
    collapsing to a single pair, which swallows SWMM_STEP. The general predicate
    is pairwise disjointness, and M9 exercises it directly.
    """
    path, code, idx = _locate_brackets(root, failures)
    ps, pe = idx["parent_start"], idx["parent_stop"]
    if ps is None or pe is None:
        return

    # (line, kind, child) events for every child bracket inside the parent.
    events = []
    for child in CHILD_MACROS:
        for i in range(ps, pe + 1):
            if re.search(r'\bst\.start\(\s*%s\s*\)' % child, code[i]):
                events.append((i, "start", child))
            if re.search(r'\bst\.stop\(\s*%s\s*\)' % child, code[i]):
                events.append((i, "stop", child))
    events.sort()

    open_now = []
    spans = []
    span_start = {}
    for line, kind, child in events:
        if kind == "start":
            if child in open_now:
                failures.append(
                    "%s is started again at :%d while already open — a "
                    "re-entrant start loses the first interval, because "
                    "super_timer::start overwrites the stored timeval."
                    % (child, line + 1))
                continue
            if open_now:
                failures.append(
                    "%s opens at :%d while %s is still open (opened at :%d). "
                    "Two child timers may not be open at once: the overlapping "
                    "region is counted in BOTH and the parent is over-"
                    "subscribed. This is INVISIBLE to the emitted sum — "
                    "SWMM_OTHER is derived by subtraction, so the identity "
                    "stays exact and the residual goes NEGATIVE instead."
                    % (child, line + 1, open_now[-1],
                       span_start[open_now[-1]] + 1))
            open_now.append(child)
            span_start[child] = line
        else:
            if child not in open_now:
                failures.append(
                    "%s is stopped at :%d without an open start — the "
                    "accumulated delta is measured from a stale timeval."
                    % (child, line + 1))
                continue
            open_now.remove(child)
            spans.append((span_start.pop(child), line, child))

    for child in open_now:
        failures.append("%s is still open at the parent's stop (:%d) — its "
                        "bracket never closes inside the parent."
                        % (child, pe + 1))

    report.append("child spans       : %s"
                  % sorted([(a + 1, b + 1, c) for a, b, c in spans]))
    report.append("max concurrently open: %d (1 required)"
                  % (1 if spans else 0))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

TESTS = (
    ("S1_three_child_macros_and_no_swmm_other", S1_three_child_macros_and_no_swmm_other),
    ("S2_children_inside_unchanged_parent", S2_children_inside_unchanged_parent),
    ("S3_swmm_step_inside_rank0_guard", S3_swmm_step_inside_rank0_guard),
    ("S4_write_times_derivation_and_columns", S4_write_times_derivation_and_columns),
    ("S5_no_bare_timer_string_literals", S5_no_bare_timer_string_literals),
    ("S6_child_spans_are_pairwise_disjoint", S6_child_spans_are_pairwise_disjoint),
)


def main(argv):
    want = None
    root = None
    args = argv[1:]
    if len(args) == 2:
        want, root = args
    elif len(args) == 1:
        root = args[0]
    else:
        print("usage: %s [subtest] <repo-root>" % os.path.basename(argv[0]))
        return 2

    if not os.path.isdir(os.path.join(root, "src")):
        print("ERROR: %r does not look like a TRITON source root" % root)
        return 2

    ran = 0
    rc = 0
    for name, fn in TESTS:
        if want and name != want:
            continue
        failures = []
        report = []
        print("RUN  %s" % name)
        fn(root, failures, report)
        for line in report:
            print("    %s" % line)
        if failures:
            for f in failures:
                print("FAIL: %s" % f)
            rc = 1
        else:
            print("PASS %s" % name)
        print("")
        ran += 1

    if want and ran == 0:
        print("ERROR: no subtest named %r" % want)
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Reviewer test (WP-0B): the fence guard must reject EVERY conditional form.

WHAT THIS FALSIFIES
-------------------
``test/test_compute_timer_fence.py`` CHECK 2 states its own contract as:

    "that call sits at the SAME brace depth as the stop -- i.e. it is not
     wrapped in a branch of its own."
    "Check 2 FAILS if a future edit guards the fence on ``gpu_direct_flag``,
     on a rank test, or on any other condition"

It implements that contract as a NET BRACE DELTA between the fence line and
the stop line.  Net brace delta measures UNBALANCED NESTING, not
conditionality, and the two come apart on any conditional written on a
single line:

    if (cond) { gpuStreamSynchronize(streams); }     ->  delta 0  (balanced)
    if (cond)   gpuStreamSynchronize(streams);       ->  delta 0  (no braces)

Both are conditional fences.  Both leave the mis-attribution standing on
every path where ``cond`` is false -- the exact hazard CHECK 2 names.  Both
are accepted by the guard.

Only the MULTI-LINE form, where the opening brace is on the ``if`` line and
the closing brace falls after the fence, produces a non-zero delta, and that
is the single form the landed package exercised.

THIS TEST DOES NOT CLAIM THE LANDED FENCE IS WRONG.  The fence at
``triton.h`` is genuinely unconditional and the runtime behaviour is correct.
The defect is in the REGRESSION GUARD that ships with it: it does not hold
the line its docstring says it holds, so a later edit can reintroduce a
conditional fence and the suite stays green.

METHOD
------
Synthesize, from the LIVE source, one mutant per conditional form by
replacing the fence statement in place.  Run the landed guard on each.  A
conditional fence MUST be rejected (exit 1).  Any form the guard accepts is
reported as a hole.

A control arm runs the guard on the unmodified source and requires exit 0,
so a guard that rejects everything cannot pass this test vacuously.

USAGE
-----
    python3 test/test_compute_timer_fence_conditional_forms.py [path-to-triton.h]

Exit 0 = the guard rejects every conditional form.  Exit 1 = at least one
conditional form is accepted.  No build, no GPU, no simulation.
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = os.path.join(HERE, "test_compute_timer_fence.py")

FENCE = re.compile(r"^(\s*)gpuStreamSynchronize\s*\(\s*streams\s*\)\s*;\s*$")
STOP_COMPUTE = re.compile(r"\bst\.stop\s*\(\s*COMPUTE_TIME\s*\)")
START_SWMM = re.compile(r"\bst\.start\s*\(\s*SWMM_TIME\s*\)")

# Each form is (label, callable(indent) -> list-of-replacement-lines).
# Every one of these is a CONDITIONAL fence and every one must be rejected.
FORMS = [
    ("multiline_braced", lambda ind: [
        "%sif (arglist.gpu_direct_flag) {" % ind,
        "%s  gpuStreamSynchronize(streams);" % ind,
        "%s}" % ind,
    ]),
    ("oneline_braced", lambda ind: [
        "%sif (arglist.gpu_direct_flag) { gpuStreamSynchronize(streams); }" % ind,
    ]),
    ("oneline_bare", lambda ind: [
        "%sif (arglist.gpu_direct_flag) gpuStreamSynchronize(streams);" % ind,
    ]),
    ("oneline_rank_test", lambda ind: [
        "%sif (rank == 0) gpuStreamSynchronize(streams);" % ind,
    ]),
    ("ternary_guard", lambda ind: [
        "%sarglist.gpu_direct_flag ? gpuStreamSynchronize(streams) : (void)0;" % ind,
    ]),
]


def locate_coupled_fence(lines):
    """Index of the fence statement that immediately precedes the coupled
    st.stop(COMPUTE_TIME) -- the one whose next code line starts SWMM_TIME.

    Located by structure, not by line number, so this survives edits above it.
    """
    for i, line in enumerate(lines):
        if not STOP_COMPUTE.search(line):
            continue
        nxt = None
        for j in range(i + 1, min(i + 8, len(lines))):
            if lines[j].strip():
                nxt = j
                break
        if nxt is None or not START_SWMM.search(lines[nxt]):
            continue
        for k in range(i - 1, max(i - 12, -1), -1):
            m = FENCE.match(lines[k])
            if m:
                return k, m.group(1)
            if lines[k].strip() and not lines[k].lstrip().startswith("//"):
                break
    return None, None


def run_guard(path):
    """Exit status of the landed guard against `path`, read from the actuator."""
    proc = subprocess.run([sys.executable, GUARD, path],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "src", "triton.h")
    src = os.path.normpath(src)

    if not os.path.exists(GUARD):
        print("FAIL: guard under test not found at %s" % GUARD)
        return 1

    with open(src, "r", errors="replace") as fh:
        lines = fh.read().split("\n")

    idx, indent = locate_coupled_fence(lines)
    if idx is None:
        print("FAIL: no unconditional fence statement found immediately above the coupled")
        print("      st.stop(COMPUTE_TIME) in %s." % src)
        print("      Either the fence is absent (run the guard itself) or the block's shape")
        print("      changed and this test must be re-grounded.")
        return 1

    print("source under test : %s" % src)
    print("fence located at  : line %d  %s" % (idx + 1, lines[idx].strip()))
    print("")

    # CONTROL -- the guard must ACCEPT the unmodified, genuinely unconditional source.
    rc, _ = run_guard(src)
    print("control (unmodified source)          : guard exit %d  %s"
          % (rc, "OK" if rc == 0 else "UNEXPECTED"))
    if rc != 0:
        print("")
        print("FAIL: the guard rejects the unmodified source, so this test cannot")
        print("      distinguish a real hole from a guard that rejects everything.")
        return 1
    print("")

    holes = []
    tmpdir = tempfile.mkdtemp(prefix="wp0b_condforms_")
    for label, build in FORMS:
        mutant = lines[:idx] + build(indent) + lines[idx + 1:]
        mpath = os.path.join(tmpdir, "%s.h" % label)
        with open(mpath, "w") as fh:
            fh.write("\n".join(mutant))
        rc, out = run_guard(mpath)
        verdict = "REJECTED (correct)" if rc == 1 else "ACCEPTED -- HOLE"
        print("conditional form %-20s : guard exit %d  %s" % (label, rc, verdict))
        if rc != 1:
            holes.append((label, build(indent), out))

    print("")
    if holes:
        print("FAIL: the fence guard accepts %d of %d conditional forms."
              % (len(holes), len(FORMS)))
        print("")
        for label, body, out in holes:
            print("  form %s -- accepted by the guard:" % label)
            for b in body:
                print("      %s" % b.strip())
            for ln in out.strip().split("\n"):
                if "brace delta" in ln or ln.startswith("PASS"):
                    print("      guard said: %s" % ln.strip())
            print("")
        print("  CHECK 2 keys on NET BRACE DELTA, which is zero for a balanced one-line")
        print("  `if (c) { ... }` and zero for a braceless `if (c) stmt;`. Net brace delta")
        print("  detects unbalanced NESTING, not CONDITIONALITY, so it holds only against")
        print("  the multi-line form. A guard that accepts a conditional fence cannot")
        print("  prevent the regression its own docstring names.")
        return 1

    print("PASS: every conditional form is rejected by the fence guard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Reviewer test (WP-0B chunk 2): three conditional forms the strengthened
guard still accepts, and which its docstring claims it catches.

WHAT THIS FALSIFIES
-------------------
``test/test_compute_timer_fence.py`` at c1ae36a is a genuine strengthening: a
15-form reviewer battery moves it from 4 rejections to 12.  It closes every
form the accompanying acceptance test enumerates, plus the two-line braceless
form that test does not.

What it does NOT close is its own CHECK 2b contract, stated in the docstring
as::

    CHECK 2b CONTROLLING HEADER.  The nearest preceding code-bearing line is
             not a dangling control header (``if (...)``, ``else``, ``for``,
             ``while``, ``switch``, ``do``, a ``case``/``default`` label) and
             is not a preprocessor conditional (``#if``/``#ifdef``/...).
             ... Catches the two-line braceless form and a compile-time-
             conditional fence.

2b is implemented as two regexes matched against the SINGLE nearest preceding
code-bearing line.  A control header is not always one line, ``if (...)`` is
not the only spelling of ``if``, and a preprocessor conditional does not have
to be adjacent.  Each of the three forms below is a genuinely conditional
fence that leaves the mis-attribution standing on every path its condition
excludes -- the exact hazard CHECK 2 names -- and each is ACCEPTED.

    U1  multi-line condition, braceless body
            if (arglist.gpu_direct_flag &&
                swmm_model.num_of_swmm_links > 0)
              gpuStreamSynchronize(streams);
        The preceding code line is the condition's CONTINUATION, which does
        not start with `if`, so DANGLING_CTRL does not match it.

    U2  `if constexpr`, braceless body
            if constexpr (sizeof(T) == 8)
              gpuStreamSynchronize(streams);
        DANGLING_CTRL requires `if` followed by optional whitespace and `(`.
        `constexpr` is not whitespace.  This form is not exotic here:
        triton.h is `template <typename T>` throughout, so a backend- or
        precision-conditional fence is idiomatically written this way.

    U3  preprocessor conditional with an intervening statement
            #ifdef TRITON_CUDA
              const int drain_hint = 1; (void)drain_hint;
              gpuStreamSynchronize(streams);
            #endif
        PREPROC_COND is matched against the NEAREST code line only, so one
        statement between the `#ifdef` and the fence hides it.  The bare
        adjacent form IS caught; this one is not, and the docstring's
        "a compile-time-conditional fence" reads as covering both.

THESE ARE NOT REGRESSIONS.  Measured against the pre-chunk guard at 87460ab,
all three were accepted there too.  They are residual holes in the SAME class
the chunk was dispatched to close, which is what distinguishes them from the
five blind spots the docstring does disclose (macro expansion, an
interprocedural fence, `goto`, string-literal lexing, two statements on the
fence line) -- those are other classes, honestly named.  The gap this test
reports is between what CHECK 2b claims and what CHECK 2b does.

WHAT THIS TEST DOES NOT CLAIM
-----------------------------
The fence in triton.h is unconditional and correct; the runtime behaviour is
right.  The guard is materially better than what it replaced.  This test
holds the narrower line that a regression guard should reject every member of
the class it names, and reports the three members it does not.

METHOD
------
Synthesize, from the LIVE source, one mutant per form by replacing the fence
statement in place.  Run the landed guard on each; a conditional fence MUST
be rejected (exit 1).  A control arm runs the guard on the unmodified source
and requires exit 0, so a guard that rejects everything cannot pass this test
vacuously.

USAGE
-----
    python3 test/test_compute_timer_fence_undetected_forms.py [path-to-triton.h]

Exit 0 = the guard rejects all three forms.  Exit 1 = at least one is
accepted.  No build, no GPU, no simulation.
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
# Every one is a CONDITIONAL fence and every one must be rejected.
FORMS = [
    ("multiline_condition", lambda ind: [
        "%sif (arglist.gpu_direct_flag &&" % ind,
        "%s    swmm_model.num_of_swmm_links > 0)" % ind,
        "%s  gpuStreamSynchronize(streams);" % ind,
    ]),
    ("if_constexpr", lambda ind: [
        "%sif constexpr (sizeof(T) == 8)" % ind,
        "%s  gpuStreamSynchronize(streams);" % ind,
    ]),
    ("ifdef_nonadjacent", lambda ind: [
        "#ifdef TRITON_CUDA",
        "%sconst int drain_hint = 1; (void)drain_hint;" % ind,
        "%sgpuStreamSynchronize(streams);" % ind,
        "#endif",
    ]),
]


def locate_coupled_fence(lines):
    """Index and indent of the fence statement immediately preceding the
    coupled st.stop(COMPUTE_TIME) -- the stop whose next code line starts
    SWMM_TIME.  Located by structure, not by line number."""
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
    tmpdir = tempfile.mkdtemp(prefix="wp0b_undetected_")
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
        print("FAIL: the fence guard accepts %d of %d conditional forms that CHECK 2b's"
              % (len(holes), len(FORMS)))
        print("      stated contract covers.")
        print("")
        for label, body, out in holes:
            print("  form %s -- accepted by the guard:" % label)
            for b in body:
                print("      %s" % b)
            for ln in out.strip().split("\n"):
                if ln.startswith("line above") or ln.startswith("PASS"):
                    print("      guard said: %s" % ln.strip())
            print("")
        print("  CHECK 2b matches two regexes against the SINGLE nearest preceding")
        print("  code-bearing line.  A control header spanning two lines, the spelling")
        print("  `if constexpr (...)`, and a `#ifdef` separated from the fence by one")
        print("  statement each defeat that shape while remaining a conditional fence.")
        print("  Fixing any of the three is a change to the guard, which is the coder's")
        print("  to make and not this reviewer's.")
        return 1

    print("PASS: all three undetected conditional forms are rejected by the fence guard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

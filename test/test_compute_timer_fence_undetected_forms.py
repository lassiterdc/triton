#!/usr/bin/env python3
"""Characterization tripwire (WP-0B chunk 2): the three conditional forms the
fence guard is DOCUMENTED not to catch, pinned in executable form.

AUTHORSHIP.  The three forms below, and their analysis, were constructed by the
WP-0B reviewer and committed at 16ab5d1 as that review's failing-test finding.
They are preserved here verbatim.  What changed at the review close is this
test's POLARITY, not its content: the finding was dispositioned on the
DISCLOSURE arm rather than the regex arm, so the assertion was inverted from
"these should be caught" to "these are known-uncaught", and the test was
registered in CTest.  The reviewer's closing line -- that fixing any of the
three is a change to the guard, which is the coder's to make -- still stands,
and is now the thing this test EXISTS to detect.

WHAT THIS PINS
--------------
``test/test_compute_timer_fence.py`` answers "is this fence conditional?" with
CHECK 2b, two regexes matched against the SINGLE nearest preceding code-bearing
line.  A control header is not always one line, ``if (...)`` is not the only
spelling of ``if``, and a preprocessor conditional does not have to be
adjacent.  Each form below is a genuinely conditional fence that CHECK 2b
accepts:

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
        adjacent form IS caught; this one is not.

THESE ARE NOT REGRESSIONS.  Measured against the pre-chunk guard at 87460ab,
all three were accepted there too.  They are disclosed by name, with code
samples, in the guard's own WHAT THIS INSTRUMENT IS BLIND TO list, because the
accepted disposition was to state the limit honestly rather than to run a regex
race against unbounded C++ spellings.

WHY THIS TEST EXISTS -- IT IS ABOUT LATER, NOT ABOUT NOW
--------------------------------------------------------
Today it passes, and passing is the point: it makes the documented gap
EXECUTABLE rather than merely written down.

Its value arrives the day someone tightens a CHECK 2b regex.  If a future edit
closes any of these three forms, this test FAILS LOUDLY and names the form and
the file to update.  That converts a one-time review finding into a PERMANENT
BIDIRECTIONAL LOCK between the guard's contract sentence and the guard's
behaviour:

  * the guard's blind-spot list says these three are not caught;
  * this test says the same thing in a form that runs;
  * so the two can no longer drift apart silently in EITHER direction.

The failure this prevents is the one that produced the original WP-0B finding:
a docstring that names a class the implementation only samples.  A red here is
not a defect -- it is the news that the guard got better and the prose did not
keep up.

TWO CONTROL ARMS, AND BOTH ARE LOAD-BEARING UNDER THE INVERTED POLARITY
-----------------------------------------------------------------------
Inverting the assertion inverts the vacuous-pass hazard, so the original
control is no longer sufficient on its own.

  POSITIVE control -- the guard must ACCEPT the unmodified source (exit 0).
  NEGATIVE control -- the guard must still REJECT a form it is documented to
  catch (`if (rank == 0) gpuStreamSynchronize(streams);`, the adjacent
  single-line rank test).

Without the negative arm, a guard gutted to accept EVERYTHING would satisfy
"all three forms are accepted" and this test would report green on a guard that
had stopped guarding.  The negative arm is what makes a green here mean "the
guard still works AND these three remain its known gaps" rather than merely
"nothing was rejected".

METHOD
------
Synthesize, from the LIVE source, one mutant per form by replacing the fence
statement in place.  Run the landed guard on each.

USAGE
-----
    python3 test/test_compute_timer_fence_undetected_forms.py [path-to-triton.h]

Exit 0 = the guard's behaviour matches its documented blind spots.
Exit 1 = it diverged; the guard and its blind-spot list disagree.
No build, no GPU, no simulation.
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
# Constructed by the WP-0B reviewer at 16ab5d1 and PRESERVED VERBATIM.
# Every one is a CONDITIONAL fence that the guard is DOCUMENTED not to catch.
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


# NEGATIVE control: a conditional form the guard IS documented to catch.  If this
# stops being rejected, the guard has been weakened wholesale and a green result
# below would be vacuous.
CAUGHT_CONTROL = ("oneline_rank_test", lambda ind: [
    "%sif (rank == 0) gpuStreamSynchronize(streams);" % ind,
])


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

    def mutate(build, label):
        mutant = lines[:idx] + build(indent) + lines[idx + 1:]
        mpath = os.path.join(tmpdir, "%s.h" % label)
        with open(mpath, "w") as fh:
            fh.write("\n".join(mutant))
        return run_guard(mpath)

    tmpdir = tempfile.mkdtemp(prefix="wp0b_undetected_")

    # POSITIVE control -- the guard must ACCEPT the unmodified source.
    rc, _ = run_guard(src)
    print("control + (unmodified source)        : guard exit %d  %s"
          % (rc, "OK" if rc == 0 else "UNEXPECTED"))
    if rc != 0:
        print("")
        print("FAIL: the guard rejects the unmodified source, so nothing below is")
        print("      interpretable. Run the guard itself and re-ground it.")
        return 1

    # NEGATIVE control -- the guard must still REJECT a form it is documented to
    # catch. Without this, a guard gutted to accept everything would satisfy the
    # three assertions below and this test would report green on a dead guard.
    nlabel, nbuild = CAUGHT_CONTROL
    rc, _ = mutate(nbuild, nlabel)
    print("control - (%-24s) : guard exit %d  %s"
          % (nlabel, rc, "OK" if rc == 1 else "UNEXPECTED"))
    if rc != 1:
        print("")
        print("FAIL: the guard ACCEPTED `%s`, a form it is documented to catch." % nlabel)
        print("      The guard has been weakened wholesale, so a pass on the three")
        print("      documented gaps below would be vacuous. Fix the guard first:")
        print("      test/test_compute_timer_fence.py, CHECK 2a/2b/2c.")
        return 1
    print("")

    # The pinned gaps. Each MUST still be accepted; a rejection means the guard
    # improved and the blind-spot list did not keep up.
    closed = []
    for label, build in FORMS:
        rc, out = mutate(build, label)
        verdict = "accepted (documented gap)" if rc == 0 else "REJECTED -- NOW CAUGHT"
        print("documented gap %-22s : guard exit %d  %s" % (label, rc, verdict))
        if rc != 0:
            closed.append((label, build(indent)))

    print("")
    if closed:
        print("FAIL: %d of %d documented gaps are NO LONGER gaps -- the guard now catches"
              % (len(closed), len(FORMS)))
        print("      them, and its blind-spot list still says it does not.")
        print("")
        for label, body in closed:
            print("  form %s -- now REJECTED by the guard:" % label)
            for b in body:
                print("      %s" % b)
            print("")
        print("  THIS IS GOOD NEWS WITH AN OBLIGATION, NOT A DEFECT. Someone tightened")
        print("  CHECK 2b. Update BOTH of these so contract and behaviour agree again:")
        print("")
        print("    1. test/test_compute_timer_fence.py -- remove the now-closed form(s)")
        print("       from the WHAT THIS INSTRUMENT IS BLIND TO list, and re-read the")
        print("       CHECK 2b paragraph, which states the scope 2b actually samples.")
        print("    2. this file -- drop the closed form(s) from FORMS.")
        print("")
        print("  If FORMS becomes empty, CHECK 2b no longer samples its class and the")
        print("  scope caveat in the guard's docstring should go with it.")
        return 1

    print("PASS: all %d documented gaps are still gaps, and the guard still rejects"
          % len(FORMS))
    print("      the form it is documented to catch. Contract and behaviour agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

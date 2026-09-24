#!/usr/bin/env python3
"""Structural regression test for the COMPUTE_TIME fence in the coupled block.

WHAT THIS FALSIFIES
-------------------
TRITON's hydrodynamic kernels are dispatched through ``triton::parallel_for``,
which is asynchronous on every device backend.  In the coupled block the
``COMPUTE_TIME`` timer is stopped and ``SWMM_TIME`` is started immediately
afterwards; the first statement inside ``SWMM_TIME`` is a synchronous
``gpuMemcpyAsync`` (it reaches ``Kokkos::deep_copy``).  If no fence stands
between the last kernel launch and ``st.stop(COMPUTE_TIME)``, that deep_copy
drains the whole timestep's GPU work and the drain is charged to the SWMM
column -- ``Compute`` under-reports and ``SWMM`` over-reports by roughly one
timestep per step.

This test asserts the SOURCE-STRUCTURAL property that makes the runtime
mis-attribution impossible:

  1. a synchronizing call appears between the LAST kernel launch of the
     COMPUTE_TIME span and the coupled ``st.stop(COMPUTE_TIME)``; and
  2. that call sits at the SAME brace depth as the stop -- i.e. it is not
     wrapped in a branch of its own.

Check 1 FAILS on the pre-fix source.  Check 2 FAILS if a future edit guards
the fence on ``gpu_direct_flag``, on a rank test, or on any other condition
-- the specific hazard of copying the conditional sibling block at
``triton.h:2478``, whose fence IS inside ``if (arglist.gpu_direct_flag)``.

DEMONSTRATING THE FAILING FORM
------------------------------
Run it against the unfenced source to see it red::

    git show a38338b0:src/triton.h > /tmp/pre.h
    python3 test/test_compute_timer_fence.py /tmp/pre.h     # exits 1

USAGE
-----
    python3 test/test_compute_timer_fence.py [path-to-triton.h]

Exit 0 = pass, 1 = fail.  No build, no GPU and no simulation required, so it
runs on any host at review time.

SCOPE NOTE: this test asserts the STRUCTURE that prevents the mis-attribution.
It does NOT measure the re-attribution itself; that is a runtime property
requiring two authorized GPU coupled runs, and it is deliberately out of
this test's scope.
"""

import os
import re
import sys

KERNEL_LAUNCH = re.compile(r"\b(?:Kernels::\w+\s*\(|triton::parallel_for\s*\()")
SYNC_CALL = re.compile(r"\b(?:gpuStreamSynchronize\s*\(|Kokkos::fence\s*\(|gpuMemcpyAsync\s*\()")
START_COMPUTE = re.compile(r"\bst\.start\s*\(\s*COMPUTE_TIME\s*\)")
STOP_COMPUTE = re.compile(r"\bst\.stop\s*\(\s*COMPUTE_TIME\s*\)")
START_SWMM = re.compile(r"\bst\.start\s*\(\s*SWMM_TIME\s*\)")


def strip_comment(line):
    """Drop a trailing // comment.  Adequate here: the COMPUTE_TIME span in
    triton.h contains no string literal carrying a // or a brace."""
    idx = line.find("//")
    return line if idx < 0 else line[:idx]


def code_lines(lines):
    return [strip_comment(l) for l in lines]


def find_coupled_stop(code):
    """The coupled stop is the st.stop(COMPUTE_TIME) whose next code-bearing
    line starts SWMM_TIME.  Keyed on that pairing rather than on a line number
    so the test survives edits above it."""
    hits = []
    for i, line in enumerate(code):
        if not STOP_COMPUTE.search(line):
            continue
        for j in range(i + 1, min(i + 8, len(code))):
            if code[j].strip():
                if START_SWMM.search(code[j]):
                    hits.append(i)
                break
    return hits


def depth_between(code, lo, hi):
    """Net brace delta over code[lo:hi]."""
    seg = "".join(code[lo:hi])
    return seg.count("{") - seg.count("}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "triton.h")
    path = os.path.normpath(path)

    with open(path, "r", errors="replace") as fh:
        lines = fh.read().split("\n")
    code = code_lines(lines)

    failures = []

    stops = find_coupled_stop(code)
    if len(stops) != 1:
        print("FAIL: expected exactly one st.stop(COMPUTE_TIME) immediately followed by "
              "st.start(SWMM_TIME); found %d." % len(stops))
        print("      The coupled block's shape has changed; this test must be re-grounded.")
        return 1
    stop_i = stops[0]

    # Walk back to the st.start(COMPUTE_TIME) that opens this span.
    start_i = None
    for i in range(stop_i - 1, -1, -1):
        if START_COMPUTE.search(code[i]):
            start_i = i
            break
    if start_i is None:
        print("FAIL: no st.start(COMPUTE_TIME) found above the coupled stop at line %d."
              % (stop_i + 1))
        return 1

    # Last kernel launch inside the span.
    last_launch = None
    for i in range(start_i + 1, stop_i):
        if KERNEL_LAUNCH.search(code[i]):
            last_launch = i
    if last_launch is None:
        print("FAIL: no kernel launch found between st.start(COMPUTE_TIME) (line %d) and the "
              "coupled st.stop(COMPUTE_TIME) (line %d)." % (start_i + 1, stop_i + 1))
        print("      Either the span moved or the launches did; re-ground this test.")
        return 1

    # CHECK 1 -- a synchronizing call after the last launch, before the stop.
    sync_i = None
    for i in range(last_launch + 1, stop_i):
        if SYNC_CALL.search(code[i]):
            sync_i = i
            break

    if sync_i is None:
        failures.append(
            "CHECK 1 FAILED: no synchronizing call between the last kernel launch "
            "(line %d: %s) and st.stop(COMPUTE_TIME) (line %d).\n"
            "    Device-backend kernels are still in flight when the timer stops, so the "
            "first deep_copy inside SWMM_TIME drains them and the drain is charged to the "
            "SWMM column." % (last_launch + 1, code[last_launch].strip()[:70], stop_i + 1))
    else:
        # CHECK 2 -- same brace depth as the stop, i.e. no branch of its own.
        delta = depth_between(code, sync_i, stop_i)
        if delta != 0:
            failures.append(
                "CHECK 2 FAILED: the synchronizing call at line %d sits at a different brace "
                "depth than st.stop(COMPUTE_TIME) at line %d (net brace delta %+d).\n"
                "    The fence must be unconditional. A fence inside a branch leaves the "
                "mis-attribution standing on every path that branch excludes -- the hazard of "
                "copying the conditional sibling at triton.h:2478, whose fence is inside "
                "if (arglist.gpu_direct_flag)." % (sync_i + 1, stop_i + 1, delta))

    print("file            : %s" % path)
    print("COMPUTE_TIME    : start line %d -> coupled stop line %d" % (start_i + 1, stop_i + 1))
    print("last launch     : line %d  %s" % (last_launch + 1, code[last_launch].strip()[:70]))
    if sync_i is not None:
        print("fence           : line %d  %s" % (sync_i + 1, code[sync_i].strip()[:70]))
        print("brace delta fence->stop : %+d (0 required)" % depth_between(code, sync_i, stop_i))
    else:
        print("fence           : ABSENT")
    print("")

    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1

    print("PASS: the COMPUTE_TIME span is fenced before its coupled stop, unconditionally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

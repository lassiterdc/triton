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

THE INVARIANT THIS GUARD HOLDS
------------------------------
    Every control path that reaches the coupled ``st.stop(COMPUTE_TIME)``
    also executes the fence.

Note what that does NOT say.  It does not say the fence is at file scope: the
fence legitimately sits inside ``#ifdef TRITON_SWMM`` and inside
``if (swmm_model.num_of_swmm_links > 0)``.  Those conditionals contain the
STOP as well, so every path reaching the stop passes the fence.  What the
invariant forbids is a conditional that contains the FENCE but not the STOP.

HOW IT IS CHECKED -- and why the previous instrument was not enough
-------------------------------------------------------------------
This guard originally implemented the invariant as a NET BRACE DELTA between
the fence line and the stop line.  Net brace delta measures unbalanced
NESTING, which is a necessary but not a sufficient condition: it is blind to
every construct that controls a statement without changing brace balance.
Measured against the landed guard, four conditional fences were ACCEPTED --

    if (c) { gpuStreamSynchronize(streams); }        delta 0, balanced
    if (c)   gpuStreamSynchronize(streams);          delta 0, no braces
    if (rank == 0) gpuStreamSynchronize(streams);    delta 0, no braces
    c ? gpuStreamSynchronize(streams) : (void)0;     delta 0, no braces

-- plus a fifth the accompanying reviewer test does not enumerate, the
two-line braceless form::

    if (c)
      gpuStreamSynchronize(streams);

Brace delta is therefore RETAINED as one conjunct (it is the only one of the
three that catches a scope CLOSING between fence and stop) and is no longer
the primary instrument.  The three conjuncts are:

  CHECK 1  a synchronizing call appears between the LAST kernel launch of the
           COMPUTE_TIME span and the coupled ``st.stop(COMPUTE_TIME)``.
           FAILS on the pre-fix source.

  CHECK 2a STATEMENT SHAPE.  The fence line, with comments stripped, is
           EXACTLY the synchronizing statement -- it begins with the call and
           ends at its semicolon.  A line that begins with ``if``, ``else``,
           a ternary condition, or any other text is a fence whose execution
           some expression governs.  Catches every single-line form above.

  CHECK 2b CONTROLLING HEADER.  The nearest preceding code-bearing line is
           not a dangling control header (``if (...)``, ``else``, ``for``,
           ``while``, ``switch``, ``do``, a ``case``/``default`` label) and
           is not a preprocessor conditional (``#if``/``#ifdef``/``#ifndef``/
           ``#else``/``#elif``).  A dangling header makes the next statement
           its controlled body with no brace anywhere.  Catches the two-line
           braceless form and a compile-time-conditional fence.

  CHECK 2c BRACE BALANCE.  The net brace delta from fence to stop is zero AND
           the running depth never dips below zero.  A negative excursion
           means the fence sits in a scope that CLOSES before the stop --
           the multi-line ``if (c) { fence; }`` form.  The running-minimum
           half additionally catches a balanced close-then-reopen (``} ... {``)
           that a net delta of zero would wave through.

WHAT THIS INSTRUMENT IS BLIND TO
--------------------------------
Every instrument is blind to something; these are this one's blind spots,
stated so a later reader does not have to rediscover them.

  * A MACRO that expands to a conditional.  ``MAYBE_FENCE(streams);`` is a
    bare statement on its own line and passes 2a, 2b and 2c, whatever the
    macro expands to.  This guard does not preprocess.
  * A fence moved INSIDE a callee, a lambda, or a helper invoked from here.
    Only the text between the located fence and the located stop is read.
  * ``goto`` or a label-based jump that skips the fence.
  * A string literal containing a brace or a ``//``.  Line and block comments
    ARE stripped (length-preservingly, so line indices stay valid); string
    literals are not parsed.  The COMPUTE_TIME span in triton.h contains no
    such literal, and CHECK 2b would flag the shape change if one appeared.
  * Two statements sharing the fence's line.  2a's end-anchor rejects that
    form, so it is deliberately over-strict rather than blind -- the failure
    is a false NEGATIVE-free false positive, which is the safe direction.
  * Anything about RUNTIME re-attribution.  This is a source-structural test.
    The runtime property needs two authorized GPU coupled runs and is out of
    scope by design.

DEMONSTRATING THE FAILING FORM
------------------------------
Run it against the unfenced source to see CHECK 1 red::

    git show a38338b0:src/triton.h > /tmp/pre.h
    python3 test/test_compute_timer_fence.py /tmp/pre.h     # exits 1

USAGE
-----
    python3 test/test_compute_timer_fence.py [path-to-triton.h]

Exit 0 = pass, 1 = fail.  No build, no GPU and no simulation required, so it
runs on any host at review time.
"""

import os
import re
import sys

KERNEL_LAUNCH = re.compile(r"\b(?:Kernels::\w+\s*\(|triton::parallel_for\s*\()")
SYNC_NAMES = r"(?:gpuStreamSynchronize|Kokkos::fence|gpuMemcpyAsync)"
SYNC_CALL = re.compile(r"\b" + SYNC_NAMES + r"\s*\(")
START_COMPUTE = re.compile(r"\bst\.start\s*\(\s*COMPUTE_TIME\s*\)")
STOP_COMPUTE = re.compile(r"\bst\.stop\s*\(\s*COMPUTE_TIME\s*\)")
START_SWMM = re.compile(r"\bst\.start\s*\(\s*SWMM_TIME\s*\)")

# CHECK 2a -- the line is EXACTLY the synchronizing statement.
BARE_SYNC_STMT = re.compile(r"^\s*" + SYNC_NAMES + r"\s*\([^;{}]*\)\s*;\s*$")

# CHECK 2b -- a header that makes the FOLLOWING statement its controlled body.
DANGLING_CTRL = re.compile(
    r"^(?:\}\s*)?(?:else\s+)?(?:if|for|while|switch)\s*\(.*\)\s*$"
    r"|^(?:\}\s*)?else\s*$"
    r"|^do\s*$"
    r"|^(?:case\b.*|default)\s*:\s*$"
)
PREPROC_COND = re.compile(r"^#\s*(?:if|ifdef|ifndef|else|elif)\b")


def strip_comments(text):
    """Remove // and /* */ comments, PRESERVING length and newlines so every
    line index into the result still addresses the same source line."""
    out = []
    i, n = 0, len(text)
    in_line, in_block = False, False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line:
            if c == "\n":
                in_line = False
                out.append(c)
            else:
                out.append(" ")
            i += 1
        elif in_block:
            if c == "*" and nxt == "/":
                in_block = False
                out.append("  ")
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
        elif c == "/" and nxt == "/":
            in_line = True
            out.append("  ")
            i += 2
        elif c == "/" and nxt == "*":
            in_block = True
            out.append("  ")
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


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


def brace_profile(code, lo, hi):
    """(net delta, running minimum) of brace depth over code[lo:hi]."""
    depth, lowest = 0, 0
    for ch in "".join(code[lo:hi]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < lowest:
                lowest = depth
    return depth, lowest


def preceding_code_line(code, idx):
    """Index of the nearest code-bearing line above `idx`, or None."""
    for k in range(idx - 1, -1, -1):
        if code[k].strip():
            return k
    return None


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "triton.h")
    path = os.path.normpath(path)

    with open(path, "r", errors="replace") as fh:
        code = strip_comments(fh.read()).split("\n")

    failures = []

    stops = find_coupled_stop(code)
    if len(stops) != 1:
        print("FAIL: expected exactly one st.stop(COMPUTE_TIME) immediately followed by "
              "st.start(SWMM_TIME); found %d." % len(stops))
        print("      The coupled block's shape has changed; this test must be re-grounded.")
        return 1
    stop_i = stops[0]

    start_i = None
    for i in range(stop_i - 1, -1, -1):
        if START_COMPUTE.search(code[i]):
            start_i = i
            break
    if start_i is None:
        print("FAIL: no st.start(COMPUTE_TIME) found above the coupled stop at line %d."
              % (stop_i + 1))
        return 1

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

    delta = lowest = None
    if sync_i is None:
        failures.append(
            "CHECK 1 FAILED: no synchronizing call between the last kernel launch "
            "(line %d: %s) and st.stop(COMPUTE_TIME) (line %d).\n"
            "    Device-backend kernels are still in flight when the timer stops, so the "
            "first deep_copy inside SWMM_TIME drains them and the drain is charged to the "
            "SWMM column." % (last_launch + 1, code[last_launch].strip()[:70], stop_i + 1))
    else:
        fence_line = code[sync_i]

        # CHECK 2a -- the fence line is EXACTLY the synchronizing statement.
        if not BARE_SYNC_STMT.match(fence_line):
            failures.append(
                "CHECK 2a FAILED: the fence at line %d is not a bare statement:\n"
                "        %s\n"
                "    Some expression on this line governs whether the fence executes -- an "
                "if, an else, a ternary, or a second statement. The fence must run on every "
                "path that reaches st.stop(COMPUTE_TIME) at line %d; a guarded fence leaves "
                "the mis-attribution standing on every path its condition excludes."
                % (sync_i + 1, fence_line.strip()[:100], stop_i + 1))

        # CHECK 2b -- the fence is not the controlled body of a dangling header.
        prev_i = preceding_code_line(code, sync_i)
        if prev_i is not None:
            prev = code[prev_i].strip()
            why = None
            if PREPROC_COND.match(prev):
                why = "a preprocessor conditional"
            elif DANGLING_CTRL.match(prev):
                why = "a control header with no opening brace"
            if why is not None:
                failures.append(
                    "CHECK 2b FAILED: the line above the fence is %s:\n"
                    "        line %d: %s\n"
                    "        line %d: %s\n"
                    "    The fence is that header's controlled body, so it executes only "
                    "when the header admits it -- a conditional fence that carries no brace "
                    "imbalance and that a net-brace-delta check cannot see."
                    % (why, prev_i + 1, prev[:100], sync_i + 1, fence_line.strip()[:100]))

        # CHECK 2c -- brace balance, net AND running minimum.
        delta, lowest = brace_profile(code, sync_i, stop_i)
        if delta != 0 or lowest < 0:
            failures.append(
                "CHECK 2c FAILED: the fence at line %d and st.stop(COMPUTE_TIME) at line %d "
                "are not in the same scope (net brace delta %+d, running minimum %+d; both "
                "must be 0).\n"
                "    A negative excursion means the fence sits inside a scope that closes "
                "before the stop -- the multi-line `if (c) { fence; }` form, and the hazard "
                "of copying the conditional sibling at triton.h:2484, whose fence is inside "
                "if (arglist.gpu_direct_flag)." % (sync_i + 1, stop_i + 1, delta, lowest))

    print("file            : %s" % path)
    print("COMPUTE_TIME    : start line %d -> coupled stop line %d" % (start_i + 1, stop_i + 1))
    print("last launch     : line %d  %s" % (last_launch + 1, code[last_launch].strip()[:70]))
    if sync_i is not None:
        print("fence           : line %d  %s" % (sync_i + 1, code[sync_i].strip()[:70]))
        print("bare statement  : %s" % ("yes" if BARE_SYNC_STMT.match(code[sync_i]) else "NO"))
        prev_i = preceding_code_line(code, sync_i)
        print("line above      : line %s  %s"
              % (prev_i + 1 if prev_i is not None else "-",
                 code[prev_i].strip()[:70] if prev_i is not None else "(none)"))
        print("brace delta fence->stop : %+d (0 required), running minimum %+d (0 required)"
              % (delta, lowest))
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

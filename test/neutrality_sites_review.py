#!/usr/bin/env python3
"""Reviewer probes of the N11 source-site scanner after the E-75 rows R-1 and R-2 (chunk 3d.1).

Property: for every capture call of src/triton.h inside an `#ifdef TRITON_SWMM` region or the
`global_num_of_swmm_links` guard, outside the T1 rows, the scanner reports a violation unless the
call is the first R-1 setup call or lies in the body of the ledger function and is its DIAG_TIME
bracket or its add_outfall_backflow call (erratum E-75).

Class: single-line edits of a copy of the branch's triton.h, each made by this script.

Why required: rule 6 of the shape declares the set of capture positions CLOSED, and N11 is the only
check that enforces it; a row's exemption that reaches past its own position silently widens the set.

What kills each probe: the named edit of the scanner (see each probe). Second correct implementation:
a scanner keyed on a parsed syntax tree instead of regexes passes every probe, because each probe
asserts the verdict on the edited source and no regex.

Usage: python test/neutrality_sites_review.py [path/to/src/triton.h]   (exit 0 when every probe reads as expected)
"""
import importlib.util
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("neut", os.path.join(HERE, "neutrality_compare.py"))
neut = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neut)

TRI = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "src", "triton.h")
SRC = open(TRI, "rb").read().decode().split("\n")
fails = []


def scan(lines):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "triton_probe.h")
    open(p, "wb").write("\n".join(lines).encode())
    return neut.scan_sites(p)


def find(pattern, nth=0):
    hits = [i for i, l in enumerate(SRC) if re.search(pattern, l)]
    return hits[nth]


def expect(name, cond, detail):
    sys.stdout.write("%s %s: %s\n" % ("ok  " if cond else "RED ", name, detail))
    if not cond:
        fails.append(name)


# P0 the branch's own file scans clean, and the scan reads the rows (a must-be-nonzero control: the same
# file scanned with the R-1 and R-2 matchers disabled reports the four lines E-75 names)
v, ex, tot = scan(SRC)
expect("P0/clean", len(v) == 0 and tot > 0, "violations %d, exempt %d, capture-call lines %d" % (len(v), ex, tot))
saved = (neut.R1_CALL, neut.R2_DEF)
never = re.compile(r"(?!x)x")
neut.R1_CALL, neut.R2_DEF = never, never
v0, ex0, _ = scan(SRC)
neut.R1_CALL, neut.R2_DEF = saved
expect("P0/control", len(v0) == 4 and ex - ex0 == 4, "rows disabled: %d violations (E-75 names 4), %d lines exempt by the rows" % (len(v0), ex - ex0))

# P1 a second R-1 setup call is a call at an unlisted position
i1 = find(r"enable_domain_balance\(arglist\.diag_domain_balance != 0, swmm_model\.driven_active\(\)\)")
dup = SRC[:i1 + 1] + [SRC[i1]] + SRC[i1 + 1:]
v1, _, _ = scan(dup)
expect("P1/second-R1", len(v1) == 1, "violations %d (want exactly 1: the second copy)" % len(v1))

# P2 a DIAG_TIME bracket in the SWMM region outside the ledger function body is not R-2
i2 = find(r"if \(diag_surface\.domain_balance && swmm_model\.driven_active\(\)\) diag_driven_outfall_ledger\(\);")
ins = SRC[:i2] + ["      st.start(DIAG_TIME); st.stop(DIAG_TIME);"] + SRC[i2:]
v2, _, _ = scan(ins)
expect("P2/bracket-outside-body", len(v2) >= 1, "violations %d (want at least 1)" % len(v2))

# P3 another capture call inside the ledger function body is not one of the two R-2 calls
i3 = find(r"diag_surface\.add_outfall_backflow\(")
body = SRC[:i3] + ["    diag_surface.probe_other_capture_call(0.0);"] + SRC[i3:]
v3, _, _ = scan(body)
expect("P3/other-call-in-body", len(v3) >= 1, "violations %d (want at least 1)" % len(v3))

# P4 the ledger wrapper itself called at a position no row lists: inside the coupling block's SWMM_TIME
# bracket right after the swmm_step return-code check (an `#ifdef TRITON_SWMM` region by compile necessity,
# because the wrapper exists only there); the wrapper hides its capture calls from CAPTURE_CALL
i4 = find(r'stop_on_swmm_error\(swmm_step_rc, "swmm_step", "during a coupled timestep"\);')
call = SRC[:i4 + 1] + ["          diag_driven_outfall_ledger();"] + SRC[i4 + 1:]
v4, _, _ = scan(call)
expect("P4/wrapper-call-at-unlisted-position", len(v4) >= 1,
       "violations %d (want at least 1: rule 6 closes the capture-call positions, and this call runs the capture in %s)"
       % (len(v4), "diag_driven_outfall_ledger"))

sys.stdout.write("RESULT %s neutrality_sites_review (%d red)\n" % ("PASS" if not fails else "FAIL", len(fails)))
sys.exit(1 if fails else 0)

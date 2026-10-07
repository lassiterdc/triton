#!/usr/bin/env python3
"""REVIEW ARTIFACT (structural reviewer) -- a
CHARACTERIZED LIMITATION of test_review_c_linkage_boundary.py, expressed as a
failing test rather than as prose.

Usage:  test_review8_linkage_guard_preprocessor_blindspot.py <repo-root>

THIS FILE IS EXPECTED TO FAIL TODAY.  It is NOT registered in CMakeLists.txt
and it is NOT a withhold against fea07ee.  It exists so that a claim which
quantifies over a class is falsifiable by running something rather than by
reading an argument, and so that a future author who narrows the limitation
has an instrument that flips.

THE CLAIM.  ``test_review_c_linkage_boundary.py``'s L1 asserts "every funcs.h
symbol a C++ TU calls has C linkage there".  Its exemption predicate is
``locally_c_declared(tu_src, name)``, which searches the TU's SOURCE TEXT for
the name inside any brace-matched ``extern "C"`` span.  The search is
deliberately PREPROCESSOR-BLIND -- and it has to be, because funcs.h's own
block is written as::

    #ifdef __cplusplus
    extern "C" {
    #endif

so a preprocessor-aware span finder would not see that block open at all, and
the partition L1 rests on would collapse.  The blindness is therefore a term of
a necessary design choice, not an oversight.

THE CONSEQUENCE, which is what this file measures.  A re-declaration that is
PRESENT IN THE SOURCE TEXT but EXCLUDED BY THE PREPROCESSOR satisfies
``locally_c_declared`` while contributing nothing to the compiled TU.  The link
then fails exactly as it does with no declaration at all, and L1 reports green.
That is a FALSE GREEN inside the guard's own failure class -- a silent
link-time break -- as distinct from a misplaced-but-compiled declaration, whose
failure is a LOUD g++ diagnostic (measured: ``error: conflicting declaration of
'void output_end()' with 'C' linkage``) and which this file does not treat as a
defect.

EXPOSURE TODAY IS ZERO, MEASURED.  A grep over the scanned TU set finds no
``extern "C"`` block under any preprocessor conditional other than funcs.h's
own ``__cplusplus`` guard.  P3 below re-measures that every run, so the day the
exposure becomes non-zero is the day P3 goes red rather than the day someone
re-reads this docstring.

A NARROWING THAT DOES NOT BREAK funcs.h, offered rather than required: exempt a
span only when its opener is NOT inside a ``#if``/``#ifdef``/``#ifndef`` region
whose controlling expression is something other than ``__cplusplus``.  That
keeps funcs.h's block visible and removes ``#if 0`` and ``#ifdef _WIN32``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

GUARD = "test/test_review_c_linkage_boundary.py"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("ok    %s" if ok else "FAIL  %s   %s") % ((name,) if ok else (name, detail)))
    if not ok:
        failures.append("%s: %s" % (name, detail))


def load_guard(repo: Path):
    path = repo / GUARD
    spec = importlib.util.spec_from_file_location("linkage_guard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: %s <repo-root>" % argv[0])
        return 2
    repo = Path(argv[1]).resolve()
    guard = load_guard(repo)
    funcs_h = (repo / "external" / "swmm" / "src" / "solver" / "funcs.h").read_text(
        errors="replace")

    inside, outside = guard.partition_funcs_h(funcs_h)
    sym = "output_end" if "output_end" in outside else sorted(outside)[0]

    # --- P0: the instrument is live before any verdict is read ---------------
    #     Without this, every P below could pass by the predicate being dead.
    no_decl = '#include "swmm_triton.h"\nvoid probe(void) { %s(); }\n' % sym
    check("P0  the guard's predicate reports an undeclared outside symbol",
          bool(guard.offenders_in_source(funcs_h, "<p0>", no_decl)),
          "the L1 predicate did not flag a bare call to %s -- every check "
          "below would pass vacuously" % sym)

    # --- P1: the correct repair is accepted (the other control direction) ----
    good = ('extern "C" {\nvoid %s(void);\n}\n#include "swmm_triton.h"\n'
            'void probe(void) { %s(); }\n' % (sym, sym))
    check("P1  the guard accepts a correctly placed re-declaration",
          not guard.offenders_in_source(funcs_h, "<p1>", good),
          "the prescribed repair was reported as an offender")

    # --- P2: THE CLAIM. A preprocessor-EXCLUDED declaration must not exempt --
    #     Two independent exclusion forms, because a fix that handles only
    #     `#if 0` leaves the platform-conditional form open.
    hidden_if0 = ('#if 0\nextern "C" {\nvoid %s(void);\n}\n#endif\n'
                  '#include "swmm_triton.h"\nvoid probe(void) { %s(); }\n'
                  % (sym, sym))
    hidden_win = ('#ifdef _WIN32\nextern "C" {\nvoid %s(void);\n}\n#endif\n'
                  '#include "swmm_triton.h"\nvoid probe(void) { %s(); }\n'
                  % (sym, sym))
    check("P2a a declaration inside `#if 0` does NOT exempt the call",
          bool(guard.offenders_in_source(funcs_h, "<p2a>", hidden_if0)),
          "locally_c_declared() accepted a declaration the preprocessor "
          "removes, so L1 reports green on a TU that will fail at link with "
          "`undefined reference to '%s()'`" % sym)
    check("P2b a declaration inside `#ifdef _WIN32` does NOT exempt the call "
          "on a non-Windows build",
          bool(guard.offenders_in_source(funcs_h, "<p2b>", hidden_win)),
          "same class as P2a in the platform-conditional form, which is the "
          "one that can be written by accident rather than on purpose")

    # --- P3: exposure meter. Red the day the blind spot becomes reachable ----
    conditional_spans = []
    for p in guard.cxx_translation_units(repo):
        try:
            src = p.read_text(errors="replace")
        except OSError:
            continue
        code = guard._blank(src, strings=False)
        for a, _b in guard._extern_c_spans(code):
            head = code[:a]
            # the innermost still-open conditional at the opener, if any
            depth = 0
            controller = None
            for m in re.finditer(r'^\s*#\s*(if|ifdef|ifndef|endif)\b([^\n]*)',
                                 head, re.M):
                if m.group(1) == "endif":
                    depth = max(0, depth - 1)
                    if depth == 0:
                        controller = None
                else:
                    depth += 1
                    if depth == 1:
                        controller = m.group(2).strip()
            if depth > 0 and controller and "__cplusplus" not in controller:
                conditional_spans.append(
                    "%s (controlled by `#if %s`)"
                    % (p.relative_to(repo), controller))
    check("P3  no scanned TU carries an extern \"C\" block under a "
          "non-__cplusplus conditional",
          not conditional_spans,
          "the P2 blind spot is now REACHABLE at: " + "; ".join(conditional_spans))

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        print("")
        print("EXPECTED FAILURE at fea07ee: P2a and P2b are the characterized "
              "limitation this file exists to express. P0, P1 and P3 passing "
              "alongside them is what makes that reading admissible -- P0/P1 "
              "prove the predicate is live in both directions, and P3 proves "
              "the limitation is UNREACHABLE in the tree as it stands.")
        return 1
    print("PASS: the linkage guard's exemption predicate is preprocessor-aware.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

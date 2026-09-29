#!/usr/bin/env python3
"""REVIEW ARTIFACT (structural reviewer, WP-1C round 7) -- the C/C++ linkage
boundary of funcs.h.

Usage:  test_review_c_linkage_boundary.py <repo-root>

WHY THIS EXISTS.  ``funcs.h`` carries its ``extern "C"`` linkage specification
over a PROPER SUBSET of its declarations: the block opens at the ``#ifdef
__cplusplus`` / ``extern "C" {`` pair partway down the file and closes before
the tail.  Every declaration OUTSIDE that window acquires C++ LANGUAGE LINKAGE
in any C++ translation unit that includes the header.

THE OUTSIDE SET IS BOTH SIDES OF THE BLOCK, and saying "above" understates it
by half.  Measured at the WP-1C commit: the window is lines 298-393, and of
324 declarations 11 are inside, 150 are ABOVE and 163 are BELOW -- so the tail
(the ``stats_*`` / ``gage_*`` / ``subcatch_*`` families) is exposed on exactly
the same terms as the head.  The code below has always partitioned by span
membership and is therefore correct on both sides; only this prose said
"above", and a reader who trusts it enumerates half the class.  The definitions
all live in ``.c`` files that CMake compiles as C, so a C++ call to one of them
emits a MANGLED reference against an UNMANGLED definition and fails at LINK.

``-fsyntax-only`` cannot see this.  The declaration is visible and well formed;
only the link exposes the mismatch.  That is exactly the ``extern "C"``
signature/linkage drift the WP-1C commit message names as the gate it could not
close, so the property is asserted here in the no-compile tier instead.

THE CHECK QUANTIFIES OVER A CLASS, not over one symbol: every funcs.h-declared
function that any C++ translation unit calls must have C linkage at that call
site -- either because its declaration sits inside funcs.h's ``extern "C"``
block, or because the calling TU re-declares it inside a local ``extern "C"``
block of its own.  A future C++ caller of a second pre-block symbol is caught
by the same run.

EVERY CHECK CARRIES A TWO-STATE CONTROL.  A structural check fails by silently
matching nothing, so each L* check below is paired with a control that mutates
the source IN MEMORY and asserts the check FLIPS.  A control that does not flip
is reported as VACUOUS and fails the run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

failures: list[str] = []
results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name.split()[0]] = ok
    print(("ok    %s" if ok else "FAIL  %s   %s") % ((name,) if ok else (name, detail)))
    if not ok:
        failures.append("%s: %s" % (name, detail))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _blank(src: str, strings: bool) -> str:
    """Blanks // and /* */ comments, and OPTIONALLY string/char literals,
    preserving every character offset.

    THE TWO MODES ARE NOT INTERCHANGEABLE AND THIS FILE PAID TO LEARN IT.
    Call-site detection MUST blank strings -- the C++ test names ``output_end``
    inside a CHECK message and inside a dispatch-table literal, and neither is
    a call.  Span detection MUST NOT: the block opener is literally
    ``extern "C" {``, so blanking strings erases the ``"C"`` and the opener
    stops matching.  The first version of this file blanked strings for both,
    found ZERO extern "C" blocks, and reported every one of funcs.h's 324
    declarations as OUTSIDE -- a confident, uniformly wrong verdict.  L0a and
    L0b are what turned that into a loud failure instead of a finding.
    """
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if strings and c in '"\'':
            q = c
            out.append(" ")
            i += 1
            while i < n:
                if src[i] == "\\" and i + 1 < n:
                    out.append("  "); i += 2; continue
                if src[i] == q:
                    out.append(" "); i += 1; break
                out.append("\n" if src[i] == "\n" else " "); i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out.append(" "); i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                out.append("\n" if src[i] == "\n" else " "); i += 1
            out.append("  "); i += 2
            continue
        out.append(c); i += 1
    return "".join(out)


_EXTERN_C_OPEN = re.compile(r'^\s*extern\s+"C"\s*\{', re.M)

#: a top-level C function declaration ending in `);`
_DECL = re.compile(
    r'^[ \t]*[A-Za-z_][A-Za-z0-9_ \t\*]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*;',
    re.M | re.S)


def _extern_c_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of every brace-matched `extern "C" {` block."""
    spans = []
    for m in _EXTERN_C_OPEN.finditer(text):
        i = text.index("{", m.start())
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((i, j))
                    break
            j += 1
    return spans


def partition_funcs_h(funcs_h: str) -> tuple[set[str], set[str]]:
    """(inside, outside) -- function names declared in / out of funcs.h's
    `extern "C"` block(s)."""
    code = _blank(funcs_h, strings=False)
    spans = _extern_c_spans(code)
    inside, outside = set(), set()
    for m in _DECL.finditer(code):
        name = m.group(1)
        pos = m.start(1)
        (inside if any(a < pos < b for a, b in spans) else outside).add(name)
    return inside, outside


def called_names(tu_src: str, candidates: set[str]) -> set[str]:
    code = _blank(tu_src, strings=True)
    hit = set()
    for name in candidates:
        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'\s*\(', code):
            hit.add(name)
    return hit


def locally_c_declared(tu_src: str, name: str) -> bool:
    """True when the TU re-declares `name` inside its own `extern "C"` block."""
    code = _blank(tu_src, strings=False)
    for a, b in _extern_c_spans(code):
        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'\s*\(', code[a:b]):
            return True
    return False


def cxx_translation_units(repo: Path) -> list[Path]:
    """The C++ TUs that reach funcs.h. swmm_triton.h includes it directly and
    is header-only, so every .cpp/.h that pulls swmm_triton.h reaches it too."""
    out = []
    for base in ("src", "test"):
        root = repo / base
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            # `.h` is scanned under BOTH bases. It was originally src-only, which
            # left a hole exactly the size of this check's own failure class: a
            # future test-side header calling an outside-the-block symbol would
            # not have been looked at, and the guard would have reported green
            # on the defect it exists to catch. The tree carries no such header
            # today (only the Kokkos config shim, which calls nothing), so the
            # widening is a no-op now and a closed hole later.
            if p.suffix in (".cpp", ".hpp", ".h"):
                out.append(p)
    return out


def offenders_in_source(funcs_h: str, label: str, src: str) -> list[tuple[str, str]]:
    """The L1 predicate over ONE translation unit's SOURCE TEXT.

    Factored out of offenders() so a control can run the very same predicate
    against a synthetic TU instead of against the tree.  That is what lets C1
    below hold in BOTH states; see its comment for why it had to.
    """
    inside, outside = partition_funcs_h(funcs_h)
    bad = []
    for name in sorted(called_names(src, outside)):
        if name in inside:
            continue
        if locally_c_declared(src, name):
            continue
        bad.append((label, name))
    return bad


def offenders(repo: Path, funcs_h: str, tus: list[Path]) -> list[tuple[str, str]]:
    bad = []
    for p in tus:
        try:
            src = p.read_text(errors="replace")
        except OSError:
            continue
        bad.extend(offenders_in_source(funcs_h, str(p.relative_to(repo)), src))
    return bad


# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: %s <repo-root>" % argv[0])
        return 2
    repo = Path(argv[1]).resolve()
    fh_path = repo / "external" / "swmm" / "src" / "solver" / "funcs.h"
    funcs_h = fh_path.read_text(errors="replace")
    tus = cxx_translation_units(repo)

    inside, outside = partition_funcs_h(funcs_h)

    # --- L0: the instrument is live before any verdict is read ---------------
    check("L0a funcs.h carries at least one extern \"C\" block",
          bool(_extern_c_spans(_blank(funcs_h, strings=False))),
          "no extern \"C\" block found -- the partition below would put every "
          "declaration OUTSIDE and L1 would fail for the wrong reason")
    check("L0b the partition is non-empty on BOTH sides",
          bool(inside) and bool(outside),
          "inside=%d outside=%d -- a one-sided partition makes L1 vacuous"
          % (len(inside), len(outside)))
    check("L0c at least one C++ TU calls a funcs.h symbol",
          any(called_names(p.read_text(errors="replace"), inside | outside)
              for p in tus),
          "no C++ call site found at all -- L1 would pass vacuously")

    # --- L1: the property ----------------------------------------------------
    bad = offenders(repo, funcs_h, tus)
    check("L1  every funcs.h symbol a C++ TU calls has C linkage there",
          not bad,
          "MANGLED-REFERENCE-AGAINST-C-DEFINITION at: "
          + "; ".join("%s -> %s()" % (f, n) for f, n in bad)
          + " -- declared in funcs.h OUTSIDE its extern \"C\" block "
            "(funcs.h opens the block partway down the file), defined in a .c "
            "compiled as C. Predicted link error: undefined reference to "
            "'<name>()'. Repair: re-declare the symbol inside the CALLING "
            "TU's own extern \"C\" block (one line, zero vendored surface). "
            "THE POSITION IS LOAD-BEARING: the block must sit BEFORE the "
            "include that pulls funcs.h. Placed after it -- e.g. alongside a "
            "TU's existing extern \"C\" globals block -- it is a second "
            "declaration giving a different language linkage to an "
            "already-declared name, which [dcl.link]p6 makes ill-formed and "
            "which g++ rejects: \"conflicting declaration of 'void f()' with "
            "'C' linkage / previous declaration with 'C++' linkage\". A "
            "globals block is NOT the model, because globals.h is typically "
            "not included and those names have no prior declaration to "
            "conflict with; a funcs.h function always does.")

    # --- controls ------------------------------------------------------------
    print("")

    # C1: the STATED REPAIR clears the defect -- and this control is
    #     STATE-INDEPENDENT, which it has to be to survive registration.
    #
    #     THE ORIGINAL FORM COULD NOT SURVIVE ITS OWN REPAIR.  It read
    #     `(results.get("L1") is False) and not offenders(repo, repaired, tus)`,
    #     so it asserted the tree is BROKEN as a conjunct of its own pass.  That
    #     is right for a one-shot review artifact aimed at a red tree and wrong
    #     for a registered guard: the moment the defect it diagnosed was fixed,
    #     C1 went red with "L1 is already green (nothing to repair)" on every
    #     subsequent run.  A control that is red in the healthy state teaches a
    #     reader to discount it, which costs exactly the signal it exists to
    #     provide.
    #
    #     The repair is to run the SAME L1 predicate over a SYNTHETIC TU, so
    #     both arms hold whatever the tree is doing:
    #       arm (a) a C++ TU calling an outside-the-block symbol, with NO local
    #               re-declaration, MUST be reported  -> the check can go red;
    #       arm (b) the same TU carrying the stated repair -- the symbol
    #               re-declared in the TU's own extern "C" block -- MUST come
    #               back clean -> the repair this file PRESCRIBES is the repair
    #               it ACCEPTS;
    #       arm (c) when a LIVE offender exists, moving its funcs.h declaration
    #               inside the block must also clear it.  This is the original
    #               C1 verbatim, retained rather than dropped; it is vacuously
    #               true on a green tree and is the arm that fires on a red one.
    repaired = funcs_h
    if bad:
        name = bad[0][1]
        decl = re.search(r'^[^\n]*\b' + re.escape(name) + r'\s*\([^;]*\);[^\n]*$',
                         funcs_h, re.M)
        if decl:
            repaired = (funcs_h[:decl.start()] + funcs_h[decl.end():])
            open_m = _EXTERN_C_OPEN.search(repaired)
            ins = repaired.index("\n", open_m.start()) + 1
            repaired = repaired[:ins] + decl.group(0) + "\n" + repaired[ins:]

    _inside, _outside = partition_funcs_h(funcs_h)
    _probe = sorted(n for n in _outside if n not in _inside)
    if not _probe:
        check("C1  -> L1 the stated repair CLEARS it", False,
              "no outside-the-block symbol exists, so no synthetic control can "
              "be built -- L0b should already have caught this")
    else:
        sym = "output_end" if "output_end" in _probe else _probe[0]
        broken_tu = "void c1_caller(void) { %s(); }\n" % sym
        fixed_tu = ('extern "C" {\nvoid %s(void);\n}\n'
                    'void c1_caller(void) { %s(); }\n' % (sym, sym))
        arm_a = offenders_in_source(funcs_h, "<c1-synthetic>", broken_tu)
        arm_b = offenders_in_source(funcs_h, "<c1-synthetic>", fixed_tu)
        arm_c = (not offenders(repo, repaired, tus)) if bad else True
        check("C1  -> L1 the stated repair CLEARS it",
              bool(arm_a) and not arm_b and arm_c,
              "probe symbol %s; arm(a) synthetic defect reported=%r (want "
              "NON-EMPTY -- the predicate cannot go red, so L1's green is "
              "vacuous); arm(b) same TU after the stated repair=%r (want EMPTY "
              "-- the prescribed repair is not the one this check accepts); "
              "arm(c) live-offender funcs.h move clears L1=%r (vacuously True "
              "when L1 is already green)" % (sym, arm_a, arm_b, arm_c))

    # C2: moving a currently-INSIDE symbol OUT must turn L1 red, proving the
    #     check discriminates on the block boundary rather than on a name list.
    victim = "output_positionForResume"
    broken = funcs_h
    dm = re.search(r'^[^\n]*\b' + re.escape(victim) + r'\s*\([^;]*\);[^\n]*$',
                   funcs_h, re.M)
    if dm:
        close = re.search(r'^#ifdef __cplusplus\s*\n\}\s*\n#endif', funcs_h, re.M)
        broken = funcs_h[:dm.start()] + funcs_h[dm.end():]
        tail = re.search(r'^#ifdef __cplusplus\s*\n\}\s*\n#endif', broken, re.M)
        broken = broken[:tail.end()] + "\n" + dm.group(0) + "\n" + broken[tail.end():]
        _ = close
    check("C2  -> L1 a symbol moved OUT of the block is CAUGHT",
          bool(dm) and any(n == victim for _, n in offenders(repo, broken, tus)),
          "moving %s outside the block did not make L1 report it -- the check "
          "is not keyed on the block boundary" % victim)

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: every funcs.h symbol called from C++ has C linkage at its call site.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

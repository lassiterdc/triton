#!/usr/bin/env python3
"""Fast-tier syntax check: every C++ translation unit must PARSE.

WHAT THIS IS
------------
``mpic++ -fsyntax-only`` over each ``.cpp`` in the tree, with a six-file Kokkos
config shim checked in beside this script.  No CMake configure, no Kokkos build,
no link, no GPU, no solver run.  Measured on this tree: five TUs at 1.2-1.6 s
each.

A full build of this project configures and compiles Kokkos first, which is why
a broken declaration in a header routinely reaches review -- the cheapest
instrument that would have caught it costs minutes to set up and is therefore
not run.  This node costs seconds and is registered, so it is run.

It is not hypothetical value: this instrument caught a real symbol redefinition
on this branch (``_snapDummyOutfall``/``_snapDummyStorage`` colliding with the
existing Stats-type dummies), fixed at d648106.

WHY THE SHIM
------------
``Kokkos_Macros.hpp`` includes ``KokkosCore_config.h`` and four
``KokkosCore_Config_*.hpp`` files that Kokkos' own CMake GENERATES at configure
time; ``desul/atomics/Config.hpp`` is generated the same way.  Without them the
parse dies in Kokkos' headers long before it reaches any TRITON source.  The
six files in ``kokkos-shim/`` are a SERIAL-backend configuration -- the minimum
that lets the headers parse.  They are a PARSE fixture and nothing else: they
select no real backend, and no binary is produced from them.

GATE ON EXIT CODE, NOT ON ZERO DIAGNOSTICS
------------------------------------------
This node asserts each TU's compiler EXIT STATUS is 0.  It does NOT assert the
absence of warnings, and a future edit must not "tighten" it into doing so.
Measured at this commit with ``-Wall -Wextra``:

  * 4 warnings from PROJECT source on every test TU -- all
    ``src/kokkos_utils.h:52,79,81,83``, unused parameter ``str``.  These are
    REAL and correct: the SERIAL backend's ``Kokkos::fence()`` takes no stream
    argument, so the stream parameter of the ``gpu*`` wrappers is genuinely
    unused in that configuration.
  * 11 warnings from PROJECT source on ``src/main.cpp`` (extbc.h 2, kernels.h 2,
    kokkos_utils.h 4, output.h 2, triton.h 1).
  * 2 further warnings on EVERY TU from the toolchain's own OpenMPI headers
    (``op_inln.h``, ``-Wcast-function-type``).  Their count is a property of the
    installed MPI, not of this tree, so even a project-scoped zero-diagnostics
    gate would be fragile across sites and a whole-output one is worse.

The single ``src/triton.h`` warning is an unused ``nbytes_swmm_int`` in
``compute_new_state``.  It traces to UPSTREAM ``ebc2f00`` ("Integrate SWMM
coupling into TRITON v2 core solver"), which is an ancestor of this branch's
base.  It is RECORDED here and deliberately NOT fixed: a drive-by edit to
upstream-authored lines costs review attention on a branch a maintainer ports by
reading its diff.

KNOWN BLIND SPOT, stated so a green is not over-read
----------------------------------------------------
``-fsyntax-only`` CANNOT catch ``extern "C"`` signature drift.  C linkage
matches on NAME ALONE -- no parameter or type information is mangled into the
symbol -- so a C++ ``extern "C"`` declaration whose type disagrees with the C
definition parses cleanly, links cleanly, and misreads memory at run time.

This tree has one such surface: the 24 declarations in
``test/snapshot/test_state_snapshot.cpp``'s ``extern "C"`` block, which reach
EPA SWMM globals.  A prior seat reports hand-checking all 24 against the solver
headers and finding them consistent.  THIS NODE DOES NOT VERIFY THAT, and the
claim is repeated here as provenance rather than as a result.  A mechanical
checker is buildable but is not a one-liner: several of the 24 are declared on
multi-declarator lines (``globals.h`` declares ``Nobjects[MAX_OBJ_TYPES]`` inside
a comma-separated run) and several others are file-scope globals in ``.c`` files
with no header declaration at all (``NodeStats``, ``LinkStats``,
``StorageStats``, ``OutfallStats``, ``NodeInflow``, ``NodeOutflow``).  A naive
one-symbol-per-line comparator reports 19 of 24 as missing, which is the
comparator being wrong rather than the declarations.  Closing this is tracked
separately.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# TUs are DISCOVERED, not listed, so a newly added .cpp is covered without an
# edit here -- a hand list silently under-covers, which is the failure mode this
# node exists to remove. The discovery is bounded below by ANCHOR_TU: a glob
# that matches nothing satisfies "every discovered TU parsed" vacuously, and
# that vacuous pass is byte-identical at the exit code to a real one.
TU_ROOTS = ("src", "test")
EXCLUDED_DIRS = ("external", ".git", "build")
ANCHOR_TU = Path("src/main.cpp")

# Include roots, relative to the repo root. `external/swmm/src/solver/include`
# is a SEPARATE root from `external/swmm/src/solver` and both are required:
# swmm5.h lives in the former, the internal headers in the latter.
INCLUDE_DIRS = (
    "test/syntax/kokkos-shim",
    "src",
    "external/swmm/src/solver",
    "external/swmm/src/solver/include",
    "external/kokkos/core/src",
    "external/kokkos/containers/src",
    "external/kokkos/algorithms/src",
    "external/kokkos/simd/src",
    "external/kokkos/tpls/desul/include",
)

# -DTRITON_SWMM is REQUIRED, not optional. Without it the whole SWMM_triton
# namespace is #ifdef'd out at swmm_triton.h:20 and the parse produces roughly
# 100 misleading "not declared in this scope" errors that read like real source
# faults.
DEFINES = ("-DTRITON_SWMM",)

BASE_FLAGS = ("-fsyntax-only", "-std=c++17", "-Wall", "-Wextra")


def discover_tus(repo):
    out = []
    for root in TU_ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.cpp")):
            rel = p.relative_to(repo)
            if any(part in EXCLUDED_DIRS for part in rel.parts):
                continue
            out.append(rel)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", nargs="?", default=".", type=Path)
    ap.add_argument("--compiler", default="mpic++",
                    help="MPI C++ compiler wrapper (default: mpic++)")
    ap.add_argument("--show-diagnostics", action="store_true",
                    help="echo each TU's compiler output even when it parses")
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    failures = []

    cxx = shutil.which(args.compiler)
    if cxx is None:
        # Fail loudly rather than skipping. A tree with no MPI C++ compiler
        # cannot build TRITON at all, so this is never a false failure -- and a
        # silent skip here would be a green that means nothing.
        print("FAIL: compiler %r not found on PATH. This node needs an MPI C++ "
              "wrapper; it does NOT need a configured build." % args.compiler)
        return 1
    print("compiler: %s" % cxx)

    missing = [d for d in INCLUDE_DIRS if not (repo / d).is_dir()]
    if missing:
        print("FAIL: missing include root(s): %s" % ", ".join(missing))
        print("      external/kokkos and external/swmm are git submodules -- "
              "clone with --recursive, or run")
        print("      git submodule update --init --recursive")
        return 1

    tus = discover_tus(repo)

    # Vacuity guard. Both halves are needed: a count floor alone passes on a
    # glob that found five unrelated files, and an anchor alone passes on a glob
    # that found only the anchor.
    if not tus:
        print("FAIL: discovered ZERO translation units under %s -- the glob "
              "broke, and 'every TU parsed' would be vacuously true."
              % ", ".join(TU_ROOTS))
        return 1
    if ANCHOR_TU not in tus:
        print("FAIL: discovered %d TU(s) but not the anchor %s. The glob is "
              "not reaching the tree it is supposed to cover."
              % (len(tus), ANCHOR_TU))
        for t in tus:
            print("        %s" % t)
        return 1

    inc = []
    for d in INCLUDE_DIRS:
        inc.append("-I%s" % (repo / d))

    print("%d translation unit(s) discovered\n" % len(tus))
    for rel in tus:
        cmd = [cxx, *BASE_FLAGS, *DEFINES, *inc, str(repo / rel)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        diag = (proc.stderr or "") + (proc.stdout or "")
        nwarn = diag.count("warning:")
        nerr = diag.count("error:")
        status = "ok   " if proc.returncode == 0 else "FAIL "
        print("%s %-58s exit=%d  %.2fs  %d warning(s)  %d error(s)"
              % (status, str(rel), proc.returncode, elapsed, nwarn, nerr))
        if proc.returncode != 0:
            failures.append(rel)
            print(diag.rstrip())
        elif args.show_diagnostics and diag.strip():
            print(diag.rstrip())

    print("")
    if failures:
        for rel in failures:
            print("FAIL: %s did not parse" % rel)
        return 1
    print("PASS: all %d translation unit(s) parse under %s -fsyntax-only."
          % (len(tus), args.compiler))
    print("      This asserts EXIT STATUS only. Warnings are expected and are "
          "NOT asserted -- see this file's docstring for the measured counts,")
    print("      and for the extern \"C\" signature-drift blind spot that no "
          "syntax-only check can close.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

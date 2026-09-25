#!/usr/bin/env python3
"""Guard-of-the-guard for Criterion P's closure check (WP-1B chunk 9).

WHY THIS EXISTS
---------------
``regenerate_snapshot_inventory.py --check`` returns 0 on the landed tree.  So
would a check whose pre-filter verified itself, whose triage tables swallowed
every name, or whose parser truncated the walk before the interesting units --
and none of that is visible from a green run.  §2.1(e) asks a closure check to
name the omission it catches and the document state in which it FAILS; this
file supplies that state, one mutation at a time.

Each DEFECT below materialises a throwaway copy of the EPA solver tree and the
inventory tooling, applies a single surgical mutation, re-runs the check there,
and asserts it goes RED **and prints the marker that names the right mode**.

  M1  drop a triaged field from ADMITTED_ROUTING_STATE -> mode (i)   UNTRIAGED
  M2  drop a whole-object exclusion rule               -> mode (i)   UNTRIAGED
  M3  declare a CONTRIBUTING unit pre-filter-excluded  -> mode (iv)  PREFILTER-VIOLATION
  M4  a pre-filter-excluded unit GAINS a static        -> mode (iv)  PREFILTER-VIOLATION
  M5  upstream adds a routing field nothing triages    -> mode (iii) UNTRIAGED
  M6  reorder the declared prologue                    -> mode (ii)  PROLOGUE-ORDER-VIOLATION
  M7  revert struct_fields to the last-identifier form -> the BLIND check:
                                                          Conduit.a1 silently
                                                          leaves the artifact
  M8  revert split_functions to reject a trailing line
      comment                                          -> the walk truncates and
                                                          inflow.c / rdii.c drop
                                                          out of the closure

M7 and M8 are the two that a diff-only check cannot catch by itself, because the
regeneration compares two outputs of the SAME helper: if the helper stops
seeing a field, neither side enumerates it and the diff is empty.  They are
asserted against the COMMITTED artifact rather than against a re-run, which is
the only comparison that survives a defective instrument.

USAGE
-----
    test_snapshot_inventory_closure_modes.py <repo-root>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = "regenerate_snapshot_inventory.py"
INVENTORY = "snapshot_inventory.txt"


def stage(repo: Path) -> Path:
    """Copy the solver tree and the inventory tooling into a throwaway root."""
    tmp = Path(tempfile.mkdtemp(prefix="snapinv-"))
    shutil.copytree(repo / "external" / "swmm" / "src" / "solver", tmp / "solver")
    (tmp / "inventory").mkdir()
    for name in (SCRIPT, INVENTORY):
        shutil.copy2(repo / "test" / "inventory" / name, tmp / "inventory" / name)
    return tmp


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise AssertionError("mutation anchor not found in %s: %r" % (path.name, old[:70]))
    path.write_text(text.replace(old, new, 1))


def run_check(tmp: Path):
    proc = subprocess.run(
        [sys.executable, str(tmp / "inventory" / SCRIPT),
         "--solver-dir", str(tmp / "solver"),
         "--inventory", str(tmp / "inventory" / INVENTORY),
         "--check"],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def m1(tmp: Path) -> None:
    edit(tmp / "inventory" / SCRIPT,
         '    ("Node", "overflow"):',
         '    ("Node", "__dropped_by_M1__"):')


def m2(tmp: Path) -> None:
    edit(tmp / "inventory" / SCRIPT, '    "Weir":', '    "__dropped_by_M2__":')


def m3(tmp: Path) -> None:
    edit(tmp / "inventory" / SCRIPT,
         'PREFILTER_EXCLUDED_UNITS = [\n    "findroot.c",',
         'PREFILTER_EXCLUDED_UNITS = [\n    "massbal.c",\n    "findroot.c",')


def m4(tmp: Path) -> None:
    src = tmp / "solver" / "xsect.c"
    src.write_text(src.read_text().replace(
        "#include \"headers.h\"",
        "#include \"headers.h\"\nstatic double M4_cross_step_memo;", 1))


def m5(tmp: Path) -> None:
    objects = tmp / "solver" / "objects.h"
    edit(objects, "   double        newLatFlow;",
         "   double        newLatFlow;\n   double        m5UpstreamAddition;")
    routing = tmp / "solver" / "routing.c"
    edit(routing, "        Node[j].oldLatFlow  = Node[j].newLatFlow;",
         "        Node[j].oldLatFlow  = Node[j].newLatFlow + Node[j].m5UpstreamAddition;")


def m6(tmp: Path) -> None:
    edit(tmp / "inventory" / SCRIPT,
         '    "initSystemInflows",\n    "link_setOldHydState",',
         '    "link_setOldHydState",\n    "initSystemInflows",')


def m7(tmp: Path) -> None:
    edit(tmp / "inventory" / SCRIPT,
         """        parts = decl.split(",")
        first = parts[0].split("[")[0]
        toks = _IDENT.findall(first)
        if len(toks) < 2:
            continue
        fields.append(toks[-1])
        for extra in parts[1:]:
            extra_toks = _IDENT.findall(extra.split("[")[0])
            if extra_toks:
                fields.append(extra_toks[-1])""",
         """        toks = _IDENT.findall(decl.split("[")[0])
        if len(toks) < 2:
            continue
        fields.append(toks[-1])""")


def m8(tmp: Path) -> None:
    edit(tmp / "inventory" / SCRIPT,
         r'\)\s*(?://.*)?$",', r'\)\s*$",')


# name -> (mutation, marker the output must carry, or None for "blind-check" form)
MUTATIONS = {
    "M1 drop a triaged field from ADMITTED_ROUTING_STATE": (m1, "UNTRIAGED"),
    "M2 drop a whole-object exclusion rule": (m2, "UNTRIAGED"),
    "M3 declare a CONTRIBUTING unit pre-filter-excluded": (m3, "PREFILTER-VIOLATION"),
    "M4 a pre-filter-excluded unit gains a static": (m4, "PREFILTER-VIOLATION"),
    "M5 upstream adds an untriaged routing field": (m5, "UNTRIAGED"),
    "M6 reorder the declared prologue": (m6, "PROLOGUE-ORDER-VIOLATION"),
}

# These two defeat a diff-only check, so they are asserted by absence from the
# regenerated artifact rather than by a marker in the check's output.
BLIND = {
    "M7 struct_fields reverts to the last-identifier form": (m7, "Conduit\ta1\t"),
    # Node.overflow SURVIVES this defect -- node.c is still reached through
    # routeFlow -- so the sentinel is a row reachable ONLY through the truncated
    # subtree.  Measured at this pin the defect silently loses 30 rows,
    # including every Subcatch entry and the KILLED rows for Node.oldLatFlow,
    # Node.oldVolume, Conduit.a2/q1Old/q2Old and Xnode.converged, while the
    # Criterion-R half stays at 172 pairs and the run exits 0.
    "M8 split_functions rejects a trailing line comment": (m8, "Subcatch\tnewRunoff\t"),
}


def main(argv) -> int:
    if len(argv) < 2:
        print("usage: %s <repo-root>" % argv[0], file=sys.stderr)
        return 2
    repo = Path(argv[1]).resolve()
    failures: list[str] = []

    for name, (mutate, marker) in MUTATIONS.items():
        tmp = stage(repo)
        try:
            mutate(tmp)
            rc, out = run_check(tmp)
            if rc == 0:
                failures.append("%s: the check stayed GREEN." % name)
                print("FAIL  %-56s stayed green" % name)
            elif marker not in out:
                failures.append(
                    "%s: the check went red but never printed %r, so it fired for "
                    "the wrong reason." % (name, marker))
                print("FAIL  %-56s red, but no %s" % (name, marker))
            else:
                print("ok    %-56s red, named %s" % (name, marker))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    for name, (mutate, row) in BLIND.items():
        tmp = stage(repo)
        try:
            mutate(tmp)
            proc = subprocess.run(
                [sys.executable, str(tmp / "inventory" / SCRIPT),
                 "--solver-dir", str(tmp / "solver"),
                 "--inventory", str(tmp / "inventory" / "regenerated.txt")],
                capture_output=True, text=True,
            )
            regenerated = (tmp / "inventory" / "regenerated.txt")
            text = regenerated.read_text() if regenerated.exists() else ""
            if proc.returncode != 0 and not text:
                print("ok    %-56s the defect aborts regeneration outright" % name)
                continue
            if row in text:
                failures.append(
                    "%s: %r survived the mutation, so this subtest no longer "
                    "distinguishes the repaired helper from the defective one."
                    % (name, row))
                print("FAIL  %-56s %r survived" % (name, row.strip()))
            else:
                print("ok    %-56s %r disappeared, as the defect predicts"
                      % (name, row.strip()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: all %d defect forms are caught."
          % (len(MUTATIONS) + len(BLIND)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

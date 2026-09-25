#!/usr/bin/env python3
"""Guard-of-the-guard for Criterion P's closure check (WP-1B chunk 9).

WHY THIS EXISTS
---------------
``regenerate_snapshot_inventory.py --check`` returns 0 only when the inventory
is BOTH current against the vendored tree AND carries no open triage decision
-- so a green run is a claim about the check as much as about the tree.  A
check whose pre-filter verified itself, whose triage tables swallowed every
name, or whose parser truncated the walk before the interesting units would
return 0 just as readily, and none of that is visible from a green run.  §2.1(e) asks a closure check to
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
  M9  re-order the scalar basis back below validation  -> every scalar row
                                                          silently leaves the
                                                          artifact

  G1  an artifact that is CURRENT but carries an open  -> EXIT_UNTRIAGED, and
      decision                                            NOT the drift status
  G2  an artifact that is current AND fully triaged    -> EXIT_OK
  G3  an artifact that has DRIFTED, decisions closed   -> EXIT_DRIFT, and NOT
                                                          the untriaged status

G1-G3 ARE THE GATE'S OWN DIFFERENTIAL, and they belong here for the reason the
whole file exists.  Before they were written, `--check` returned 0 on a tree
carrying 140 open decisions: the rows had been correctly surfaced and correctly
committed to the artifact, which emptied the diff, and a diff-only check has
nothing left to say.  A specified gate was disarmed as a side effect of doing
the right thing with the rows.  G1 is the assertion that would have caught it,
and it FAILS against the pre-arming script.  G2 and G3 are the other two
corners, because a gate that fires on every input is as useless as one that
fires on none, and because the two red reasons must stay distinguishable to a
maintainer who has to know WHICH problem the tree has.

  G1  an artifact that is CURRENT but carries an open  -> EXIT_UNTRIAGED, and
      decision                                            NOT the drift status
  G2  an artifact that is current AND fully triaged    -> EXIT_OK
  G3  an artifact that has DRIFTED, decisions all      -> EXIT_DRIFT, and NOT
      closed                                              the untriaged status

G1-G3 are the GATE's own differential and they belong here for the reason the
whole file exists.  Before they were written, `--check` returned 0 on a tree
carrying 140 open decisions: the rows had been correctly surfaced and correctly
committed to the artifact, which emptied the diff, and a diff-only check has
nothing left to say.  A specified gate was disarmed as a side effect of doing
the right thing with the rows.  G1 is the assertion that would have caught it
and it FAILS against the pre-arming script; G2 and G3 are the other two corners,
because a gate that fires on every input is as useless as one that fires on
none, and because the two red reasons must stay distinguishable to a maintainer
who has to know WHICH problem the tree has.

M9 IS THE FIRST SCALAR-AXIS MODE IN THIS FILE, and its absence is why the defect
it catches survived. M1-M8 are all field-axis: they perturb a struct field, a
struct-typed object, or a helper that parses struct declarations. While the
scalar half of Criterion P emitted no rows, a scalar-axis mutation had nothing
to remove and was unwritable -- so this guard-of-the-guard could not have gone
red on a basis that produced nothing, and its green was silent about it. The
lesson generalises: a mutation suite can only perturb what its subject already
emits, so "all modes pass" never establishes that the subject's coverage is
complete.

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


# The SCALAR-axis mutation. Every mode above it is field-axis -- it perturbs a
# struct field, a struct-typed object, or a helper that parses struct
# declarations -- so while the scalar basis produced no rows a scalar-axis
# mutation was literally unwritable: there was nothing for it to remove. That is
# why the basis being dead code went unnoticed by this file for its whole life.
#
# M9 reverts the ORDERING, which is the defect as it actually shipped: build
# `stream` and add the ("-", name) scalars to `candidates` AFTER the validation
# loop has already consumed `candidates`. The kill pass takes `validated`, not
# `candidates`, so under that ordering no scalar ever reaches it, the
# `if obj == "-"` branch inside the validation loop is unreachable by
# construction, and the whole second Criterion-P basis is dead. The two
# statements are byte-identical in both positions -- only where they sit
# changes, which is exactly why the defect was invisible to a reader scanning
# for a wrong expression.
#
# BLIND form rather than marker form, deliberately. A marker-form assertion
# ("--check went red and said UNTRIAGED") would be satisfied here for a trivial
# reason -- 140 rows vanish from the diff, so the check reddens no matter what
# the scalar basis is doing. Asserting that one specific scalar row is ABSENT
# from the REGENERATED artifact is what ties this subtest to the basis rather
# than to the diff.
_HOISTED_SCALAR_BLOCK = """    stream = step_event_stream(defs, reached, objects, scalars)
    for _kind, obj, field, _fn in stream:
        if obj == "-":
            candidates.add((obj, field))

"""
_KILL_PASS_LINE = "    live, killed = kill_pass(validated, stream)"


def m9(tmp: Path) -> None:
    script = tmp / "inventory" / SCRIPT
    # 1. lift the hoisted block out of its correct position
    edit(script, _HOISTED_SCALAR_BLOCK, "")
    # 2. put it back where it used to be -- below the validation loop, too late
    edit(script, _KILL_PASS_LINE, _HOISTED_SCALAR_BLOCK + _KILL_PASS_LINE)


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
    # ReportTime is the sentinel because Sec 4.6.3 names it as one of the three
    # fields (with BetweenEvents and VariableStep) that widening the boundary to
    # swmm_step is SUPPOSED to make visible, and it is referenced nowhere in the
    # four-root routing closure -- so it is reachable through the scalar basis
    # and through nothing else. A struct-field sentinel could not distinguish
    # this defect from a dozen others.
    "M9 the scalar basis is re-ordered back into dead code": (m9, "-\tReportTime\t"),
}


# =============================================================================
# THE GATE'S OWN DIFFERENTIAL (G1-G3)
# =============================================================================
#
# Every mode above perturbs the OPERATION and asks whether the check notices.
# These three perturb the TREE'S TRIAGE STATE and ask whether the check's
# VERDICT is the right one -- a different question, and the one that went
# unasked while `--check` reported a 140-open-decision tree as clean.
#
# Each stages the tree, REGENERATES the artifact in place so the drift axis is
# quiet by construction, and only then asks what `--check` says.  Regenerating
# first is what makes these differential rather than incidental: without it a
# red would be explained by drift and would establish nothing about the gate.

# The END of EXCLUDED_ROUTING_STATE, anchored on the table's TERMINATOR and the
# comment that follows it rather than on whichever entry happens to be last.
# The entry-anchored form broke the moment the table grew, which is the right
# failure -- it raised, naming the missing anchor -- but it is a failure this
# subtest will meet again on every future triage round, so the anchor moves to
# something growth-invariant.
_EXCLUDED_TABLE_END = (
    "}\n"
    "\n"
    "# Whole-object exclusion RULES."
)


def _exit_codes(tmp: Path) -> tuple[int, int, int]:
    """``(EXIT_OK, EXIT_DRIFT, EXIT_UNTRIAGED)`` read from the STAGED script.

    Read from the subject rather than hard-coded here, so the two cannot drift
    apart into a test that asserts a status the script no longer returns.
    """
    import importlib.util
    path = tmp / "inventory" / SCRIPT
    spec = importlib.util.spec_from_file_location("staged_snapshot_inventory", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXIT_OK, mod.EXIT_DRIFT, mod.EXIT_UNTRIAGED


def regenerate(tmp: Path) -> str:
    """Rewrite the staged artifact from the staged script; return its text."""
    proc = subprocess.run(
        [sys.executable, str(tmp / "inventory" / SCRIPT),
         "--solver-dir", str(tmp / "solver"),
         "--inventory", str(tmp / "inventory" / INVENTORY)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError("staged regeneration failed: %s" % (proc.stderr[-400:],))
    return (tmp / "inventory" / INVENTORY).read_text()


def close_every_open_decision(tmp: Path) -> int:
    """Triage every currently-open Criterion-P row EXCLUDED; return how many.

    The names are taken from the staged artifact itself rather than from a list
    here, so this stays correct as the real triage closes: when nothing is open
    it injects nothing and G2 still asserts the clean verdict.  The disposition
    is a throwaway -- G2 is about the gate's arithmetic, not about whether any
    of these belong in the snapshot, and inventing a real verdict here would be
    the over-capture direction this design treats as the worse one.
    """
    text = regenerate(tmp)
    rows = [ln.split("\t") for ln in text.splitlines() if "\tUNTRIAGED\t" in ln]
    keys = [(r[0], r[1]) for r in rows if len(r) >= 2]
    if not keys:
        return 0
    injected = "".join(
        '    (%r, %r):\n        "test-only disposition injected by G2",\n' % (obj, field)
        for obj, field in keys
    )
    edit(tmp / "inventory" / SCRIPT,
         _EXCLUDED_TABLE_END,
         injected + _EXCLUDED_TABLE_END)
    return len(keys)


def g1_current_but_open(tmp: Path) -> tuple[bool, str]:
    """An artifact that is CURRENT and carries an open decision must go RED."""
    ok_, drift, untriaged = _exit_codes(tmp)
    # Open ONE decision deliberately rather than relying on the 140 that happen
    # to be open today: this subtest must keep discriminating after the real
    # triage closes, and an assertion resting on today's backlog would invert
    # into a wrong answer on the day the backlog is cleared.
    edit(tmp / "inventory" / SCRIPT,
         '    ("Conduit", "barrels"):',
         '    ("Conduit", "__opened_by_G1__"):')
    regenerate(tmp)                   # artifact now CURRENT; the drift axis is quiet
    rc, out = run_check(tmp)
    if rc == ok_:
        return False, "the check stayed GREEN on a tree with an open decision"
    if rc == drift:
        return False, ("reported DRIFT (%d) on a freshly regenerated artifact, so "
                       "the two red reasons are not separable" % rc)
    if rc != untriaged:
        return False, "expected EXIT_UNTRIAGED (%d), got %d" % (untriaged, rc)
    if "open triage decision" not in out:
        return False, "red with the right status but no open-decision message"
    return True, "EXIT_UNTRIAGED (%d), named the open decision" % rc


def g2_current_and_closed(tmp: Path) -> tuple[bool, str]:
    """Current AND fully triaged is the ONE state that may report clean."""
    ok_, _drift, _untriaged = _exit_codes(tmp)
    n = close_every_open_decision(tmp)
    regenerate(tmp)
    rc, out = run_check(tmp)
    if rc != ok_:
        return False, ("went red (%d) on a current, fully-triaged artifact, so the "
                       "gate fires on more than its invariant's violations: %s"
                       % (rc, out.strip().splitlines()[:1]))
    return True, "EXIT_OK after closing %d open decision(s)" % n


def g3_drifted(tmp: Path) -> tuple[bool, str]:
    """Drift must still report DRIFT, and must not be masked by the new axis."""
    _ok, drift, untriaged = _exit_codes(tmp)
    close_every_open_decision(tmp)
    regenerate(tmp)
    inv = tmp / "inventory" / INVENTORY
    # Drop a data row: the cheapest perturbation that is unambiguously drift.
    kept = [ln for ln in inv.read_text().splitlines()
            if not ln.startswith("Conduit\tq1\t")]
    inv.write_text("\n".join(kept) + "\n")
    rc, out = run_check(tmp)
    if rc == untriaged:
        return False, "reported open decisions (%d) for what is drift" % rc
    if rc != drift:
        return False, "expected EXIT_DRIFT (%d), got %d" % (drift, rc)
    if "differs from the committed one" not in out:
        return False, "red with the right status but no drift message"
    return True, "EXIT_DRIFT (%d), named the drift" % rc


GATE = {
    "G1 current artifact, one open decision": g1_current_but_open,
    "G2 current artifact, every decision closed": g2_current_and_closed,
    "G3 drifted artifact, every decision closed": g3_drifted,
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

    for name, probe in GATE.items():
        tmp = stage(repo)
        try:
            passed, detail = probe(tmp)
            if passed:
                print("ok    %-56s %s" % (name, detail))
            else:
                failures.append("%s: %s" % (name, detail))
                print("FAIL  %-56s %s" % (name, detail))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: all %d defect forms are caught."
          % (len(MUTATIONS) + len(BLIND) + len(GATE)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Criterion-P assertions over the snapshot field inventory (WP-1B chunk 9).

WHY THIS EXISTS
---------------
``WP-1B``'s acceptance gate failed because the snapshot captured report
accumulators and NO live routing state: §4.6.1 carried one criterion and it was
the report criterion, and the implementation followed the criterion it was
given.  Criterion P is the routing criterion, and this file holds the
assertions that would have caught the gap.

Every subtest below pins a property whose VIOLATION is a wrong answer that
looks like a right one -- a classification that is confidently emitted, reads
plausibly in the artifact, and is wrong:

  P1  the ORDER-SENSITIVITY regression.  A FLAT kill pass -- one that subtracts
      a field whenever the prologue writes it, without asking whether a read
      came first -- reports Node.inflow, Node.outflow and Node.newLatFlow
      KILLED.  All three are LIVE.  This subtest runs BOTH passes and asserts
      they disagree, because asserting only that the real pass says LIVE would
      also pass against a pass that says LIVE about everything.
  P2  Node.overflow is LIVE and ADMITTED.  It is read at massbal.c:635 by the
      OPENING massbal_updateRoutingTotals, before the step prologue runs, and
      massbal.c was not among the nine translation units the retired bound
      named -- so the retired bound killed a provably-live field.
  P3  the two fields that refuse a hand-written list are present: Conduit.a1
      (dropped by struct_fields until this chunk repaired it) and
      Xnode.oldSurfArea (in NO struct body in objects.h at all).
  P4  the pre-filter DISCRIMINATES, and massbal.c is admitted by the GLOBALS
      clause rather than the statics clause.  Its only two file-scope statics
      are `static const`, so a statics-only pre-filter would exclude it and kill
      Node.overflow a second time.
  P5  the walk is UNRESTRICTED: it reaches the first-hop units the retired
      nine-unit list omitted.
  P6  the declared prologue order is the order the source executes.
  P7  the D-R6 TFile enumeration is TYPE-keyed and returns both declaration
      sites, admitted set empty, nothing untriaged.
  P8  struct_fields returns all twenty TConduit fields (the multi-declarator
      repair) and split_functions recovers a definition whose header carries a
      trailing line comment (the closure-truncation repair).
  P9  regeneration is idempotent at the pin.

USAGE
-----
    test_snapshot_inventory_criterion_p.py <repo-root>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_script(repo: Path):
    path = repo / "test" / "inventory" / "regenerate_snapshot_inventory.py"
    spec = importlib.util.spec_from_file_location("snapshot_inventory", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat_kill_pass(R, candidates, stream):
    """The UNSOUND pass this file exists to keep out of the tree.

    Subtracts a field whenever some prologue function performs a pure write to
    it, without asking whether a read came first.  Reproduced here rather than
    described, so P1 compares two behaviours instead of asserting one.
    """
    written = {
        (obj, field)
        for kind, obj, field, fn in stream
        if kind == "w" and fn in R.ROUTING_PROLOGUE
    }
    killed = {k for k in candidates if k in written}
    return sorted(set(candidates) - killed), sorted(killed)


def main(argv) -> int:
    if len(argv) < 2:
        print("usage: %s <repo-root>" % argv[0], file=sys.stderr)
        return 2
    repo = Path(argv[1]).resolve()
    solver = repo / "external" / "swmm" / "src" / "solver"
    R = load_script(repo)

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print("ok    %s" % name)
        else:
            failures.append("%s: %s" % (name, detail))
            print("FAIL  %-58s %s" % (name, detail))

    globals_h = (solver / "globals.h").read_text(errors="replace")
    objects_h = (solver / "objects.h").read_text(errors="replace")

    defs = R.all_solver_functions(solver)
    reached = R.routing_closure(defs, R.ROUTING_SEED_ROOTS)
    reached_units = sorted({u for u, _ in reached})

    # --- rebuild the candidate pool the emitter builds -----------------------
    ext_globals = R.extern_global_names(globals_h)
    contributing, unit_statics = [], {}
    for unit in reached_units:
        src = (solver / unit).read_text(errors="replace")
        statics, file_globals = R.file_scope_mutable_state(src)
        unit_statics[unit] = statics
        if statics or file_globals:
            contributing.append(unit)
            continue
        for name in sorted(ext_globals):
            import re as _re
            if _re.search(
                rf"\b{_re.escape(name)}\b\s*(?:\[[^\]]*\]\s*)?"
                rf"(?:(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*\s*)?"
                rf"(?:\[[^\]]*\]\s*)?(?:[-+*/%|&^]|<<|>>)?=(?!=)",
                src,
            ):
                contributing.append(unit)
                break

    objects = set(R._struct_typed_globals(globals_h))
    scalars = set(R._scalar_globals(globals_h))
    for unit in contributing:
        objects.update(unit_statics[unit])
        scalars.update(unit_statics[unit])
    objects, scalars = frozenset(objects), frozenset(scalars)

    bodies = {(u, n): b for n, v in defs.items() for u, b in v}
    candidates = set()
    for unit, fn in sorted(reached):
        if unit not in contributing:
            continue
        for _off, _kind, obj, field in R.state_events(bodies[(unit, fn)], objects, frozenset()):
            candidates.add((obj, field))

    stream = R.step_event_stream(defs, reached, objects, scalars)
    for _kind, obj, field, _fn in stream:
        if obj == "-":
            candidates.add((obj, field))

    live, killed = R.kill_pass(candidates, stream)
    flat_live, flat_killed = flat_kill_pass(R, candidates, stream)
    live_set, killed_set = set(live), set(killed)

    # --- P1: the order-sensitivity regression --------------------------------
    ordered_wrong, flat_agreed = [], []
    for field in ("inflow", "outflow", "newLatFlow"):
        key = ("Node", field)
        if key not in live_set:
            ordered_wrong.append("Node.%s" % field)
        if key not in set(flat_killed):
            flat_agreed.append("Node.%s" % field)
    check(
        "P1a ordered kill pass classifies the three named fields LIVE",
        not ordered_wrong,
        "not LIVE: %s" % ", ".join(ordered_wrong),
    )
    check(
        "P1b a FLAT kill pass classifies all three KILLED (so P1a is not vacuous)",
        not flat_agreed,
        "the flat pass did NOT kill %s, so this tree no longer distinguishes the "
        "two passes and P1a proves nothing" % ", ".join(flat_agreed),
    )

    # --- P2: the field the retired bound killed ------------------------------
    check(
        "P2a Node.overflow is LIVE",
        ("Node", "overflow") in live_set,
        "classified %s" % ("KILLED" if ("Node", "overflow") in killed_set else "absent"),
    )
    check(
        "P2b Node.overflow is ADMITTED, not excluded",
        ("Node", "overflow") in R.ADMITTED_ROUTING_STATE,
        "absent from ADMITTED_ROUTING_STATE",
    )
    check(
        "P2c Node.newVolume is ADMITTED",
        ("Node", "newVolume") in R.ADMITTED_ROUTING_STATE,
        "absent from ADMITTED_ROUTING_STATE",
    )

    # --- P3: the fields that refuse a hand-written list ----------------------
    for obj, field in (("Conduit", "a1"), ("Xnode", "oldSurfArea")):
        check(
            "P3  %s.%s is LIVE and ADMITTED" % (obj, field),
            (obj, field) in live_set and (obj, field) in R.ADMITTED_ROUTING_STATE,
            "live=%s admitted=%s"
            % ((obj, field) in live_set, (obj, field) in R.ADMITTED_ROUTING_STATE),
        )

    # --- P3b: the four cross-step CLOCKS on the scalar axis -------------------
    #
    # These are the scalar-axis counterpart of P3, and they exist because the
    # scalar basis shipped as dead code for its whole life: until it was made
    # reachable, no scalar row existed for a subtest to assert on at all.  Each
    # of the four is read by the step before the step writes it, and each is
    # RESET by the start path, which is the pair of facts that makes a resume
    # wrong without it.  A read-modify-write counts as a read here (the kill
    # pass's own rule), which is why three of the four are live despite being
    # "assigned" every step -- their only PURE write is the start-path reset.
    #
    # The reset site is asserted alongside the verdict, because the verdict
    # without it is just a claim that something is live; the reset is what
    # makes it CONSEQUENTIAL on a resume.
    for name, reset_unit, reset_needle in (
        ("NewRoutingTime", "swmm5.c",   "NewRoutingTime = 0.0"),
        ("ReportTime",     "swmm5.c",   "ReportTime = 1000 * (double)ReportStep"),
        ("NewRuleTime",    "routing.c", "NewRuleTime = 0.0"),
        ("NextEvent",      "routing.c", "NextEvent = 0"),
    ):
        key = ("-", name)
        check(
            "P3b %s is LIVE and ADMITTED on the scalar axis" % name,
            key in live_set and key in R.ADMITTED_ROUTING_STATE,
            "live=%s admitted=%s"
            % (key in live_set, key in R.ADMITTED_ROUTING_STATE),
        )
        unit_src = (solver / reset_unit).read_text(errors="replace")
        check(
            "P3c %s is reset by the start path (%s), which is what a resume "
            "would otherwise restart from" % (name, reset_unit),
            reset_needle in unit_src,
            "%r not found in %s -- either the reset moved or this assertion is "
            "naming the wrong site" % (reset_needle, reset_unit),
        )

    # --- P4: the pre-filter discriminates ------------------------------------
    check(
        "P4a xsect.c is pre-filter EXCLUDED (the largest contributor, no cross-step state)",
        "xsect.c" not in contributing,
        "xsect.c contributes, so the pre-filter no longer discriminates",
    )
    check(
        "P4b massbal.c CONTRIBUTES",
        "massbal.c" in contributing,
        "massbal.c is excluded -- Node.overflow would be killed a second time",
    )
    mb_statics, mb_globals = R.file_scope_mutable_state(
        (solver / "massbal.c").read_text(errors="replace")
    )
    check(
        "P4c massbal.c is admitted by the GLOBALS clause, not the statics clause",
        not mb_statics and bool(mb_globals),
        "statics=%s globals=%s -- if the statics clause now fires here, the "
        "const-blind spot this subtest pins has moved" % (mb_statics, mb_globals),
    )
    check(
        "P4d the declared pre-filter exclusions all hold at this pin",
        all(u not in contributing for u in R.PREFILTER_EXCLUDED_UNITS),
        "contributing but declared excluded: %s"
        % [u for u in R.PREFILTER_EXCLUDED_UNITS if u in contributing],
    )

    # --- P5: the walk is unrestricted ----------------------------------------
    first_hop = ["massbal.c", "stats.c", "inflow.c", "table.c", "rdii.c"]
    missing = [u for u in first_hop if u not in reached_units]
    check(
        "P5  the walk reaches the first-hop units the retired nine-unit list omitted",
        not missing,
        "unreached: %s" % ", ".join(missing),
    )

    # --- P6: the declared prologue order is the executed order ---------------
    seen = R.prologue_order_holds(stream)
    check(
        "P6  the prologue executes in the declared order",
        seen == [f for f in R.ROUTING_PROLOGUE if f in seen] and len(seen) == len(R.ROUTING_PROLOGUE),
        "declared %s, reached %s" % (R.ROUTING_PROLOGUE, seen),
    )

    # --- P7: D-R6 ------------------------------------------------------------
    handles = R._tfile_handles(globals_h, objects_h)
    names = [n for n, _ in handles]
    sites = {s for _, s in handles}
    check("P7a the TFile enumeration returns 12 handles", len(handles) == 12,
          "returned %d: %s" % (len(handles), names))
    check("P7b it is keyed on the TYPE, so BOTH declaration sites appear",
          len(sites) == 2, "sites=%s" % sorted(sites))
    check("P7c the per-time-series cursor is enumerated",
          "TTable.file" in names,
          "TTable.file absent -- the enumeration was keyed on a PLACE")
    check("P7d every handle is triaged", 
          all(n in R.TFILE_ADMITTED or n in R.TFILE_EXCLUDED for n in names),
          "untriaged: %s" % [n for n in names
                             if n not in R.TFILE_ADMITTED and n not in R.TFILE_EXCLUDED])
    check("P7e the admitted set is EMPTY (a passing result)",
          not R.TFILE_ADMITTED, "admitted=%s" % sorted(R.TFILE_ADMITTED))

    # --- P8: the two helper repairs ------------------------------------------
    check("P8a struct_fields returns all 20 TConduit fields (multi-declarator repair)",
          len(R.struct_fields(objects_h, "TConduit")) == 20,
          "returned %d" % len(R.struct_fields(objects_h, "TConduit")))
    routing_c = (solver / "routing.c").read_text(errors="replace")
    check("P8b split_functions recovers a definition with a trailing line comment",
          "addSystemInflows" in R.split_functions(routing_c),
          "addSystemInflows unrecovered -- the closure truncates at routing.c and "
          "inflow.c / rdii.c drop out of the walk entirely")

    # --- P9: regeneration is idempotent --------------------------------------
    import subprocess
    rc = subprocess.run(
        [sys.executable, str(repo / "test" / "inventory" / "regenerate_snapshot_inventory.py"),
         "--solver-dir", str(solver), "--check"],
        capture_output=True, text=True,
    )
    # P9 asks about DRIFT and nothing else, so it is keyed on the drift status
    # rather than on "exit 0".  Since the untriaged gate was armed, `--check`
    # also reddens (EXIT_UNTRIAGED) while open triage decisions remain, and a
    # bare `== 0` here would have turned P9 red for a condition P9 does not
    # claim to measure -- reporting the vendored tree as drifted when it has
    # not moved at all.
    check("P9  the committed inventory matches a regeneration at this pin",
          rc.returncode != R.EXIT_DRIFT, rc.stderr.strip().splitlines()[:1])

    # --- P10: the two sections' SCALAR ROW SHAPES, which T6's key-builder
    #          depends on and cannot assert about itself -----------------------
    #
    # T6 (test/snapshot/test_state_snapshot.cpp) turns an inventory row into the
    # key `object.field`, and a SCALAR row carries a '-' sentinel in whichever
    # column its section does not use.  THE TWO SECTIONS SPELL IT IN OPPOSITE
    # ORDERS -- Criterion R writes `name<TAB>-` and Criterion P writes
    # `-<TAB>name` -- so T6 needs a section-aware branch to land both on
    # `Name.value`.  It carries one.
    #
    # P10 asserts the SHAPES that branch was written against.  It lives here and
    # not there because T6 is COMPILE-BEARING: it cannot run without building
    # the coupled test binary.  If these shapes ever converge or swap, the C++
    # branch becomes wrong SILENTLY -- a P scalar row would normalize to a key
    # the serializer never emits, and T6's over-capture half would go vacuous
    # again exactly as it was before the branch landed.  That failure mode is
    # green-looking, which is why it wants a check that runs cheaply.
    #
    # P10 IS NOT A SUBSTITUTE FOR T6 and asserts nothing T6 asserts.  T6
    # compares the serializer's RUN-TIME manifest against the inventory; this
    # compares two column orders in a text file.
    inv_text = (repo / "test" / "inventory" / "snapshot_inventory.txt").read_text()
    r_scalar_rows, p_scalar_rows, sec_p = 0, 0, False
    for line in inv_text.splitlines():
        if line.startswith("#"):
            if not sec_p and "CRITERION P" in line:
                sec_p = True
            elif sec_p and "D-R6 --" in line:
                break
            continue
        col = line.split("\t")
        if len(col) < 2:
            continue
        if sec_p:
            if col[0] == "-":
                p_scalar_rows += 1
        elif col[1] == "-":
            r_scalar_rows += 1
    check("P10a Criterion-R scalar rows carry the sentinel in the FIELD column",
          r_scalar_rows > 0,
          "found %d -- either the R scalar block is gone or its column order "
          "moved, and T6's `if (fld == \"-\")` branch is now wrong"
          % r_scalar_rows)
    check("P10b Criterion-P scalar rows carry the sentinel in the OBJECT column",
          p_scalar_rows > 0,
          "found %d -- either the P scalar basis is dead again or its column "
          "order moved, and T6's `if (obj == \"-\")` branch is now wrong"
          % p_scalar_rows)

    # --- P11: the serializer carries every ADMITTED scalar and no refused one -
    #
    # The scalar-axis half of what T6 asserts, at TEXT level against
    # snapshot.c's `snap_name(c, "Name", "value")` calls.  Again NOT a
    # substitute: T6 builds the manifest by RUNNING snapshot_traverse, so it
    # catches a name spelled in a call the traversal never reaches, and it
    # covers the struct axis too.  What P11 adds is coverage of the one failure
    # mode a compile-bearing test cannot reach on an un-built tree -- an
    # ADMITTED scalar that nobody added to the serializer at all, which is
    # precisely the state the four routing clocks were in when it was written.
    import re as _re11
    snap_c = (solver / "snapshot.c").read_text(errors="replace")
    emitted_scalars = set(
        _re11.findall(
            r'snap_name\(\s*c\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*"value"\s*\)',
            snap_c,
        )
    )
    check("P11a the snap_name('value') scan found a non-empty set",
          bool(emitted_scalars),
          "zero matches -- the scan regex no longer matches the emit form, so "
          "P11c and P11d below would pass vacuously")
    admitted_scalars = sorted(f for (o, f) in R.ADMITTED_ROUTING_STATE if o == "-")
    check("P11b the admitted-scalar set is non-empty",
          bool(admitted_scalars),
          "zero ADMITTED scalar rows -- P11c would pass vacuously")
    missing_scalars = [n for n in admitted_scalars if n not in emitted_scalars]
    check("P11c every ADMITTED Criterion-P scalar is emitted by the serializer",
          not missing_scalars,
          "absent from snapshot.c: %s -- T6 will report these as MISSING FROM "
          "MANIFEST" % missing_scalars)
    # SERIALIZED_SCALARS is a list of (name, ctype, unit, linkage) TUPLES, not
    # of names.  Membership-testing a bare name against it returns False for
    # every name, which would make this check fire on the four Criterion-R
    # scalars that the P section also refuses and that ARE correctly serialized
    # -- a confident, plausible, wrong finding.  Project the first column.
    r_serialized = {t[0] for t in R.SERIALIZED_SCALARS}
    overcaptured_scalars = sorted(
        f for (o, f) in R.EXCLUDED_ROUTING_STATE
        if o == "-" and f in emitted_scalars
        and ("-", f) not in R.ADMITTED_ROUTING_STATE
        and f not in r_serialized
    )
    check("P11d no EXCLUDED Criterion-P scalar is emitted by the serializer",
          not overcaptured_scalars,
          "serialized despite being refused by BOTH criteria: %s -- "
          "over-capture of .inp configuration lets a stale snapshot override "
          "the model the operator is running" % overcaptured_scalars)

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: Criterion P holds on the tree as landed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

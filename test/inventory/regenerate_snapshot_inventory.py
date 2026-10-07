#!/usr/bin/env python3
"""Regenerate the SWMM state-snapshot field inventory from EPA SWMM source.

This script performs the OPERATION that defines the snapshot's contents:

    A quantity is IN the snapshot iff some report-writing path reads it.

It is deliberately an OPERATION over source rather than a hand-maintained list.
A hand-maintained list is what produces the documented failure: ``TNodeStats``
declares sixteen fields and ``statsrpt.c`` references fourteen; the two it never
touches -- ``timeCourantCritical`` and ``nonConvergedCount`` -- reach the report
only through ``stats.c``'s own ``MaxCourantCrit`` / ``MaxNonConverged`` path.
Those two are precisely the not-reconstructible counter class that justifies
capturing the accumulators at all, so a list-driven inventory drops exactly the
fields the snapshot exists to carry.

Method
------
1. SEED on the report-emitting roots (``stats_report``, ``statsrpt_writeReport``,
   ``massbal_report``).
2. CLOSE TRANSITIVELY over the functions those roots call, restricted to the
   report-bearing translation units.  Closure is at FUNCTION granularity, not
   file granularity: a field referenced only by an update path that no report
   root reaches is not admitted by the criterion.
3. EMIT ``{object, field}`` pairs for every reached reference into a serialized
   object.

Field extents are taken from the STRUCT BODY in ``objects.h``, never from the
whole file.  A field count over the whole ``objects.h`` returns several hundred
and is internally consistent and silently wrong; the correct extent for
``TNodeStats`` returns 16.

Usage
-----
    python3 regenerate_snapshot_inventory.py --solver-dir <path> [--check]

Without ``--check`` the inventory is written to ``snapshot_inventory.txt``
beside this script.  With ``--check`` nothing is written and the exit status is
non-zero if the regenerated inventory differs from the committed one -- this is
the regeneration check, which fires as a slow-tier test and as a named step in
the upstream-bump procedure.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

# --- the objects the snapshot serializes -------------------------------------
#
# Each entry maps the C *variable* name a report path dereferences to the struct
# type whose body defines the legal field set.  Both spellings are needed: the
# closure scans for the variable, the extent comes from the type.
SERIALIZED_OBJECTS = {
    "NodeStats": "TNodeStats",
    "LinkStats": "TLinkStats",
    "StorageStats": "TStorageStats",
    "OutfallStats": "TOutfallStats",
    "PumpStats": "TPumpStats",
    "SubcatchStats": "TSubcatchStats",
    "TimeStepStats": "TTimeStepStats",
    "MaxMassBalErrs": "TMaxStats",
    "MaxCourantCrit": "TMaxStats",
    "MaxFlowTurns": "TMaxStats",
    "MaxNonConverged": "TMaxStats",
    # Mass-balance accumulators.  `massbal_report` reads all of these, so the
    # criterion admits them on exactly the same footing as the statistics
    # accumulators; option B2 is "capture and restore the statistics AND the
    # mass-balance accumulators", and omitting this half would leave the
    # continuity table covering only the post-resume segment.
    "RunoffTotals": "TRunoffTotals",
    "GwaterTotals": "TGwaterTotals",
    "FlowTotals": "TRoutingTotals",
    "LoadingTotals": "TLoadingTotals",
    "QualTotals": "TRoutingTotals",
    "StepFlowTotals": "TRoutingTotals",
    "OldStepFlowTotals": "TRoutingTotals",
}

# Scalar accumulators that are not struct members.  These are read by report
# paths but carry no struct body, so the extent check does not apply to them.
# `owner` is the translation unit that defines the storage -- it decides whether
# an accessor is needed (a `static` cannot be reached by `extern`).
SERIALIZED_SCALARS = [
    # (name, C type, owner, storage)
    ("MaxOutfallFlow", "double", "stats.c", "extern"),
    ("MaxRunoffFlow", "double", "stats.c", "extern"),
    ("RoutingTimeSpan", "double", "stats.c", "extern"),
    ("SysOutfallFlow", "double", "stats.c", "static"),
    ("TotalArea", "double", "massbal.c", "extern"),
    ("NodeInflow", "double*", "massbal.c", "extern"),
    ("NodeOutflow", "double*", "massbal.c", "extern"),
    ("ReportStepCount", "long", "globals.h", "extern"),
    ("NonConvergeCount", "long", "globals.h", "extern"),
    ("TotalStepCount", "long", "globals.h", "extern"),
]

# Scalars the report path reads that are DELIBERATELY not snapshotted, each with
# the reason.  The discovery pass reports any globals.h scalar a report path
# reads that appears in neither this map nor SERIALIZED_SCALARS, so a name can
# only leave the report by being triaged -- never by being overlooked.
EXCLUDED_SCALARS = {
    # Configuration re-derived from the .inp at every swmm_open().  Snapshotting
    # these would let a stale snapshot silently override the .inp the operator
    # is actually running, which is worse than not restoring them at all.
    "CourantFactor": "config: re-read from the .inp at swmm_open",
    "FlowUnits": "config: re-read from the .inp at swmm_open",
    "UnitSystem": "config: re-read from the .inp at swmm_open",
    "RouteModel": "config: re-read from the .inp at swmm_open",
    "IgnoreGwater": "config: re-read from the .inp at swmm_open",
    "IgnoreQuality": "config: re-read from the .inp at swmm_open",
    "IgnoreRainfall": "config: re-read from the .inp at swmm_open",
    "IgnoreRouting": "config: re-read from the .inp at swmm_open",
    "IgnoreSnowmelt": "config: re-read from the .inp at swmm_open",
    # Derived at report time by massbal_get*Error() from the totals this
    # snapshot DOES restore.  Storing them too would be a second copy of a
    # derived quantity, free to disagree with the totals it came from.
    "FlowError": "derived at report time from FlowTotals",
    "RunoffError": "derived at report time from RunoffTotals",
    "GwaterError": "derived at report time from GwaterTotals",
    "QualError": "derived at report time from QualTotals",
    # Runoff clock.  The coupled TRITON-SWMM configuration drives SWMM as
    # hydraulics-only with inflows supplied externally, so the runoff clock is
    # not advanced; it is re-initialized by swmm_start(TRUE) on resume.
    "NewRunoffTime": "runoff clock; re-initialized by swmm_start on resume",
}

# Translation units the closure may walk.  A function reached outside this set
# terminates the walk: its fields are not admitted, and the call is reported.
REPORT_BEARING_UNITS = ["stats.c", "statsrpt.c", "massbal.c", "report.c"]

# The report-emitting roots the operation seeds on.
SEED_ROOTS = ["stats_report", "statsrpt_writeReport", "massbal_report"]

_FUNC_DEF = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_ \t\*]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*$",
    re.MULTILINE,
)
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def struct_fields(objects_h: str, type_name: str) -> list[str]:
    """Return the field names declared in ``type_name``'s struct BODY.

    The extent is the brace-delimited body only.  Scanning the whole header
    instead returns every identifier in the file, which is the wrong-extent
    failure this function exists to prevent.
    """
    end = objects_h.find("}  " + type_name)
    if end < 0:
        end = objects_h.find("} " + type_name)
    if end < 0:
        raise SystemExit(f"struct {type_name} not found in objects.h")
    start = objects_h.rfind("typedef struct", 0, end)
    if start < 0:
        raise SystemExit(f"typedef for {type_name} not found")
    body = objects_h[objects_h.find("{", start) + 1 : end]
    fields: list[str] = []
    for raw in body.splitlines():
        line = raw.split("//")[0].strip()
        if not line or not line.endswith(";"):
            continue
        decl = line[:-1]
        decl = decl.split("[")[0]  # drop array extents
        toks = _IDENT.findall(decl)
        if len(toks) < 2:
            continue
        fields.append(toks[-1])
    return fields


_SWMM_MODULE = re.compile(
    r"^(lid|inlet|node|link|subcatch|gwater|snow|massbal|stats|statsrpt|report|"
    r"output|datetime|xsect|table|street|culvert|roadway|project|error)_"
)


def _looks_like_swmm_api(name: str) -> bool:
    """True for a SWMM module-scoped function, false for a C library call.

    Without this the boundary list fills with ``strlen``/``fprintf``/``sprintf``
    and the one entry that matters is unreadable inside it.
    """
    return bool(_SWMM_MODULE.match(name))


def _globals_scalars(globals_h: str) -> set[str]:
    """Names declared in globals.h under an EXTERN scalar block.

    Array declarations are skipped -- they are per-object state the struct pass
    already covers -- as are the file handles and string buffers.
    """
    out: set[str] = set()
    cur: str | None = None
    for raw in globals_h.splitlines():
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^EXTERN\s+(\w+)", line)
        if m:
            cur = m.group(1)
            line = line[m.end() :]
        if cur not in ("double", "long", "int"):
            continue
        for tok in re.split(r"[,;]", line):
            tok = tok.strip()
            if not tok or "[" in tok or "*" in tok:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
                out.add(tok)
        if ";" in raw:
            cur = None
    return out


_SIG_CACHE: dict[str, list[str]] = {}


def _signature_params(bodies, origin, solver: Path, fn: str) -> list[str]:
    """Return ``fn``'s parameter NAMES, in declaration order.

    Used to bind a call-site argument to the name the callee dereferences, so a
    field read through a pointer parameter is admitted by the closure.
    """
    if fn in _SIG_CACHE:
        return _SIG_CACHE[fn]
    params: list[str] = []
    src = (solver / origin[fn]).read_text(errors="replace")
    body = bodies[fn]
    at = src.find(body)
    if at > 0:
        header = src[max(0, at - 400) : at]
        open_paren = header.rfind("(")
        close_paren = header.rfind(")")
        if 0 <= open_paren < close_paren:
            for raw in header[open_paren + 1 : close_paren].split(","):
                toks = _IDENT.findall(raw.split("[")[0])
                if len(toks) >= 2:
                    params.append(toks[-1])
                elif toks:
                    params.append(toks[-1])
    _SIG_CACHE[fn] = params
    return params


def split_functions(src: str) -> dict[str, str]:
    """Split a translation unit into ``{function name: body}`` by brace match."""
    out: dict[str, str] = {}
    for m in _FUNC_DEF.finditer(src):
        name = m.group(1)
        brace = src.find("{", m.end())
        if brace < 0:
            continue
        # A definition's body must open on the line directly after the header.
        between = src[m.end() : brace]
        if between.strip().replace("\n", "") not in ("", "//"):
            if "//" not in between and between.strip():
                continue
        depth, i = 0, brace
        while i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.setdefault(name, src[brace : i + 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver-dir", required=True, type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--inventory", type=Path, default=None)
    args = ap.parse_args()

    solver = args.solver_dir
    inv_path = args.inventory or (Path(__file__).resolve().parent / "snapshot_inventory.txt")

    objects_h = (solver / "objects.h").read_text(errors="replace")

    # --- steps 1 and 2: seed, then close transitively at function granularity
    bodies: dict[str, str] = {}
    origin: dict[str, str] = {}
    for unit in REPORT_BEARING_UNITS:
        for name, body in split_functions((solver / unit).read_text(errors="replace")).items():
            if name not in bodies:
                bodies[name] = body
                origin[name] = unit

    reached: set[str] = set()
    stack = [r for r in SEED_ROOTS]
    while stack:
        fn = stack.pop()
        if fn in reached or fn not in bodies:
            continue
        reached.add(fn)
        for callee in _CALL.findall(bodies[fn]):
            if callee in bodies and callee not in reached:
                stack.append(callee)

    missing_roots = [r for r in SEED_ROOTS if r not in reached]
    if missing_roots:
        raise SystemExit(f"seed root(s) not found in the report-bearing units: {missing_roots}")

    reached_src = "\n".join(bodies[f] for f in sorted(reached))

    # --- alias resolution ----------------------------------------------------
    #
    # A report path frequently reaches a serialized object through a pointer
    # rather than by name: `report_writeTimeStepStats(&TimeStepStats)` binds the
    # object to a callee parameter, and every field read in that callee is
    # spelled `param->field`.  A purely lexical scan for `Obj.field` misses all
    # of them and reports a field the report demonstrably prints as unread.
    # Resolving the binding is what makes the closure transitive rather than
    # lexical, which is the difference the criterion turns on.
    # The binding must PROPAGATE, not just bind once: `stats_report` passes
    # `&TimeStepStats` to `report_writeTimeStepStats`, which passes its own
    # parameter onward to `report_RouteStepFreq`, and only that third frame
    # reads `timeStepIntervals` / `timeStepCounts` -- the two arrays the report's
    # time-step frequency table is built from.  A one-hop binding reports both
    # as unread, which is a wrong answer that looks like a right one.
    alias_reads: dict[str, set[str]] = {obj: set() for obj in SERIALIZED_OBJECTS}
    _CALLSITE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;()]*)\)")
    for obj in SERIALIZED_OBJECTS:
        # worklist of (function, name-bound-to-obj-inside-that-function)
        work = [(fn, obj) for fn in sorted(reached)]
        seen: set[tuple[str, str]] = set()
        while work:
            fn, name = work.pop()
            if (fn, name) in seen or fn not in bodies:
                continue
            seen.add((fn, name))
            body = bodies[fn]
            for fm in re.finditer(
                rf"\b{re.escape(name)}\s*(?:->|\[[^\]]*\]\s*\.|\.)\s*([A-Za-z_][A-Za-z0-9_]*)", body
            ):
                alias_reads[obj].add(fm.group(1))
            for cm in _CALLSITE.finditer(body):
                callee, arglist = cm.group(1), cm.group(2)
                if callee not in bodies:
                    continue
                call_args = [a.strip() for a in arglist.split(",")]
                idx = [i for i, a in enumerate(call_args) if re.fullmatch(rf"&?\s*{re.escape(name)}", a)]
                if not idx:
                    continue
                sig = _signature_params(bodies, origin, solver, callee)
                for i in idx:
                    if i < len(sig):
                        work.append((callee, sig[i]))

    # --- step 3: emit {object, field} pairs
    lines: list[str] = []
    lines.append("# SWMM state-snapshot field inventory -- GENERATED, do not hand-edit.")
    lines.append("# Regenerate with test/inventory/regenerate_snapshot_inventory.py")
    lines.append("# Criterion: a quantity is IN the snapshot iff some report-writing path reads it.")
    lines.append(f"# Seed roots: {', '.join(SEED_ROOTS)}")
    lines.append(f"# Report-bearing units walked: {', '.join(REPORT_BEARING_UNITS)}")
    lines.append(f"# Functions reached by transitive closure: {len(reached)}")
    lines.append("#")
    # The walk stops at any callee defined outside the report-bearing units.
    # Naming those boundaries keeps the closure's EXTENT visible: a field read
    # only beyond one of them is annotated `no` for a reason a reader can check,
    # rather than looking like an absence of any reader at all.
    boundary: set[str] = set()
    for fn in reached:
        for callee in _CALL.findall(bodies[fn]):
            if callee not in bodies and _looks_like_swmm_api(callee):
                boundary.add(callee)
    # DISCOVERY PASS.  SERIALIZED_SCALARS is the one part of this operation that
    # is a list, and a list cannot tell you about the entry you forgot.  So scan
    # globals.h for scalar EXTERN declarations, ask which of them the reached
    # report path actually reads, and name any that are not in the list.  This
    # found NonConvergeCount -- read at report.c:1076 inside the time-step
    # frequency table, declared nowhere near a stats struct, and therefore
    # invisible to an inventory seeded on the structs or on stats.c's statics.
    declared = _globals_scalars((solver / "globals.h").read_text(errors="replace"))
    listed = {n for n, _, _, _ in SERIALIZED_SCALARS}
    unlisted = sorted(
        n for n in declared
        if n not in listed and n not in EXCLUDED_SCALARS
        and re.search(rf"\b{re.escape(n)}\b", reached_src)
    )
    lines.append("# discovery pass -- globals.h scalars read by the report path:")
    if unlisted:
        for n in unlisted:
            lines.append(f"#   UNTRIAGED: {n}  <-- add to SERIALIZED_SCALARS or to EXCLUDED_SCALARS")
    else:
        lines.append("#   none untriaged")
    lines.append("#")
    lines.append("# deliberately excluded, with reason:")
    for n in sorted(EXCLUDED_SCALARS):
        if re.search(rf"\b{re.escape(n)}\b", reached_src):
            lines.append(f"#   {n}: {EXCLUDED_SCALARS[n]}")
    lines.append("#")
    lines.append("# closure boundary -- report-reaching calls into units NOT walked:")
    for c in sorted(boundary):
        lines.append(f"#   {c}")
    lines.append("#")
    lines.append("# object<TAB>field<TAB>declared<TAB>read_by_report_path")

    total_pairs = 0
    for obj in sorted(SERIALIZED_OBJECTS):
        typ = SERIALIZED_OBJECTS[obj]
        fields = struct_fields(objects_h, typ)
        lines.append(f"#")
        lines.append(f"# {obj} : {typ} -- {len(fields)} field(s) declared in the struct body")
        for f in fields:
            pat = re.compile(rf"\b{re.escape(obj)}\b[^;\n]*?\.\s*{re.escape(f)}\b")
            read = bool(pat.search(reached_src)) or f in alias_reads[obj]
            lines.append(f"{obj}\t{f}\t{typ}\t{'yes' if read else 'no'}")
            total_pairs += 1

    lines.append("#")
    lines.append("# scalar accumulators (no struct body; extent check does not apply)")
    for name, ctype, owner, storage in SERIALIZED_SCALARS:
        read = bool(re.search(rf"\b{re.escape(name)}\b", reached_src))
        lines.append(f"{name}\t-\t{ctype}\t{'yes' if read else 'no'}\t{owner}\t{storage}")
        total_pairs += 1

    lines.append("#")
    lines.append(f"# TOTAL PAIRS: {total_pairs}")
    text = "\n".join(lines) + "\n"

    if args.check:
        if not inv_path.exists():
            print(f"FAIL: committed inventory not found at {inv_path}", file=sys.stderr)
            return 1
        committed = inv_path.read_text()
        if committed != text:
            print("FAIL: regenerated inventory differs from the committed one.", file=sys.stderr)
            print(
                "".join(
                    difflib.unified_diff(
                        committed.splitlines(keepends=True),
                        text.splitlines(keepends=True),
                        fromfile="committed",
                        tofile="regenerated",
                    )
                ),
                file=sys.stderr,
            )
            print(
                "\nThe vendored EPA SWMM source changed a struct the snapshot serializes.\n"
                "Re-run without --check, review the diff, and update the serializer to\n"
                "match before committing the new inventory.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: regenerated inventory matches {inv_path} ({total_pairs} pairs)")
        return 0

    inv_path.write_text(text)
    print(f"wrote {inv_path} ({total_pairs} pairs, {len(reached)} functions reached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

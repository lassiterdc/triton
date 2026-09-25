#!/usr/bin/env python3
"""Regenerate the SWMM state-snapshot field inventory from EPA SWMM source.

This script performs the TWO OPERATIONS that define the snapshot's contents,
because the snapshot serves two requirements and one criterion cannot reach
both:

    Criterion R (report state)  -- a quantity is IN the snapshot iff some
                                   report-writing path reads it.
    Criterion P (routing state) -- a quantity is IN the snapshot iff a step
                                   beginning at t_k READS it before that step
                                   writes it.

Criterion R is the original and is documented immediately below; Criterion P was
added after WP-1B's acceptance gate FAILED, because the snapshot captured report
accumulators and no live routing state and the implementation had followed the
only criterion it was given.  Criterion P's machinery, its order-sensitive kill
pass and its own closure check live in a clearly-marked block further down, as
does the D-R6 stream-position exclusion table.

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

# =============================================================================
# CRITERION P -- routing state live on entry at the step boundary
# =============================================================================
#
# Criterion R above answers the REPORT requirement.  It cannot answer the
# PHYSICS requirement, and WP-1B's acceptance gate failed because the
# implementation followed the only criterion it was given.  Criterion P is the
# second criterion:
#
#     a quantity is IN the snapshot iff a step beginning at t_k READS it before
#     that step writes it -- i.e. iff it is LIVE-ON-ENTRY at the step boundary
#
# and the step boundary is ``swmm_step``'s entry, NOT ``routing_execute``'s.
#
# The two criteria have OPPOSITE error costs.  Under R over-capture is harmful:
# a second copy of a derived quantity is free to disagree with what it came
# from.  Under P UNDER-capture is the wrong numbers and over-capture is merely a
# maintenance cost -- EXCEPT for .inp configuration, where a stale snapshot
# would silently override the model the operator is running.  So P admits state
# freely and excludes configuration by triage.

# The routing-step entry points the closure seeds on.
ROUTING_SEED_ROOTS = [
    "routing_execute",
    "routeFlow",
    "flowrout_execute",
    "dynwave_execute",
]

# The step PROLOGUE, IN EXECUTION ORDER.  The kill pass is order-sensitive and a
# FLAT kill set is UNSOUND: `Node.inflow` and `Node.outflow` are read by
# `node_setOldHydState` BEFORE `node_initFlows` overwrites them, and
# `Node.newLatFlow` is read one line before it is zeroed inside
# `initSystemInflows`.  All three are LIVE; a flat pass reports all three
# KILLED.  Order is therefore part of the operation, not an optimisation.
# The step the criterion is defined over.  §4.6.1 declares the boundary at
# ``swmm_step``'s entry, so the liveness stream is rooted there even though the
# four SEED roots above are the ROUTING roots: a stream rooted at
# ``routing_execute`` cannot see the pre-prologue read of ``Node[j].overflow``,
# and a closure rooted only at the four cannot see ``ReportTime`` at all.
STEP_ROOT = "swmm_step"

ROUTING_PROLOGUE = [
    "initSystemInflows",
    "link_setOldHydState",
    "node_setOldHydState",
    "node_initFlows",
    "initRoutingStep",
    "initNodeStates",
]

# Translation units the MUTABLE-STATE PRE-FILTER excludes from the candidate
# pool.  This is a COMMITTED DECLARATION, deliberately not a recomputation: the
# script re-derives each unit's mutable-state verdict from source and reports
# PREFILTER-VIOLATION for any member that declares mutable state at the current
# pin.  That is closure-check failure mode (iv), and it is the mode a
# translation-unit bound could not fail on.  A pre-filter that both computed and
# checked itself would be self-consistent and blind -- the shape this design
# calls a check that runs, returns a pass, and sees nothing.
#
# An excluded unit is still WALKED.  What the pre-filter removes is the unit's
# own DECLARATIONS from the candidate pool, never the unit from the closure, so
# state the unit reaches through a call is still found.
PREFILTER_EXCLUDED_UNITS = [
    "findroot.c",
    "forcmain.c",
    "xsect.c",
]

# Routing state ADMITTED by Criterion P: read by the step before the step writes
# it, and therefore restored from the snapshot.
ADMITTED_ROUTING_STATE = {
    ("Conduit", "a1"):
        "routing state: initRoutingStep copies a1 into a2 at the top of every step; the multi-declarator drop in struct_fields is the defect that would have hidden it",
    ("Conduit", "capacityLimited"):
        "routing state: per-step capacity flag read by the next step",
    ("Conduit", "evapLossRate"):
        "routing state: per-step loss rate; admitted because under P over-capture is a maintenance cost and under-capture is the wrong numbers",
    ("Conduit", "fullState"):
        "routing state: full-flow state carried across steps",
    ("Conduit", "q1"):
        "routing state: upstream flow carried across steps",
    ("Conduit", "q2"):
        "routing state: downstream flow carried across steps",
    ("Conduit", "seepLossRate"):
        "routing state: per-step loss rate; admitted on the same ground as evapLossRate",
    ("Link", "dqdh"):
        "routing state: the derivative the dynamic-wave solution carries across steps",
    ("Link", "flowClass"):
        "routing state: flow classification carried across steps",
    ("Link", "froude"):
        "routing state: Froude number carried across steps",
    ("Link", "inletControl"):
        "routing state: inlet-control flag carried across steps",
    ("Link", "newDepth"):
        "routing state: the depth the resumed step continues from",
    ("Link", "newFlow"):
        "routing state: the flow the resumed step continues from",
    ("Link", "newVolume"):
        "routing state: conduit volume carried across steps",
    ("Link", "normalFlow"):
        "routing state: normal-flow limiter state",
    ("Link", "setting"):
        "control state: evolved by control rules during the run, not re-derivable from the .inp",
    ("Link", "targetSetting"):
        "control state: evolved by control rules during the run, not re-derivable from the .inp",
    ("Link", "timeLastSet"):
        "control state: the time the last control action fired",
    ("Node", "inflow"):
        "routing state: read by node_setOldHydState BEFORE node_initFlows overwrites it",
    ("Node", "losses"):
        "routing state: per-step evaporation and seepage losses",
    ("Node", "newDepth"):
        "routing state: the depth the resumed step continues from",
    ("Node", "newLatFlow"):
        "routing state: read one line before it is zeroed inside initSystemInflows",
    ("Node", "newVolume"):
        "routing state: reconstruction from restored primitives is bitwise only in the non-flooded branch, and the coupled case is the flooded branch",
    ("Node", "oldDepth"):
        "routing state: previous-step depth, read by the routing solution",
    ("Node", "oldFlowInflow"):
        "routing state: previous-step inflow, read by the steady-state test",
    ("Node", "oldNetInflow"):
        "routing state: previous-step net inflow, read by the steady-state test",
    ("Node", "outflow"):
        "routing state: read by node_setOldHydState BEFORE node_initFlows overwrites it",
    ("Node", "overflow"):
        "routing state: read at massbal.c:635 by the opening massbal_updateRoutingTotals, before the prologue runs",
    ("Node", "updated"):
        "routing state: the per-step updated flag the dynamic-wave solution reads",
    ("Outfall", "vRouted"):
        "routing accumulator: volume routed to the outfall since t=0",
    ("Outfall", "wRouted"):
        "routing accumulator: mass routed to the outfall since t=0",
    ("Storage", "evapLoss"):
        "routing state: per-step evaporation loss",
    ("Storage", "exfilLoss"):
        "routing state: per-step exfiltration loss",
    ("Storage", "hrt"):
        "routing state: hydraulic residence time accumulated across steps",
    ("Xnode", "dYdT"):
        "routing state: depth derivative carried across Picard iterations and steps",
    ("Xnode", "oldSurfArea"):
        "routing state: written only in setNodeDepth's non-surcharged branch and read only in its surcharged branch, by design; appears in NO struct body in objects.h and is the single field that refuses a hand-written list",
}

# Routing state EXCLUDED by Criterion P, every exclusion carrying its reason.
EXCLUDED_ROUTING_STATE = {
    # CORRECTED at WP-1B(10). This was ADMITTED with the reason "routing state:
    # surcharge depth carried across steps". Measured: the only writes to
    # Node[].surDepth are node.c's .inp readers (node_readParams and its
    # per-type arms) and the only reads are in dynwave.c -- no routing write
    # exists anywhere in the tree. It is .inp CONFIGURATION, which is the one
    # class Sec 4.6.1 excludes by triage rather than admitting freely, because
    # over-capture there lets a stale snapshot silently override the model the
    # operator is running. Under-capture is the wrong numbers; THIS direction is
    # the wrong model, which is worse and is why the exception exists.
    ("Node", "surDepth"):
        "config: .inp surcharge depth -- written only by node.c's .inp readers "
        "and read only in dynwave.c; no routing write exists, so admitting it "
        "would let a stale snapshot override the running model",
    ("Conduit", "barrels"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "beta"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "hasLosses"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "length"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "modLength"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "qMax"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "roughFactor"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "roughness"):
        "config: re-read from the .inp at swmm_open",
    ("Conduit", "slope"):
        "config: re-read from the .inp at swmm_open",
    ("Gage", "isUsed"):
        "config: re-read from the .inp at swmm_open",
    ("Gage", "pastRain"):
        "runoff state; not advanced in the coupled hydraulics-only configuration (the ground NewRunoffTime already carries)",
    ("Gage", "rainfall"):
        "runoff state; not advanced in the coupled hydraulics-only configuration (the ground NewRunoffTime already carries)",
    ("Link", "ID"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "cLossAvg"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "cLossInlet"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "cLossOutlet"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "direction"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "hasFlapGate"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "inlet"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "newQual"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Link", "node1"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "node2"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "offset1"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "offset2"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "oldQual"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Link", "qFull"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "qLimit"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "seepRate"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "subIndex"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "totalLoad"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Link", "type"):
        "config: re-read from the .inp at swmm_open",
    ("Link", "xsect"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "ID"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "apiExtInflow"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "crownElev"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "degree"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "dwfInflow"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "extInflow"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "fullDepth"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "fullVolume"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "inlet"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "invertElev"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "newQual"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Node", "oldQual"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Node", "pondedArea"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "qualInflow"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Node", "subIndex"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "treatment"):
        "config: re-read from the .inp at swmm_open",
    ("Node", "type"):
        "config: re-read from the .inp at swmm_open",
    ("Outfall", "fixedStage"):
        "config: re-read from the .inp at swmm_open",
    ("Outfall", "routeTo"):
        "config: re-read from the .inp at swmm_open",
    ("Outfall", "stageSeries"):
        "config: re-read from the .inp at swmm_open",
    ("Outfall", "tideCurve"):
        "config: re-read from the .inp at swmm_open",
    ("Outfall", "type"):
        "config: re-read from the .inp at swmm_open",
    ("Storage", "a0"):
        "config: re-read from the .inp at swmm_open",
    ("Storage", "a1"):
        "config: re-read from the .inp at swmm_open",
    ("Storage", "a2"):
        "config: re-read from the .inp at swmm_open",
    ("Storage", "aCurve"):
        "config: re-read from the .inp at swmm_open",
    ("Storage", "exfil"):
        "config: re-read from the .inp at swmm_open",
    ("Storage", "fEvap"):
        "config: re-read from the .inp at swmm_open",
    ("Storage", "shape"):
        "config: re-read from the .inp at swmm_open",
    ("Subcatch", "area"):
        "config: re-read from the .inp at swmm_open",
    ("Subcatch", "groundwater"):
        "config: re-read from the .inp at swmm_open",
    ("Subcatch", "infilPattern"):
        "config: re-read from the .inp at swmm_open",
    ("Subcatch", "lidArea"):
        "config: re-read from the .inp at swmm_open",
    ("Subcatch", "newQual"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Subcatch", "newRunoff"):
        "runoff state; not advanced in the coupled hydraulics-only configuration (the ground NewRunoffTime already carries)",
    ("Subcatch", "oldQual"):
        "water-quality state; not advanced in the coupled hydraulics-only configuration",
    ("Subcatch", "oldRunoff"):
        "runoff state; not advanced in the coupled hydraulics-only configuration (the ground NewRunoffTime already carries)",
    ("Subcatch", "outNode"):
        "config: re-read from the .inp at swmm_open",
}

# Whole-object exclusion RULES.  An object appears here only when every field it
# carries belongs to one class for one reason; a per-field entry above always
# wins over the rule.  A rule does NOT weaken closure-check modes (i)/(iii): the
# artifact lists every field the rule swallowed, so a field EPA adds to a
# rule-excluded struct still changes the emitted text and still fails the diff.
EXCLUDED_ROUTING_OBJECTS = {
    "A":
        "config: re-read from the .inp at swmm_open",
    "Adjust":
        "config: re-read from the .inp at swmm_open",
    "Curve":
        "config: re-read from the .inp at swmm_open",
    "Divider":
        "config: re-read from the .inp at swmm_open",
    "Evap":
        "config: recomputed every step by climate_setState from .inp-derived series",
    "Event":
        "config: re-read from the .inp at swmm_open",
    "LidProcs":
        "config: re-read from the .inp at swmm_open",
    "Orifice":
        "config: re-read from the .inp at swmm_open",
    "Outlet":
        "config: re-read from the .inp at swmm_open",
    "Pattern":
        "config: re-read from the .inp at swmm_open",
    "Pollut":
        "config: re-read from the .inp at swmm_open",
    "Pump":
        "config: re-read from the .inp at swmm_open",
    "R":
        "config: re-read from the .inp at swmm_open",
    "RptFlags":
        "config: re-read from the .inp at swmm_open",
    "Street":
        "config: re-read from the .inp at swmm_open",
    "TimeStepStats":
        "report accumulator; admitted by Criterion R above and restored by the same snapshot",
    "Transect":
        "config: re-read from the .inp at swmm_open",
    "Weir":
        "config: re-read from the .inp at swmm_open",
    "pXsect":
        "config: re-read from the .inp at swmm_open",
    "xsect":
        "config: re-read from the .inp at swmm_open",
}

# --- D-R6: the non-field (stream-position) exclusion table -------------------
#
# A field-liveness closure cannot see a FILE STREAM POSITION, whatever axis it
# is bounded on.  The enumeration is keyed on the TYPE ``TFile``, never on a
# declaration site: a grep for the type over the solver headers returns exactly
# two sites -- the ``EXTERN TFile`` block in globals.h introducing eleven
# handles, and a ``TFile file;`` MEMBER of ``TTable``, one stream per external
# time series.  Enumerating from the globals block alone is the obvious move and
# it silently misses the per-time-series cursor, which is the same failure shape
# as a hand-written field list missing a module static: the enumeration was
# keyed on a PLACE rather than on the property that defines membership.
#
# The captured quantity would be uniform -- one ``long`` from ``ftell`` per
# handle -- so the cost is O(1) in handle count either way.  An EMPTY admitted
# set is a PASSING result: the deliverable of this pass is the table.
TFILE_ADMITTED = {}

TFILE_EXCLUDED = {
    "Fclimate":
        "climate file; the coupled hydraulics-only configuration does not advance the climate clock",
    "Fhotstart1":
        "hot-start INPUT file; read once at swmm_start and never re-read, so no cursor survives the call",
    "Fhotstart2":
        "hot-start OUTPUT file; written once at swmm_end, after the resumed segment has completed",
    "Finflows":
        "routing-interface inflow file; not in use in the coupled configuration (mode NO_FILE)",
    "Finp":
        "input file; re-read from the .inp at swmm_open, and the resumed process reopens it at position 0 by construction",
    "Fout":
        "unrestorable by ANY offset: output_openOutFile reopens it 'w+b', which TRUNCATES, and output_open resets Nperiods = 0, so there are no bytes to seek past -- and the 0..t_k period records were never written in the resumed process because saveResults() is never called for those steps",
    "Foutflows":
        "routing-interface outflow file; not in use in the coupled configuration (mode NO_FILE)",
    "Frain":
        "rainfall file; the coupled hydraulics-only configuration supplies inflows externally and does not read rainfall",
    "Frdii":
        "RDII interface file; not in use in the coupled configuration (mode NO_FILE)",
    "Frpt":
        "report file; the full-window hydraulics.rpt is produced by the replayed report path, not by a restored offset",
    "Frunoff":
        "runoff interface file; not in use in the coupled configuration (mode NO_FILE)",
    "TTable.file":
        "per-time-series cursor. It SELF-HEALS: table_tseriesLookup has two healing branches -- a degenerate bracket (x1 == x2) and a lookup falling left of it (x < x1) -- and both route to table_getFirstEntry, which rewinds and re-scans for a file-backed series (table.c:773-781). The design offered a SECOND ground, that admitting it is strictly worse than not because struct_fields dropped x1/y1 from `double x1, x2;` and a half-restored bracket interpolates against a bogus y1; THAT GROUND IS RETIRED, because this chunk repaired struct_fields and the helper now returns all four. The self-healing ground stands alone and is sufficient. Do not cite the repaired defect.",
}


_FUNC_DEF = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_ \t\*]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*(?://.*)?$",
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
        # A C declaration may declare SEVERAL fields: `double x1, x2;`.  Taking
        # only the last identifier of the whole line drops every declarator but
        # the final one, with no diagnostic -- and because the regeneration
        # check compares two outputs of THIS helper, a field neither side
        # enumerates makes the check pass while blind.  `TConduit` uses the form
        # three times (`a1`/`a2`, `q1`/`q2`, `q1Old`/`q2Old`) and `a1` is
        # live-on-entry routing state, so Criterion P cannot tolerate the drop.
        parts = decl.split(",")
        first = parts[0].split("[")[0]
        toks = _IDENT.findall(first)
        if len(toks) < 2:
            continue
        fields.append(toks[-1])
        for extra in parts[1:]:
            extra_toks = _IDENT.findall(extra.split("[")[0])
            if extra_toks:
                fields.append(extra_toks[-1])
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


# --- Criterion P machinery ---------------------------------------------------

_STATIC_DECL = re.compile(
    r"^static\s+(const\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(\**)\s*([^;(){}=]*?)\s*[;=]",
    re.MULTILINE,
)
_FILE_SCOPE_DEF = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*(\**)\s+([A-Za-z_][A-Za-z0-9_][^;(){}=]*?)\s*;",
    re.MULTILINE,
)
_C_KEYWORD = {
    "return", "typedef", "struct", "union", "enum", "extern", "static", "const",
    "void", "if", "else", "for", "while", "do", "switch", "case", "break",
    "continue", "goto", "sizeof", "default",
}


def _strip_function_bodies(src: str) -> str:
    """Return ``src`` with every brace-delimited block replaced by a blank.

    File-scope declarations are what the pre-filter asks about.  Scanning the
    raw text instead finds every local ``static`` and every local declaration
    inside a function body, which is the wrong extent -- the same class of
    error as counting fields over the whole header rather than the struct body.
    """
    out, depth = [], 0
    for ch in src:
        if ch == "{":
            depth += 1
            out.append(" ")
        elif ch == "}":
            depth = max(0, depth - 1)
            out.append(" ")
        else:
            out.append(" " if depth else ch)
    return "".join(out)


def _declarator_names(blob: str) -> list[str]:
    names = []
    for part in blob.split(","):
        toks = _IDENT.findall(part.split("[")[0])
        if toks and toks[-1] not in _C_KEYWORD:
            names.append(toks[-1])
    return names


def file_scope_mutable_state(src: str) -> tuple[list[str], list[str]]:
    """Return ``(non-const file-scope statics, non-static file-scope globals)``.

    ``const`` is the discriminator the pre-filter's own wording turns on and it
    is the one a line-counting grep loses.  Measured at this pin, a
    ``grep -cE '^static +[^(]*[;=]'`` returns 2 for ``massbal.c`` and both hits
    are ``static const double MAX_*_BALANCE_ERR`` -- constants.  Under a
    statics-only pre-filter ``massbal.c`` would be EXCLUDED and
    ``Node[].overflow`` killed a second time, undoing the repair this criterion
    exists to make.  ``massbal.c`` is admitted by the SECOND clause: it declares
    mutable file-scope state as non-static globals (``FlowTotals``,
    ``NodeInflow``, ``NodeOutflow``, ``TotalArea``) and writes through them.
    """
    head = _strip_function_bodies(src)
    statics: list[str] = []
    for m in _STATIC_DECL.finditer(head):
        if m.group(1):  # const
            continue
        statics.extend(_declarator_names(m.group(4)))
    globals_: list[str] = []
    for m in _FILE_SCOPE_DEF.finditer(head):
        if m.group(1) in _C_KEYWORD:
            continue
        globals_.extend(_declarator_names(m.group(3)))
    return sorted(set(statics)), sorted(set(globals_))


def extern_global_names(globals_h: str) -> set[str]:
    """Every name the ``EXTERN`` blocks of globals.h declare, any type."""
    out: set[str] = set()
    cur: str | None = None
    for raw in globals_h.splitlines():
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^EXTERN\s+([A-Za-z_][A-Za-z0-9_]*)\s*\**", line)
        if m:
            cur = m.group(1)
            line = line[m.end():]
        if cur is None:
            continue
        out.update(_declarator_names(line.rstrip(";")))
        if ";" in raw:
            cur = None
    return out


def all_solver_functions(solver: Path) -> dict[str, list[tuple[str, str]]]:
    """``{function name: [(translation unit, body), ...]}`` over every ``*.c``.

    Keyed as a MULTIMAP because seven names are defined in more than one unit at
    this pin.  Collapsing them to the first definition attributes a body to the
    wrong unit and silently drops the others from the walk.
    """
    defs: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(solver.glob("*.c")):
        for name, body in split_functions(path.read_text(errors="replace")).items():
            defs.setdefault(name, []).append((path.name, body))
    return defs


def routing_closure(defs, roots: list[str]) -> set[tuple[str, str]]:
    """Transitive closure from ``roots`` with NO translation-unit restriction.

    The retired form of this operation bounded the walk to nine named units.
    That bound provably killed a provably-live field: ``routing_execute``'s
    FIRST statement is ``massbal_updateRoutingTotals``, whose per-node loop
    reads ``Node[j].overflow`` before anything in the step prologue runs, and
    ``massbal.c`` was not among the nine.  A unit list does not separate
    state-carrying from stateless, and it does not separate reachable from
    unreachable -- most of the units outside the nine are entered by an ordinary
    first-hop call.  The bound moved to the FIELD axis; the walk has none.
    """
    bodies = {(unit, name): body for name, v in defs.items() for unit, body in v}
    reached: set[tuple[str, str]] = set()
    stack = [(unit, name) for name in roots for unit, _ in defs.get(name, [])]
    missing = [r for r in roots if r not in defs]
    if missing:
        raise SystemExit(f"routing seed root(s) not found in the solver tree: {missing}")
    while stack:
        key = stack.pop()
        if key in reached:
            continue
        reached.add(key)
        for callee in _CALL.findall(bodies[key]):
            for unit2, _ in defs.get(callee, []):
                if (unit2, callee) not in reached:
                    stack.append((unit2, callee))
    return reached


_WRITE_AFTER = re.compile(r"^\s*(\+\+|--|(?:[-+*/%|&^]|<<|>>)=|=(?!=))")
_PREFIX_INCDEC = re.compile(r"(\+\+|--)\s*$")


def _classify(body: str, start: int, end: int) -> str:
    """``'w'`` for a pure write, ``'r'`` otherwise.

    A read-modify-write (``+=``, ``++``) counts as a READ, because it consumes
    the value the previous step left.  Only a PURE assignment kills.
    """
    m = _WRITE_AFTER.match(body[end:end + 4])
    if m:
        op = m.group(1)
        return "w" if op == "=" else "r"
    if _PREFIX_INCDEC.search(body[max(0, start - 3):start]):
        return "r"
    return "r"


_REF = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]\[]*\]\s*)?(?:(->|\.)\s*([A-Za-z_][A-Za-z0-9_]*))?"
)


def state_events(body: str, objects: frozenset, scalars: frozenset):
    """Ordered ``(offset, kind, object, field)`` events in one span of source.

    ``objects`` are struct-typed globals and file-scope statics reached through
    ``[i].f`` / ``->f`` / ``.f``; ``scalars`` are bare names.  Order is the
    character offset, which is the statement order the kill pass needs.

    ONE combined pass rather than a regex per candidate: the candidate pool runs
    to several hundred names and the step stream re-scans every inlined span, so
    a per-name scan is quadratic in a way that does not finish.
    """
    events = []
    for m in _REF.finditer(body):
        name, arrow, field = m.group(1), m.group(2), m.group(3)
        if arrow and name in objects:
            events.append((m.start(), _classify(body, m.start(), m.end()), name, field))
        elif not arrow and name in scalars:
            events.append((m.start(), _classify(body, m.start(), m.end()), "-", name))
    return events


def step_event_stream(defs, reached, objects, scalars):
    """Events over the STEP, in execution order, from ``swmm_step``'s entry.

    The boundary is ``swmm_step``'s entry and NOT ``routing_execute``'s, and the
    distinction is load-bearing rather than pedantic.  ``routing_execute``'s
    FIRST statement is ``massbal_updateRoutingTotals``, whose per-node loop
    reads ``Node[j].overflow`` -- so a stream that began at the prologue would
    see only the later prologue zeroing and classify the field DEAD.  It is
    LIVE.  Four pieces of the step run before ``routing_execute`` and one
    (``saveResults``) runs after it; the same widening is what makes
    ``ReportTime`` visible as ordinary live-on-entry state.

    A callee is INLINED at its call site so the stream is EXECUTION order rather
    than declaration order.  Recursion is guarded by the current path, so a
    cycle terminates without truncating a sibling subtree.
    """
    bodies = {(unit, name): body for name, v in defs.items() for unit, body in v}
    by_name: dict[str, str] = {}
    for (unit, name), body in bodies.items():
        by_name.setdefault(name, body)

    stream: list[tuple[str, str, str, str]] = []

    def walk(name: str, path: frozenset) -> None:
        if name in path or name not in by_name:
            return
        body = by_name[name]
        path = path | {name}
        marks = [
            (m.start(), m.group(1))
            for m in _CALL.finditer(body)
            if m.group(1) in by_name and m.group(1) not in path
        ]
        cut = 0
        for at, callee in marks:
            for _off, kind, obj, field in state_events(body[cut:at], objects, scalars):
                stream.append((kind, obj, field, name))
            walk(callee, path)
            cut = at
        for _off, kind, obj, field in state_events(body[cut:], objects, scalars):
            stream.append((kind, obj, field, name))

    walk(STEP_ROOT, frozenset())
    return stream


def kill_pass(candidates, stream):
    """Split ``candidates`` into ``(live, killed)`` by STEP execution order.

    A field is KILLED only when a PURE WRITE performed by one of the six
    PROLOGUE functions precedes EVERY read of it in the step's own execution
    order.  Two halves of that rule each rule out a measured wrong answer:

      * Only a PROLOGUE write kills.  The prologue's writes are unconditional
        per-object loops; a write elsewhere in the step may sit inside a branch,
        and treating a textually-earlier conditional write as a kill would
        subtract a field the step can read first.

      * Reads are counted over the WHOLE step, not the prologue.  The read of
        ``Node[j].overflow`` at ``massbal.c:635`` happens before the prologue
        begins, so a prologue-only read set would kill a live field.

    A read-modify-write (``+=``, ``++``) counts as a READ: it consumes the value
    the previous step left.  A FLAT kill set -- one that ignores order -- was
    measured wrong on three fields: ``Node.inflow`` and ``Node.outflow`` are
    read by ``node_setOldHydState`` BEFORE ``node_initFlows`` overwrites them,
    and ``Node.newLatFlow`` is read one line before it is zeroed inside
    ``initSystemInflows``.  All three are LIVE.
    """
    decided: dict[tuple[str, str], str] = {}
    for kind, obj, field, fn in stream:
        key = (obj, field)
        if key in decided:
            continue
        if kind == "r":
            decided[key] = "live"
        elif fn in ROUTING_PROLOGUE:
            decided[key] = "killed"
        # a pure write outside the prologue decides nothing
    killed = {k for k in candidates if decided.get(k) == "killed"}
    return sorted(candidates - killed), sorted(killed)


def prologue_order_holds(stream) -> list[str]:
    """Return the prologue functions, in the order the step stream reaches them.

    Closure-check mode (ii) asks whether a field marked killed has a kill site
    preceding every read of it IN PROLOGUE ORDER.  That question is only
    well-posed while the declared prologue order is the order the source
    actually executes, so the order is re-derived here and compared against the
    committed ``ROUTING_PROLOGUE`` rather than assumed.
    """
    seen: list[str] = []
    for _kind, _obj, _field, fn in stream:
        if fn in ROUTING_PROLOGUE and fn not in seen:
            seen.append(fn)
    return seen


def _struct_typed_globals(globals_h: str) -> set[str]:
    """``EXTERN`` names whose type is a SWMM struct (``T``-prefixed)."""
    out: set[str] = set()
    cur: str | None = None
    for raw in globals_h.splitlines():
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^EXTERN\s+([A-Za-z_][A-Za-z0-9_]*)\s*\**", line)
        if m:
            cur = m.group(1)
            line = line[m.end():]
        if cur == "TFile":
            # A stream handle is enumerated by the D-R6 table keyed on the TYPE,
            # not as a field of a struct-typed global.  Leaving it in both pools
            # would triage the same object twice under two different criteria.
            if ";" in raw:
                cur = None
            continue
        if cur is None or not re.fullmatch(r"T[A-Z][A-Za-z0-9_]*", cur):
            if ";" in raw:
                cur = None
            continue
        out.update(_declarator_names(line.rstrip(";")))
        if ";" in raw:
            cur = None
    return out


def _scalar_globals(globals_h: str) -> set[str]:
    """``EXTERN`` names of plain scalar type -- the clock and counter class."""
    out: set[str] = set()
    cur: str | None = None
    for raw in globals_h.splitlines():
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^EXTERN\s+([A-Za-z_][A-Za-z0-9_]*)\s*\**", line)
        if m:
            cur = m.group(1)
            line = line[m.end():]
        if cur is None:
            continue
        if cur in ("long", "double", "int", "DateTime"):
            for part in line.rstrip(";").split(","):
                if "[" in part:
                    continue  # array: per-object config, not a scalar
                names = _declarator_names(part)
                out.update(names)
        if ";" in raw:
            cur = None
    return out


def emit_p_section(solver: Path, lines: list[str]) -> int:
    """Append Criterion P's section to ``lines``; return the row count."""
    globals_h = (solver / "globals.h").read_text(errors="replace")
    objects_h = (solver / "objects.h").read_text(errors="replace")

    defs = all_solver_functions(solver)
    reached = routing_closure(defs, ROUTING_SEED_ROOTS)
    reached_units = sorted({unit for unit, _ in reached})

    # --- the mutable-state PRE-FILTER, recomputed from source ----------------
    verdict: dict[str, tuple[bool, str]] = {}
    unit_statics: dict[str, list[str]] = {}
    ext_globals = extern_global_names(globals_h)
    for unit in reached_units:
        src = (solver / unit).read_text(errors="replace")
        statics, file_globals = file_scope_mutable_state(src)
        unit_statics[unit] = statics
        writes_global = False
        for name in sorted(ext_globals):
            if re.search(
                rf"\b{re.escape(name)}\b\s*(?:\[[^\]]*\]\s*)?(?:(?:->|\.)\s*[A-Za-z_][A-Za-z0-9_]*\s*)?"
                rf"(?:\[[^\]]*\]\s*)?(?:[-+*/%|&^]|<<|>>)?=(?!=)",
                src,
            ):
                writes_global = True
                break
        if statics:
            verdict[unit] = (True, f"declares {len(statics)} non-const file-scope static(s)")
        elif file_globals:
            verdict[unit] = (True, f"defines {len(file_globals)} file-scope global(s)")
        elif writes_global:
            verdict[unit] = (True, "writes through a global")
        else:
            verdict[unit] = (False, "declares no mutable state and writes no global")

    computed_excluded = sorted(u for u in reached_units if not verdict[u][0])
    contributing = [u for u in reached_units if verdict[u][0]]

    # --- candidate pool ------------------------------------------------------
    struct_globals = _struct_typed_globals(globals_h)
    scalar_globals = _scalar_globals(globals_h)
    bodies = {(unit, name): body for name, v in defs.items() for unit, body in v}

    objects = set(struct_globals)
    scalars = set(scalar_globals)
    for unit in contributing:
        for name in unit_statics[unit]:
            objects.add(name)
            scalars.add(name)

    objects = frozenset(objects)
    scalars = frozenset(scalars)

    # STRUCT-FIELD candidates come from the ROUTING closure (the four seed
    # roots) restricted to contributing units.  SCALAR candidates come from the
    # STEP stream instead, because §4.6.3 records that widening the boundary to
    # swmm_step is what makes `ReportTime` visible as ordinary live-on-entry
    # state -- and `ReportTime`, `Nperiods`, `TotalStepCount` and
    # `ReportStepCount` are referenced nowhere in the four-root routing closure.
    # Two bases rather than one because the two classes are reached by
    # different halves of the same declaration.
    candidates: set[tuple[str, str]] = set()
    for unit, fn in sorted(reached):
        if unit not in contributing:
            continue
        for _, _kind, obj, field in state_events(bodies[(unit, fn)], objects, frozenset()):
            candidates.add((obj, field))



    # A lexical match against a struct-typed global must name a REAL member of
    # that struct, or it is a false positive the pool would carry forever.
    validated: set[tuple[str, str]] = set()
    rejected: list[tuple[str, str]] = []
    member_cache: dict[str, set[str]] = {}
    for obj, field in sorted(candidates):
        if obj == "-":
            validated.add((obj, field))
            continue
        if obj not in member_cache:
            member_cache[obj] = _members_of_global(objects_h, globals_h, obj)
        members = member_cache[obj]
        if not members or field in members:
            validated.add((obj, field))
        else:
            rejected.append((obj, field))

    stream = step_event_stream(defs, reached, objects, scalars)
    for _kind, obj, field, _fn in stream:
        if obj == "-":
            candidates.add((obj, field))
    live, killed = kill_pass(validated, stream)
    prologue_seen = prologue_order_holds(stream)

    # --- emit ----------------------------------------------------------------
    lines.append("#")
    lines.append("# " + "=" * 74)
    lines.append("# CRITERION P -- routing state LIVE ON ENTRY at swmm_step's boundary")
    lines.append("# " + "=" * 74)
    lines.append("# Criterion: a quantity is IN the snapshot iff a step beginning at t_k READS")
    lines.append("#            it before that step writes it.")
    lines.append(f"# Seed roots: {', '.join(ROUTING_SEED_ROOTS)}")
    lines.append("# Walk: UNRESTRICTED -- every solver translation unit the closure reaches.")
    lines.append(f"# Functions reached: {len(reached)} across {len(reached_units)} translation unit(s)")
    lines.append(f"# Step stream rooted at: {STEP_ROOT} (the declared boundary)")
    lines.append(f"# Prologue, as declared: {' -> '.join(ROUTING_PROLOGUE)}")
    lines.append(f"# Prologue, as the step stream reaches it: {' -> '.join(prologue_seen)}")
    _declared_seen = [f for f in ROUTING_PROLOGUE if f in prologue_seen]
    if prologue_seen != _declared_seen:
        lines.append("#   PROLOGUE-ORDER-VIOLATION: the source order no longer matches the")
        lines.append("#   declared order, so mode (ii) is not well-posed at this pin.")
    _absent = [f for f in ROUTING_PROLOGUE if f not in prologue_seen]
    if _absent:
        lines.append(f"#   PROLOGUE-UNREACHED: {', '.join(_absent)}")
    lines.append("#")
    lines.append("# mutable-state pre-filter -- per reached unit:")
    for unit in reached_units:
        ok, why = verdict[unit]
        lines.append(f"#   {'contributes' if ok else 'EXCLUDED   '} {unit}: {why}")
    lines.append("#")
    lines.append("# closure-check mode (iv) -- a pre-filter-EXCLUDED unit that declares")
    lines.append("# mutable state at the current pin:")
    violations = [u for u in PREFILTER_EXCLUDED_UNITS if u in verdict and verdict[u][0]]
    stale = [u for u in PREFILTER_EXCLUDED_UNITS if u not in verdict]
    newly = [u for u in computed_excluded if u not in PREFILTER_EXCLUDED_UNITS]
    if violations:
        for u in violations:
            lines.append(f"#   PREFILTER-VIOLATION: {u} -- {verdict[u][1]}")
    if stale:
        for u in stale:
            lines.append(f"#   PREFILTER-STALE: {u} is no longer reached by the walk")
    if newly:
        for u in newly:
            lines.append(f"#   PREFILTER-UNDECLARED: {u} -- add to PREFILTER_EXCLUDED_UNITS")
    if not (violations or stale or newly):
        lines.append("#   none -- every declared exclusion holds at this pin")
    lines.append("#")
    if rejected:
        lines.append("# lexical matches rejected as non-members of their struct:")
        for obj, field in rejected:
            lines.append(f"#   {obj}.{field}")
        lines.append("#")
    lines.append("# TRIAGE -- object<TAB>field<TAB>liveness<TAB>disposition<TAB>reason")
    rows = 0
    untriaged = 0
    for key in live:
        obj, field = key
        if key in ADMITTED_ROUTING_STATE:
            lines.append(f"{obj}\t{field}\tLIVE\tADMITTED\t{ADMITTED_ROUTING_STATE[key]}")
        elif key in EXCLUDED_ROUTING_STATE:
            lines.append(f"{obj}\t{field}\tLIVE\tEXCLUDED\t{EXCLUDED_ROUTING_STATE[key]}")
        elif obj in EXCLUDED_ROUTING_OBJECTS:
            lines.append(f"{obj}\t{field}\tLIVE\tEXCLUDED\t{EXCLUDED_ROUTING_OBJECTS[obj]}")
        else:
            lines.append(f"{obj}\t{field}\tLIVE\tUNTRIAGED\t<-- add to ADMITTED_ROUTING_STATE or EXCLUDED_ROUTING_STATE")
            untriaged += 1
        rows += 1
    lines.append("#")
    lines.append("# KILLED by the prologue -- a PURE WRITE precedes every read, in order.")
    for obj, field in killed:
        lines.append(f"{obj}\t{field}\tKILLED\t-\tprologue writes before any read")
        rows += 1
    lines.append("#")
    lines.append(f"# UNTRIAGED COUNT: {untriaged}")

    # --- D-R6: the TFile exclusion table -------------------------------------
    handles = _tfile_handles(globals_h, objects_h)
    lines.append("#")
    lines.append("# " + "-" * 74)
    lines.append("# D-R6 -- non-field (stream position) state, keyed on the TYPE TFile")
    lines.append("# " + "-" * 74)
    lines.append(f"# Declaration sites found by type: {len(handles)} handle(s)")
    lines.append("# handle<TAB>site<TAB>disposition<TAB>reason")
    dr6_untriaged = 0
    for name, site in handles:
        if name in TFILE_ADMITTED:
            lines.append(f"{name}\t{site}\tADMITTED\t{TFILE_ADMITTED[name]}")
        elif name in TFILE_EXCLUDED:
            lines.append(f"{name}\t{site}\tEXCLUDED\t{TFILE_EXCLUDED[name]}")
        else:
            lines.append(f"{name}\t{site}\tUNTRIAGED\t<-- add to TFILE_ADMITTED or TFILE_EXCLUDED")
            dr6_untriaged += 1
        rows += 1
    lines.append("#")
    lines.append(f"# D-R6 ADMITTED: {len(TFILE_ADMITTED)} (an empty admitted set is a PASSING result)")
    lines.append(f"# D-R6 UNTRIAGED COUNT: {dr6_untriaged}")
    return rows


def _members_of_global(objects_h: str, globals_h: str, obj: str) -> set[str]:
    """Struct members of ``obj``'s type, or an empty set when not struct-typed."""
    m = re.search(rf"EXTERN\s+(T[A-Za-z0-9_]*)\s*\**[^;]*\b{re.escape(obj)}\b", globals_h)
    typ = m.group(1) if m else None
    if typ is None:
        m2 = re.search(rf"^static\s+(T[A-Za-z0-9_]*)\s*\**\s*{re.escape(obj)}\b", objects_h, re.M)
        typ = m2.group(1) if m2 else None
    if typ is None:
        return set()
    try:
        return set(struct_fields(objects_h, typ))
    except SystemExit:
        return set()


def _tfile_handles(globals_h: str, objects_h: str) -> list[tuple[str, str]]:
    """Every ``TFile`` the TYPE-keyed enumeration returns.

    Keyed on the type and never on a declaration site: enumerating from the
    globals block alone silently misses the per-time-series cursor, which is the
    same failure shape as a hand-written field list missing a module static.
    Keying on the type stays complete when upstream adds a twelfth handle.
    """
    out: list[tuple[str, str]] = []
    m = re.search(r"^EXTERN\s+TFile\b(.*?);", globals_h, re.S | re.M)
    if m:
        # Strip the trailing comment from EVERY line before splitting on commas.
        # Splitting first leaves each chunk as `<comment>\n<next name>`, whose
        # text BEFORE the `//` is whitespace -- so a comment-per-line block
        # yields exactly one name, the first, and the enumeration silently
        # returns 1 of 11 while looking like it worked.
        blob = "\n".join(ln.split("//")[0] for ln in m.group(1).splitlines())
        for raw in blob.split(","):
            name = raw.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                out.append((name, "globals.h EXTERN TFile block"))
    for mm in re.finditer(r"^\s*TFile\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", objects_h, re.M):
        owner = objects_h[: mm.start()]
        typ = re.findall(r"\}\s*(T[A-Za-z0-9_]*)\s*;", objects_h[mm.end():])
        owner_name = typ[0] if typ else "?"
        out.append((f"{owner_name}.{mm.group(1)}", f"objects.h member of {owner_name}"))
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

    # --- Criterion P: the routing half of the same artifact ------------------
    p_rows = emit_p_section(solver, lines)
    lines.append("#")
    lines.append(f"# CRITERION P ROWS: {p_rows}")
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

---
type: verification-design
title: Snapshot inventory and closure checks
description: How source-derived report and routing inventories, explicit triage, serializer manifests, and mutation controls jointly guard coupled snapshot completeness.
tags: [testing, swmm, snapshots, inventory]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-38bc7820992350740b3f6d1c
    resource: repo://external/swmm/src/solver/snapshot.c
  - id: openwiki-source-83cd1e3fd74a62467366891d
    resource: repo://test/inventory/CMakeLists.txt
  - id: openwiki-source-d6984c93d926b9621612c12d
    resource: repo://test/inventory/regenerate_snapshot_inventory.py
  - id: openwiki-source-00d2b7de9aba3fcc5bd415f0
    resource: repo://test/inventory/snapshot_inventory.txt
  - id: openwiki-source-0b10da7888a2bce763889ee3
    resource: repo://test/snapshot/CMakeLists.txt
  - id: openwiki-source-ae0d0317106a831f74289ee5
    resource: repo://test/snapshot/test_state_snapshot.cpp
  - id: openwiki-source-a03b101ec860088174ea1000
    resource: repo://test/test_snapshot_inventory_closure_modes.py
  - id: openwiki-source-5a3e33c1d6c6fc2379101a5b
    resource: repo://test/test_snapshot_inventory_criterion_p.py
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Snapshot inventory and closure checks

## Responsibility and independent checks

`test/inventory/regenerate_snapshot_inventory.py` produces the committed `snapshot_inventory.txt`. The generator owns source discovery, classification, deterministic text emission, and a check-only verdict. Its explicit admission/exclusion tables carry review decisions; the generated inventory is not a file to edit by hand.

Two independent relationships matter:

1. **Source to inventory:** regeneration/check detects an inventory that no longer represents the generator's operation over the current solver source, and rejects current artifacts with open triage.
2. **Inventory to serializer:** `UNIT-SNAPSHOT-T6_manifest_covers_committed_inventory` builds the serializer's actual manifest and checks required and refused names against that committed record.

Agreement in the second relationship cannot establish the first: serializer and record can agree while both omit a new source quantity. Conversely, a current record does not prove the serializer implements its admissions.

Evidence: [generator entrypoint](../../test/inventory/regenerate_snapshot_inventory.py#L2162-L2386), [manifest test](../../test/snapshot/test_state_snapshot.cpp#L582-L761).

## Criterion R: report reachability and declared extent

Criterion R asks which quantities a report-writing path reads. The operation seeds `stats_report`, `statsrpt_writeReport`, and `massbal_report`, closes calls at function granularity within the declared report-bearing units, and fails if a seed is missing. It records calls that cross that boundary rather than implying those callees were analyzed.

For selected serialized objects, struct bodies define field extent. Field extraction handles comma-separated declarators rather than retaining only the last name. References reached directly or through propagated pointer-parameter bindings determine the `read_by_report_path` annotation. A separate discovery pass scans declared global scalars for report-path references absent from both scalar admission and exclusion tables.

The emitted R section is a **coverage/extent record**, not just a filtered list of `yes` fields: it lists all declared fields of the selected objects, with `yes`/`no` read annotations, followed by scalar rows. T6 requires every R row, including `no` rows, in the manifest. This distinction matters when interpreting a row such as an accumulator captured for completeness despite no direct report-path match.

Output ordering is stable through sorted object/function/boundary iteration and declaration-order fields. This is deterministic text construction, not a claim that regeneration was executed here.

Evidence: [field extraction](../../test/inventory/regenerate_snapshot_inventory.py#L1180-L1219), [report closure, aliases, discovery, and emission](../../test/inventory/regenerate_snapshot_inventory.py#L2172-L2314), [committed R format](../../test/inventory/snapshot_inventory.txt#L1-L34), [R membership in T6](../../test/snapshot/test_state_snapshot.cpp#L660-L730).

## Criterion P: state live at a step boundary

Report state alone cannot describe everything needed to continue routing. Criterion P asks whether a step beginning at the `swmm_step` boundary reads a quantity before overwriting it. Its routing discovery roots are `routing_execute`, `routeFlow`, `flowrout_execute`, and `dynwave_execute`; the call closure is not restricted to a fixed translation-unit list.

The candidate bases differ deliberately:

- Struct-field candidates come from the routing closure in contributing units.
- Scalar candidates come from the wider `swmm_step` event stream and are inserted **before** validation. A clock read outside the routing-root closure can therefore still become a candidate.

A mutable-state prefilter determines which reached units contribute declarations: non-const file-scope statics, defined globals, or writes through globals qualify. It does not remove those units' callees from the walk. A committed exclusion declaration is compared with freshly computed verdicts; changes produce explicit prefilter diagnostics.

Per-unit visibility limits static-name events to the owning translation unit while globals stay visible across units. The emitted scope census identifies duplicated static names or static/global shadowing because a bare-name row would otherwise conflate distinct C objects. Struct-member validation removes lexical matches that do not name real members where that extent is available.

Evidence: [unrestricted closure](../../test/inventory/regenerate_snapshot_inventory.py#L1504-L1545), [candidate construction and scope](../../test/inventory/regenerate_snapshot_inventory.py#L1774-L2004).

### Why kill order is load-bearing

The event stream inlines callees at textual call positions from `swmm_step`, with recursion guarded along the current path. `kill_pass` classifies a candidate as killed only when its first decisive event is a pure write in a declared prologue function. A read wins if it occurs first; `+=`, increments, and other read-modify-writes count as reads. Pure writes outside the prologue do not establish a kill, since they may be conditional.

The Criterion-P regression explicitly compares this ordered pass with an intentionally unsound flat-write-set pass. `Node.inflow`, `Node.outflow`, and `Node.newLatFlow` must remain live under the ordered operation and be killed by the flat control. Other checks cover early `Node.overflow`, routing clocks/sentinels, prologue order, field-parser extent, scalar-row shape, and scalar serializer coverage.

The declared prologue order is also compared with the order the stream reaches its functions; order violations and unreached functions become visible in generated text.

Evidence: [events and kill pass](../../test/inventory/regenerate_snapshot_inventory.py#L1548-L1717), [emitted prologue diagnostics](../../test/inventory/regenerate_snapshot_inventory.py#L1939-L1958), [ordered/flat controls](../../test/test_snapshot_inventory_criterion_p.py#L139-L235), [scalar checks](../../test/test_snapshot_inventory_criterion_p.py#L362-L458).

## Triage and stream-position axis

Live P rows consult `ADMITTED_ROUTING_STATE`, then explicit field exclusions, then whole-object exclusion rules; remaining candidates become `UNTRIAGED`. Killed rows are emitted separately. Reasons distinguish persisted state from configuration re-read from the model and derived/reinitialized quantities. Whole-object rules still emit each discovered member, so they do not hide changes to the generated record's extent.

Stream positions have a separate type-keyed `TFILE-TRIAGE` table. Its enumerator scans both global `TFile` declarations and `TFile` members in object types, rather than assuming all streams are globals. These are handle decisions, not `Object.field` payload entries. In particular, the committed `Fout` row admits continuity without storing a raw file offset: the recorded design derives output position from restored `Nperiods` and output-layout information. That row does not add a stream-position field to the snapshot payload.

Evidence: [triage emission](../../test/inventory/regenerate_snapshot_inventory.py#L2011-L2055), [type-keyed enumeration](../../test/inventory/regenerate_snapshot_inventory.py#L2073-L2099), [committed handle decisions](../../test/inventory/snapshot_inventory.txt#L657-L676).

## Serializer manifest contract

`snapshot.c` shares one traversal between manifest, write, read, and count modes. Names are `Object.field`; singleton scalar names use `Name.value`. Manifest mode uses dummy storage/counts so the manifest is available without an open model. That makes the runtime manifest a stronger connection to serializer implementation than a second hand-maintained field list.

T6 treats the inventory's sections distinctly:

- Every R row and each `ADMITTED` P row belongs to the required set.
- Other P rows belong to the refused set; they are forbidden only when neither criterion requires the same key.
- Stream-handle rows are skipped.
- R scalar rows (`name`, `-`) and P scalar rows (`-`, `name`) normalize differently to the same `Name.value` form.

The test reports missing and overcaptured names and requires nonempty R, admitted-P, and refused-P populations. This guards against a parser that finds nothing and passes set comparisons vacuously. It is not a global equality assertion over arbitrary manifest names: its negative arm specifically checks known refused keys.

Evidence: [shared serializer modes](../../external/swmm/src/solver/snapshot.c#L147-L299), [manifest entrypoint](../../external/swmm/src/solver/snapshot.c#L823-L838), [section-aware reconciliation](../../test/snapshot/test_state_snapshot.cpp#L592-L761).

## Check mode, regeneration, and failure meanings

The generator requires `--solver-dir`; `--inventory` overrides the default file beside the script. Adding `--check` computes output without writing it.

| Result | Meaning | Correct maintenance action |
|---|---|---|
| `EXIT_DRIFT` / 1 | Missing inventory or generated text differs; a unified diff is printed for differences. | Regenerate without `--check`, review source/record changes, and reconcile serializer coverage. |
| `EXIT_UNTRIAGED` / 3 | Text matches, but report scalar, routing, stream, or scalar-scope decisions remain open. | Resolve the named decision tables or row-key ambiguity, then regenerate and check again. |
| `EXIT_OK` / 0 | Text matches and the emitted open-decision axes are closed. | Continue other independent checks; this is not a complete runtime-resume proof. |

Drift is evaluated first when both conditions hold. Open counts are read from generated summaries, and a missing required counter is treated as an open decision rather than silently as empty. Writing a regenerated artifact by itself is not acceptance: non-check mode writes and returns success without requiring triage closure.

Evidence: [verdict constants and open-axis parsing](../../test/inventory/regenerate_snapshot_inventory.py#L2102-L2159), [CLI and verdict implementation](../../test/inventory/regenerate_snapshot_inventory.py#L2162-L2386).

## Test tiers and guard-of-the-guard

Under `BUILD_TESTS`, the inventory CMake file registers fast `SOURCE-SNAPSHOT-INVENTORY-CRITERION-P` and `SOURCE-SWMM-GUARD-PREDICATE-CLOSURE` checks, plus slow-labeled regeneration and closure-mode mutation checks. From a configured build, `ctest -LE slow` excludes that slow tier and `ctest -L slow` selects it. The manifest test belongs to the coupled snapshot target and receives the committed inventory path through `TRITON_SNAPSHOT_INVENTORY`.

Closure-mode tests stage throwaway copies, mutate the operation or source, and require the expected diagnostic or missing sentinel row. They cover lost triage, weakened object rules/prefiltering, added state, reordered prologue, parser truncation, scalar-basis ordering, and comment contamination. Separate differential cases regenerate first to isolate verdict behavior: current-but-open must return the open-decision code, current-and-closed must succeed, and deliberately drifted text must return drift. A red result for an unrelated reason does not satisfy the named controls.

Comments refer to an upstream-bump process document; this page does not establish that document's existence or prescribe an upstream contribution workflow. The verified repository entrypoints here are the generator, CTest registrations, and test scripts themselves.

## Limits and verification status

This is a regex/brace/event parser over source conventions, **not a C compiler or path-sensitive formal analyzer**. Report closure is explicitly bounded; routing closure can follow several same-named definitions, while event streaming selects a body by name and follows textual call order rather than evaluating branches. Changes in declarations, macros, aliasing, or call structure require checking the instrument as well as its output.

The generator's full-tree inputs were described from its allowed code only. Excluded solver inputs were not opened, and neither the generator nor tests were executed to bypass `.openwikiignore`. No regeneration or runtime passing result is claimed for this session.

- [SWMM state snapshots](../architecture/swmm-state-snapshots.md)
- [Regression and resume verification](regression-and-resume.md)
- [Checkpoint and resume](../workflows/checkpoint-and-resume.md)

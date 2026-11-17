---
type: architecture
title: SWMM snapshot format and output continuity
description: Full-precision SWMM state serialization, compatibility checks, publication markers, and the binary-output positioning required to continue a restored run.
tags: [swmm, snapshots, serialization, restart, output]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-a81b440996ab692a2dfa4cb7
    resource: repo://external/swmm/src/solver/output.c
  - id: openwiki-source-38bc7820992350740b3f6d1c
    resource: repo://external/swmm/src/solver/snapshot.c
  - id: openwiki-source-c26b636827b23cd5791d2bcb
    resource: repo://src/swmm_triton.h
  - id: openwiki-source-cb1ebd39e98464de24c55af6
    resource: repo://test/snapshot/test_quality_snapshot_review.cpp
  - id: openwiki-source-ae0d0317106a831f74289ee5
    resource: repo://test/snapshot/test_state_snapshot.cpp
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# SWMM snapshot format and output continuity

`external/swmm/src/solver/snapshot.c` is TRITON-owned serialization inside the bundled solver. It captures report accumulators and routing state at the coupled step boundary, avoiding replay of the entire exchange prefix. This is not merely a depth hotstart: continuation also needs clocks, cross-step state, quality arrays, and the binary-output period count.

## Format and compatibility boundary

The **writer implementation**, rather than its introductory layout comment, defines this order:

| Region | Contents |
|---|---|
| Header | Four native C `int` words: magic `0x534E5331`, version `7`, `sizeof(double)`, manifest byte length |
| Shape | Twelve integer dimensions: nodes, links, subcatchments, pollutants, storage nodes, outfalls, pumps, critical-stat slots, timestep levels, flow classes, conduits, dynamic-wave extended nodes |
| Manifest | Ordered newline-separated `Object.field` names, without the terminating NUL |
| Payload | Field values stored individually as `double`, including integer/long/boolean state converted through double |

The reader requires exact magic, version and value width, then compares every shape dimension and the manifest against the current serializer. It distinguishes a genuinely absent file (`ENOENT`) from an existing/unopenable or incompatible file. Older versions are refused, not migrated. Although the introductory comment calls the format little-endian/int32, the implementation uses native `fwrite` representations without endian conversion: do not infer cross-endian portability.

Sources: [format constants and shape](../../external/swmm/src/solver/snapshot.c#L62-L145), [slot representation](../../external/swmm/src/solver/snapshot.c#L194-L267), [actual write order](../../external/swmm/src/solver/snapshot.c#L946-L955), [reader checks](../../external/swmm/src/solver/snapshot.c#L1044-L1150).

## One traversal, multiple uses

`snapshot_traverse` drives manifest generation, payload writing, payload reading, and expected-value counting. Manifest mode substitutes dummy objects and representative dimensions, so field descriptions can be produced without an open model. Count mode uses the live model and executes structural guards without writing a file. Save compares the write traversal's visited value count with that separate count pass and checks stream close before publication.

Named field access catches removed/renamed members during compilation. Size assertions catch growth of selected statistics structures, but explicitly cannot detect additions that fit existing padding. Inventory regeneration and manifest/inventory comparison close a different boundary; see [snapshot inventory](../testing/snapshot-inventory.md).

The loader measures payload extent to report expected/actual values when traversal fails. It does not first enforce an exact file-length equality or checksum, and reading writes directly into live state. A failed load is not a rollback transaction and must not be treated as untouched state safe for replay.

Sources: [traversal modes](../../external/swmm/src/solver/snapshot.c#L147-L169), [dummy selection](../../external/swmm/src/solver/snapshot.c#L269-L319), [manifest/count entrypoints](../../external/swmm/src/solver/snapshot.c#L823-L855), [save validation](../../external/swmm/src/solver/snapshot.c#L910-L995), [size guard boundary](../../external/swmm/src/solver/snapshot.c#L85-L111), [load traversal](../../external/swmm/src/solver/snapshot.c#L1152-L1183).

## What survives the step boundary

The traversal includes:

- Node/link/storage/outfall/pump/subcatchment statistics; timestep statistics and critical-stat arrays.
- Runoff, groundwater, flow and quality mass-balance totals, per-node cumulative inflow/outflow, and report-bearing scalar counters.
- Node/link/conduit/storage/outfall routing state, dynamic-wave `Xnode.oldSurfArea` and `dYdT`, routing/report/rule clocks, event state, `BetweenEvents`, and `VariableStep`.
- `Nperiods`, which startup resets but restored output offsets and final reports require.
- Version-7 quality state: node/link `oldQual` and `newQual`, link `totalLoad`, and each pollutant's `StepQualTotals` row.

Private statistics and routing/dynamic-wave state is reached through narrow address-returning accessors; ownership stays in the defining C modules. Input-only state such as node `surDepth` is deliberately not restored over current configuration.

Payload length must be determined by recorded shape, not unrecorded switches. Quality arrays are emitted regardless of `IgnoreQuality`. Every outfall reserves a pollutant-width `wRouted` section; route-less outfalls use zero placeholders on save and discard slots on load. A routed outfall missing its required array is a structural refusal rather than permission to shorten the payload.

Sources: [statistics and totals](../../external/swmm/src/solver/snapshot.c#L322-L619), [node/link state](../../external/swmm/src/solver/snapshot.c#L628-L704), [remaining routing state and clocks](../../external/swmm/src/solver/snapshot.c#L706-L818), [statistics accessor](../../external/swmm/src/solver/stats.c#L102-L126), [routing accessor](../../external/swmm/src/solver/routing.c#L1021-L1037), [dynamic-wave accessor](../../external/swmm/src/solver/dynwave.c#L117-L149).

## Publication and diagnostic markers

The writer targets `{snapshot}.part`, closes and validates the traversal, then renames it to the published snapshot name. A structural count-pass refusal attempts to create an exclusive `.failed` text marker; a payload/close/count failure attempts to retain the temporary as `.failed`. Other failures, including some allocation/open/header/publication failures, may leave no marker. The live simulation warns and continues when a snapshot write fails.

On restore, a published snapshot takes precedence over same-checkpoint markers. Without a published file, surviving `.part`, `.killed`, or `.failed` is not ordinary absence: `.part` is renamed `.killed` where possible, and the wrapper classifies the interrupted/failed write. A `.failed` marker normally stops; `swmm_snapshot_failed_replay=1` allows its explicit replay route, while cut-off writes retain their step-back classification. Repeated consultation still recognizes `.killed` instead of accidentally falling into the absent-file route.

**Durability limit:** this is temporary-file publication for process interruption, not a power-loss guarantee. The serializer explicitly performs no `fsync`; atomic rename behavior on the target filesystem is assumed, not measured. `output_flush` likewise performs a userspace flush, not filesystem synchronization.

Sources: [write and marker paths](../../external/swmm/src/solver/snapshot.c#L859-L1012), [wrapper warning/retention](../../src/swmm_triton.h#L1774-L1824), [marker precedence](../../src/swmm_triton.h#L1833-L1901), [restore result contract](../../src/swmm_triton.h#L598-L620), [output flush](../../external/swmm/src/solver/output.c#L674-L685).

## Binary output is part of restoration

Preserving state without preserving its output prefix would produce incomplete reports. `output_openOutFile` attempts `r+b` when an existing file can be probed and falls back to `w+b` if it cannot open non-truncating. On a successful reopen, `output_open` renders the current prologue into a scratch stream, compares it byte-for-byte with the old prologue, refuses mismatch, and otherwise leaves the real stream at `OutputStartPos` without rewriting that prefix. This is prologue compatibility, not a hash of every input or payload value.

Before snapshot save, the wrapper calls `output_flush`, intending to put all output periods on disk before persisting their count. That function does not return/check a flush error; later restore checks are still necessary.

After snapshot load, the wrapper exports coupled node depths through the routing writer, positions exchange history without replaying it, and calls `output_positionForResume`. That function measures file length, computes how many complete fixed-width periods it can hold, and refuses when restored `Nperiods` exceeds that capacity. It seeks to `OutputStartPos + Nperiods * BytesPerPeriod`, truncates the stale/torn tail, and reasserts the stream position. EOF is not a valid substitute for the checkpoint offset: an earlier checkpoint can precede a longer prior execution.

If output positioning fails after state restoration, the wrapper returns `OutputMismatch`; it does not replay on top of restored accumulators. At normal finalization, `output_end` writes the closing offsets/count/error/magic trailer and truncates to the payload end plus six integer trailer words. This makes the new trailer the physical end even when the previous execution was longer.

Sources: [non-truncating open](../../external/swmm/src/solver/output.c#L512-L541), [prologue comparison](../../external/swmm/src/solver/output.c#L439-L463), [comparison implementation](../../external/swmm/src/solver/output.c#L640-L669), [snapshot flush](../../src/swmm_triton.h#L1774-L1786), [restore handoff](../../src/swmm_triton.h#L2032-L2059), [positioning/truncation](../../external/swmm/src/solver/output.c#L611-L762), [final trailer](../../external/swmm/src/solver/output.c#L767-L800).

## Evidence and test boundaries

`test_state_snapshot.cpp` contains direct state tests, including clobber-before-restore bit comparisons, shape/magic refusal, period-count restoration, short-output refusal, final-trailer truncation, payload-shape invariance, and interrupted/failed-write classification. The independent quality review fixture exercises routed and route-less outfalls, storage residence time, distinct pollutant values, and emission under both `IgnoreQuality` settings. These are focused serializer/wrapper checks, not substitutes for an end-to-end coupled resume with a real model and binary-output readers.

Sources: [round-trip method](../../test/snapshot/test_state_snapshot.cpp#L183-L287), [output test boundaries](../../test/snapshot/test_state_snapshot.cpp#L837-L857), [quality review and its assertions](../../test/snapshot/test_quality_snapshot_review.cpp#L3-L23), [quality review execution](../../test/snapshot/test_quality_snapshot_review.cpp#L202-L244).

For recovery procedures use [checkpoint and resume](../workflows/checkpoint-and-resume.md). For the runtime exchange boundary use [SWMM coupling](../integrations/swmm-coupling.md); for campaign-level proof use [regression and resume testing](../testing/regression-and-resume.md).

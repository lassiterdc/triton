---
type: workflow
title: Checkpoint artifacts and coupled resume
description: Select and preserve a surface or coupled checkpoint, understand snapshot-versus-replay routing, and interpret interrupted-write, retention, and clean-start policies.
tags: [checkpoint, restart, swmm, operations]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-34648547d88511ebf4f3016d
    resource: repo://src/config_utils.h
  - id: openwiki-source-fbe082bc4fe57162918ea012
    resource: repo://src/output.h
  - id: openwiki-source-c26b636827b23cd5791d2bcb
    resource: repo://src/swmm_triton.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-47ce21353e1f2c4629fd35a5
    resource: repo://test/coupled_resume_suite.sh
  - id: openwiki-source-d08c570a9c098cb2385720da
    resource: repo://test/unit/test_quality_resume.py
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Checkpoint artifacts and coupled resume

## Operator entrypoint and prerequisites

Pass the saved `cfg/config_{checkpoint}.cfg` as the executable's configuration argument, as the integration tests do. The saved cfg carries `checkpoint_id`, `sim_start_time`, `time_step`, and `it_count` updates when those keys exist in the original cfg. The writer replaces matching existing lines rather than constructing a complete schema or adding every absent key; use a restart-capable original cfg with explicit checkpoint/time fields. A nonzero start time alone is not the checkpoint restore trigger: loaders and coupled restore branch on `checkpoint_id > 0`.

Preserve the original inputs, build precision, coupling declaration/order, output layout, and rank decomposition. The executable derives `project_dir` by making its invocation path absolute and stripping two path components; it does not resolve symlinks. Output and SWMM paths in these checkpoint paths concatenate `project_dir` with `output_folder`. A standalone copy therefore needs the same coherent project layout, or the temporary project/bin symlink arrangement used by the tests, plus a relative output folder. Do not assume an absolute output folder is honored consistently by these paths.

For a copied output tree, update its saved cfg's `output_folder` to the copy before launch. Resume mutates the exchange side-file by trimming its tail and reopening it for append; archive an untouched tree separately if the original must remain evidence. The quality test and standalone suite both clone outputs before changing this key.

Evidence: [configuration selection](../../src/triton.h#L631-L650), [saved cfg writer](../../src/output.h#L1503-L1564), [root derivation](../../src/config_utils.h#L608-L645), [coupled paths](../../src/swmm_triton.h#L1091-L1149), [copy-and-resume test](../../test/unit/test_quality_resume.py#L324-L356).

## Preserve the checkpoint set, not just its cfg

| Artifact | Resume role |
|---|---|
| `cfg/config_{id}.cfg` | Selects checkpoint/time and retains the run's remaining settings. |
| Binary `H`, `QX`, `QY` rasters | Restores surface depth and unit discharge; loaders use fixed checkpoint/rank filename conventions. |
| Binary `MH` rasters when maximum-depth output is enabled | Restores the historical `MAXH` accumulator. Without that option, the accumulator is initialized from restored depth instead. |
| Per-rank binary `GR` side-files | Restores the exact H/QX/QY ghost ring, including runs whose ordinary rasters use SEQ output. |
| `domain_decomposition/domain_decomposition{id}.txt` for multi-rank dynamic decomposition | Restores rank row counts before partitioning. |
| `swmm/{model}_exchange_replay.bin` for a coupled run | Stores the full exchange prefix and remains required even on the snapshot route. |
| `swmm/{model}_state_cp{id}.snapshot` when retained | Restores SWMM state directly, avoiding repeated SWMM stepping over the prefix. |
| Existing SWMM binary output | Carries the pre-checkpoint output periods for snapshot-based output continuity. |

Raster emission is controlled by `print_option`: `h`, `u`, and `v` select H, QX, and QY. The checkpoint readers still load all three. Omitting a selected field can therefore leave a cfg that is not a complete restart set. `max_value_print_option` must match the preserved MH state. ASC and GDAL-enabled GTIFF output also emit binary companions for selected fields; those text/image products do not replace the binary checkpoint state.

Use the conventional `outfile_pattern` compatible with the fixed raster readers. The writer formats `root/subdir/field/checkpoint/rank` arguments through that pattern, whereas loaders explicitly construct names such as `H_01_00.out`. An arbitrary output naming pattern is not proven restart-compatible.

Evidence: [sequential reads](../../src/triton.h#L849-L888), [parallel reads and MAXH fallback](../../src/triton.h#L1683-L1774), [output selection](../../src/output.h#L525-L704), [formatter](../../src/output.h#L1423-L1437).

## Rank decomposition and the ghost ring

SEQ rasters are restored globally and scattered; PAR rasters are read by rank with local dimensions. Dynamic multi-rank restart first reads the saved partition table on rank zero and broadcasts row counts. That parser indexes the supplied row array by saved rank IDs without establishing a complete rank-count compatibility check. Preserve the writing rank count and partition rather than relying on a migration/refusal guarantee.

Ghost rings make the same-rank/layout requirement relevant even for SEQ rasters. `GR` paths include the current rank; the payload records local rows and columns followed by boundary H/QX/QY values. Missing files, mismatched dimensions, or a truncated final read are fatal. Before the first resumed flux evaluation, the solver replaces reconstructed ghosts with these persisted values and pushes H/QX/QY back to the device. The source explains why reconstructing ghosts from interior checkpoint values is insufficient: boundary state may precede later wet/dry momentum changes.

Dynamic partition metadata is written for the checkpoint before a subsequent rebalance. A checkpoint thus records the decomposition that wrote its state, not the next decomposition after balancing. See [parallel execution](../architecture/parallel-execution.md).

Evidence: [partition restore](../../src/triton.h#L400-L424), [partition parser](../../src/config_utils.h#L649-L679), [ghost I/O](../../src/output.h#L796-L882), [ghost device restore](../../src/triton.h#L2227-L2245), [partition publication order](../../src/triton.h#L2351-L2361).

## Checkpoint publication order

At a print boundary the solver copies surface state to host and synchronizes. For coupled runs rank zero flushes the exchange log **before** `out.write_output` can publish the new saved cfg. Append or flush failure aborts the communicator rather than continuing with later checkpoint cfgs above an incomplete exchange prefix. This is buffered-stream flushing, not a documented fsync/transaction guarantee for the whole checkpoint set.

`out.write_output` writes selected rasters and each rank's ghost ring, then rank zero writes the saved cfg and current-directory `cid`. SWMM snapshot writing follows. It flushes SWMM binary output before saving the snapshot, so its restored output-period count refers to flushed output bytes. Snapshot save failure warns and lets the live simulation continue; the surviving marker controls the next resume policy. A cfg's existence therefore does not establish that its snapshot was published, or that every other artifact is usable.

Evidence: [checkpoint ordering](../../src/triton.h#L2300-L2335), [cfg publication](../../src/output.h#L704-L716), [exchange stream failures](../../src/swmm_triton.h#L1411-L1453), [snapshot writer](../../src/swmm_triton.h#L1774-L1812).

## Coupled restore: snapshot first, exchange history always

Rank zero initializes SWMM, then attempts the selected snapshot for a positive checkpoint ID. A successful snapshot load exports coupled node depths, scans the exchange log without replaying SWMM steps, trims to the checkpoint prefix, and positions SWMM binary output using the restored period count. Output-position failure cannot safely fall back on top of restored accumulators and is refused.

An absent snapshot takes exchange replay: each stored `(dt, exchange_q[])` record is passed through `swmm_step` to rebuild SWMM to the checkpoint. This cost grows with the prefix. Both paths still scan the exchange series; snapshot restore avoids SWMM stepping, not every operation proportional to history length.

The V2 exchange header contains magic, node count, and stored `value_t` width. Validation checks completeness/magic before node count before width, rejecting legacy V1 rather than interpreting its third field as a width. Records must cover the requested time within the positioning tolerance. Missing files, incompatible headers, or incomplete history stop the operation. The accepted prefix is retained and any later/partial tail is removed before append. Header checks do not establish equality of all model settings or the identity/order of same-count coupling nodes; preserve those inputs rather than treating equal counts as a model fingerprint.

After either successful route, rank zero reorders global node depths and every rank participates in `MPI_Scatterv`, including ranks owning no coupled node. This happens before device state construction. It prevents the first resumed exchange from using initialization's zero-depth sewer values. Whole-model restore guards use the global count, not rank zero's local count.

Evidence: [restore orchestration and depth scatter](../../src/triton.h#L453-L521), [header contract](../../src/swmm_triton.h#L72-L140), [history scan/replay/truncation](../../src/swmm_triton.h#L1603-L1735), [snapshot export and positioning](../../src/swmm_triton.h#L2000-L2080).

## Snapshot decision table

All snapshot policy keys below default to false in configuration parsing. Decisions apply to the selected checkpoint; the executable does not silently load a different checkpoint cfg.

| Observed state or policy | Executable behavior | Operator action |
|---|---|---|
| Snapshot exists and loads; output positioning succeeds | Continue with `path=snapshot`. A same-ID published file takes precedence over failure markers. | Confirm completion and expected artifacts. |
| Snapshot absent with no live write marker | Full replay, with classified missing reason. | Preserve a covering exchange side-file and compatible inputs. |
| `swmm_snapshot_disable=1` | Empty snapshot stem disables writing/restoring and yields replay with reason `absent`; snapshot cleanup is also skipped. | Accept replay cost explicitly; this is not diagnosis or repair of an earlier failure. |
| Snapshot present but unreadable/incompatible | Refuse; no automatic replay. | Inspect the named disagreement and use a coherent clean run rather than relabeling the file as absent. |
| Snapshot loads but SWMM output cannot carry continuation | Refuse with output mismatch. | Restore the matching output set or clean-start; replaying over already-restored state would double-apply history. |
| No published snapshot; `.failed` survives | Default stop even if an earlier snapshot exists. | Inspect/clear the write failure cause; the marker may hold a refusal line or failed snapshot bytes. |
| Same `.failed`, plus `swmm_snapshot_failed_replay=1` | Full replay with `reason=write-failed`; marker remains. | Explicit opt-in only; it does not fix a full disk or structural model defect. |
| No published snapshot; `.part` or `.killed` survives | Rename `.part` toward `.killed`, refuse this launch, identify the highest earlier published snapshot if any. | Relaunch its saved cfg; if none exists, clean-start. The failed-write replay opt-in does not bypass this interrupted-write branch. |

A `.failed` marker takes precedence over interrupted-write markers when no published snapshot exists. Machine-readable refusal lines distinguish `path=step-back`, `path=stop`, and `path=clean-start`, with checkpoint/target fields. These are instructions to orchestration or the operator: the refusing invocation aborts instead of automatically restarting from the target. A subsequent successful earlier-snapshot launch may emit `path=step-back-taken` naming a higher live marker.

Evidence: [policy defaults](../../src/config_utils.h#L531-L534), [restore classifier](../../src/swmm_triton.h#L1827-L2080), [orchestrator abort](../../src/triton.h#L466-L501).

## Retention and clean-start behavior

After a successful snapshot save at N, default pruning attempts to remove N-2, retaining the current and immediately previous snapshots in a normal contiguous series. `swmm_snapshot_keep_all=1` skips this pruning; it is separate from clean-start retention. Missing-snapshot labels are based on directory shape: no published series gives `pre-snapshot-checkpoint`, a higher snapshot gives `retention-collision`, otherwise `absent`. These labels are classifications, not proof of a historical cause.

On a clean start, enabled snapshot machinery disposes of the earlier run's published snapshots and live markers before opening a fresh exchange log. Default policy removes published snapshots and interrupted markers but preserves `.failed` evidence by renaming it. `swmm_snapshot_keep_previous=1` instead renames all recognized artifacts to `.prev{k}`, choosing one suffix above every existing such suffix in the directory. Retained names are excluded from active snapshot/marker parsing.

Failure to dispose of a published snapshot refuses the clean start because a later resume could otherwise load stale state. Marker disposal failure warns and continues because markers are not loaded as snapshots. This cleanup is not an archive policy for all rasters, cfgs, reports, or exchange logs; preserve an entire desired run separately.

Evidence: [pruning](../../src/swmm_triton.h#L1815-L1823), [name parsing and disposal](../../src/swmm_triton.h#L1470-L1571), [missing classification](../../src/swmm_triton.h#L2084-L2113), [clean-start hook](../../src/triton.h#L492-L501).

## Verification and related pages

The standalone resume suite documents same-rank replay/snapshot legs and compares binary rasters, replay series, and SWMM binary output only after a completion marker. The quality integration adds regenerated-output and active-pollutant controls. Neither means arbitrary output suppression, rank changes, model edits, or a partially copied tree guarantees exact replay. No simulation was executed while generating this page, and no excluded binary output was opened.

- [SWMM state snapshots](../architecture/swmm-state-snapshots.md) — serializer layout and compatibility internals.
- [SWMM coupling](../integrations/swmm-coupling.md) — node mapping and live exchange.
- [Parallel execution](../architecture/parallel-execution.md)
- [Regression and resume verification](../testing/regression-and-resume.md)
- [Outputs and diagnostics](outputs-and-diagnostics.md)

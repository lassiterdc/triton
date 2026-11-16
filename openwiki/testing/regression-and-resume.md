---
type: testing-guide
title: Regression, coupling, and resume verification
description: Build gates, fixture ordering, focused CTest entrypoints, and standalone coupled-resume checks, including non-vacuity controls and external benchmark requirements.
tags: [testing, ctest, swmm, restart]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-8a69e9dd86720528c61788e4
    resource: repo://cmake/util.cmake
  - id: openwiki-source-d44494ef3e497fea81240ef8
    resource: repo://CMakeLists.txt
  - id: openwiki-source-4b8e7dce368774e92d99ea30
    resource: repo://test/CMakeLists.txt
  - id: openwiki-source-47ce21353e1f2c4629fd35a5
    resource: repo://test/coupled_resume_suite.sh
  - id: openwiki-source-83cd1e3fd74a62467366891d
    resource: repo://test/inventory/CMakeLists.txt
  - id: openwiki-source-0b10da7888a2bce763889ee3
    resource: repo://test/snapshot/CMakeLists.txt
  - id: openwiki-source-cb1ebd39e98464de24c55af6
    resource: repo://test/snapshot/test_quality_snapshot_review.cpp
  - id: openwiki-source-ae0d0317106a831f74289ee5
    resource: repo://test/snapshot/test_state_snapshot.cpp
  - id: openwiki-source-c0c9e3840da0f06d67f05b32
    resource: repo://test/test_swmm_guard_predicate_closure.py
  - id: openwiki-source-067d862a7f99c83abd27847f
    resource: repo://test/test_swmm_return_codes_source.py
  - id: openwiki-source-c7d7d29065ef3bf66ae5cb6c
    resource: repo://test/unit/CMakeLists.txt
  - id: openwiki-source-c82914037a426374077d0a2b
    resource: repo://test/unit/test_coupled_cells_review.cpp
  - id: openwiki-source-d08c570a9c098cb2385720da
    resource: repo://test/unit/test_quality_resume.py
  - id: openwiki-source-56d2799c89ed33060f1c34bf
    resource: repo://test/unit/test_zero_node_refusal.py
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Regression, coupling, and resume verification

## Registration and execution boundary

Configure with `BUILD_TESTS=ON` to enable CTest, add `test/`, and generate `triton_ctest.sh`. That helper sources the build environment and forwards arguments to `ctest`; run it from the build directory. Compilation and usable MPI launch settings remain prerequisites, not consequences of test registration. See [build configuration](../operations/build-and-backends.md).

Within that test tree:

| Gate | Registered coverage |
|---|---|
| Non-ensemble build | PARABOLOID, CIRCULAR-DAMBREAK, ALLATOONA simulations and their `BIT4BIT-*` comparator jobs. |
| `ENSEMBLE_BUILD` | `ENSEMBLE_PARABOLOID` instead of the ordinary simulation/comparator branch. |
| `TRITON_ENABLE_SWMM` | Coupled unit targets and snapshot targets, linked with Kokkos and `swmm5`. Snapshot targets explicitly require MPI CXX. |
| Coupled build with `BACKEND=SERIAL` | `UNIT-ZERO-NODE-REFUSAL` and `UNIT-QUALITY-RESUME`, which launch the actual solver. |
| Every enabled test build | Inventory subtree and `SOURCE-SWMM-RETURN-CODES`; Python-dependent inventory registrations include source closure checks. |

Evidence: [root test gate](../../CMakeLists.txt#L100-L107), [test tree](../../test/CMakeLists.txt), [unit gates](../../test/unit/CMakeLists.txt#L146-L169), [snapshot dependencies](../../test/snapshot/CMakeLists.txt#L7-L19).

## Surface regressions and ensemble comparison

Each ordinary case has a CTest fixture chain: `CLEAN-OUTPUT-{case}` removes its build/test output directory, the simulation requires that clean fixture and establishes its output fixture, and the comparator requires the simulation fixture. This prevents parallel CTest scheduling from comparing stale outputs or running the comparator before its producer. Simulation commands use configured `RUN_COMMAND` and run in the build's `test` directory.

The comparator is registered with `--ref` and `--test` directories. `test/reference/` is excluded by `.openwikiignore`; its implementation and data were not inspected. **The name `BIT4BIT` is not evidence of bitwise comparison semantics.** This page establishes registration and scheduling only.

The ensemble branch runs `triton_ensrun.sh`, requires both selected member QY rasters to be nonempty using `-s`, then compares those files with `cmp -s`. That is a specific member-output equality assertion, not coverage of every ensemble artifact.

Evidence: [simulation and comparator registration](../../test/CMakeLists.txt#L1-L91), [ignore rules](../../.openwikiignore).

## Coupling unit boundaries

The coupled unit registry separates properties into named CTest nodes:

- `UNIT-EXCHANGE-LOG-*`: V2 header layout and round trip, stored-width/node-count mismatch refusals, short or V1 headers, and validation order.
- `UNIT-COUPLED-LIST-*`: parser layouts, junction ordering, FLOW selection, case-insensitive names, startup list validation, report echo, rim rules, and unit conversion. A review target addresses SWMM keyword-prefix section recognition.
- `UNIT-COUPLING-DECLARATION-*`: explicit list, all-junction, legacy FLOW-derived declarations, list-file syntax, and repeated names after case folding.
- `UNIT-COUPLED-CELLS-*`: one-node identity, shared surface cells, draining/filling and indexing. The review target directly evaluates conservation for a several-node cell at the near-zero limiter band, using both two drains and a drain plus surcharged return.

The conservation review computes surface volume change plus summed SWMM exchange volume and compares its magnitude against `1e-12` times initial surface volume plus absolute moved volume. A separate precondition establishes that its chosen timestep really lands in `[0, EPS12)`. It therefore does not pass merely because an arbitrary fixture missed the limiter edge.

Evidence: [unit registry](../../test/unit/CMakeLists.txt#L7-L144), [conservation assertions](../../test/unit/test_coupled_cells_review.cpp#L41-L125).

### Zero-node refusal

`UNIT-ZERO-NODE-REFUSAL` writes a temporary DEM, SWMM model without a coupling FLOW declaration, and cfg, then invokes the current executable through a temporary project/bin symlink using the configured launcher. It requires **both** a nonzero return code and `couples no node` in combined output. An unrelated startup error or a warning followed by successful completion cannot satisfy both conditions. CTest sets a five-second timeout and `RUN_SERIAL=TRUE`; registration is SERIAL-backend-only. Temporary artifacts are removed.

Evidence: [refusal runner](../../test/unit/test_zero_node_refusal.py#L122-L153), [registration](../../test/unit/CMakeLists.txt#L146-L156).

### Water-quality resume and non-vacuity

`UNIT-QUALITY-RESUME` creates an isolated fixture, runs an uninterrupted model, and resumes from checkpoint 2 through both the snapshot path and a forced replay control. Each run must return successfully and print `Simulation ends`; each resume must announce its expected route. It byte-compares the uninterrupted SWMM `.out` files against the resumed files and compares the Quality Routing Continuity, Link Pollutant Load Summary, and Outfall Loading Summary report sections.

Controls prevent an equality result from being only a comparison of copied artifacts:

1. The uninterrupted binary output must contain a positive maximum node pollutant concentration.
2. The comparison requires a non-node SWMM binary, node output, a report, and the named report sections.
3. Before each resume, reports and node outputs are removed from the cloned tree; binary modification stamps are recorded.
4. The resume must recreate comparable outputs and rewrite the binary files, in addition to completing and identifying its route.

CTest gives this SERIAL-only integration node a 120-second timeout. This is stronger evidence than snapshot save/load symmetry alone, but covers the generated fixture rather than every possible SWMM model.

Evidence: [comparison and runtime checks](../../test/unit/test_quality_resume.py#L186-L363), [registration](../../test/unit/CMakeLists.txt#L158-L169).

## Snapshot serializer and refusal coverage

The snapshot registry includes named properties for full-precision state round trips, routing fields, output-period counts, inventory/manifest agreement, payload length, and water-quality state. Compatibility tests corrupt only the magic or version, change the running model shape, and distinguish `SNAPSHOT_OK`, `SNAPSHOT_ABSENT`, and `SNAPSHOT_UNREADABLE`. The precision control checks that the probe value would actually change through float conversion and reads the stored width from a real saved file.

Additional registered cases cover `.part`/`.failed`/`.killed` publication and classification, step-back diagnostics, clean-start disposal, failed-write replay policy, output continuity, and missing quality-array refusal. The missing-array test requires nonzero save status, no published snapshot or `.part`, and the expected `.failed` marker. Independent quality review cases exercise storage and routed/route-less outfalls and check that `IgnoreQuality` does not change payload length or restored quality state.

These are coverage descriptions, not test results from this generation session. See [snapshot architecture](../architecture/swmm-state-snapshots.md) and [checkpoint and resume](../workflows/checkpoint-and-resume.md).

Evidence: [snapshot registry](../../test/snapshot/CMakeLists.txt), [precision and shape checks](../../test/snapshot/test_state_snapshot.cpp#L492-L578), [graduated refusals](../../test/snapshot/test_state_snapshot.cpp#L904-L953), [quality-array refusal](../../test/snapshot/test_state_snapshot.cpp#L2929-L2958), [independent review](../../test/snapshot/test_quality_snapshot_review.cpp#L223-L242).

## Source checks and inventory tiers

`SOURCE-SWMM-GUARD-PREDICATE-CLOSURE` checks rank-zero whole-model guards against the global node-count predicate, prohibits MPI and coupling-object method calls inside local-count blocks, requires those blocks to be walkable, and conversely requires SWMM-indexed host/device vectors to be enclosed by local-count guards. Nonempty-population checks and a scan for indexed accesses in other headers protect the check's scope.

`SOURCE-SWMM-RETURN-CODES` lexically checks `swmm_open`, `swmm_start`, and `swmm_step` calls in `triton.h` and `swmm_triton.h` for captured return values after blanking comments/strings. Synthetic capturing/discarding controls and a population floor guard against an empty scan. This does not prove arbitrary semantic error handling; it detects discarded call results in its declared source population.

Inventory registration separates fast Criterion-P and guard-closure checks from slow regeneration and closure-mode mutation checks. `SLOW-SNAPSHOT-INVENTORY-REGENERATION` and `SLOW-SNAPSHOT-INVENTORY-CLOSURE-MODES` carry the `slow` label. Regeneration distinguishes source drift from unresolved triage; see [snapshot inventory](snapshot-inventory.md).

Focused commands from a configured build directory include `./triton_ctest.sh -LE slow`, `./triton_ctest.sh -L slow`, `./triton_ctest.sh -R '^UNIT-QUALITY-RESUME$'`, and `./triton_ctest.sh -R '^SOURCE-SWMM-'`. These are documented entrypoints, not commands executed here.

Evidence: [guard assertions](../../test/test_swmm_guard_predicate_closure.py#L193-L390), [return-code predicate and controls](../../test/test_swmm_return_codes_source.py#L64-L119), [inventory registration](../../test/inventory/CMakeLists.txt).

## Standalone coupled-resume suite

`test/coupled_resume_suite.sh {executable} {suite-directory}` is a separate shell entrypoint, not registered in the inspected CTest tree. It copies inputs from repository `benchmark/`, creates a build/executable symlink, clones clean outputs for resume legs, and uses matching rank counts. Required assets include `TRITONSWMM.cfg`, DEM/Manning rasters, `strmflow`, `extbc`, `cid`, the SWMM model tree, and the external `dats` files copied into that model tree. The attempted directory read of `benchmark/` during page research returned `Path 'benchmark' not found`; **this checkout was not established as ready to run this suite**. Supply its real benchmark assets before attempting it.

The script uses `mpirun --allow-run-as-root --oversubscribe` for MPI legs, plus `timeout`, Python, file-copy/diff tools, and `ldd`/`gomp` detection for its optional OpenMP leg. Those hard-coded launcher options require an appropriate MPI environment; a configured CMake `RUN_COMMAND` is not used by this shell suite.

Implemented legs include first/middle/last checkpoint resumes, forced replay, snapshots, chained resumes, retention collision, eight-rank SEQ/PAR outputs, and a sixteen-rank completion/report regression. The latter does not itself perform a resume. The degraded-input leg now checks refusal at `swmm_open` with ERROR 361 and a report, despite older introductory comments describing baseline comparison. Failure legs match missing/V1 replay-side-file and corrupt-snapshot diagnostics. Performance checks inspect SWMM child columns, nonnegative residual time, and approximate child/parent accounting.

The comparison helper requires `Simulation ends` before recursive binary-raster comparison and byte comparisons of the exchange side-file and SWMM binary. This blocks the simplest copied-tree startup-failure false pass, but unlike the quality test it does not delete copied report/node outputs or enforce binary rewrite stamps. Do not attribute those stronger controls to this suite.

Evidence: [suite setup and comparison](../../test/coupled_resume_suite.sh#L37-L102), [runtime matrix](../../test/coupled_resume_suite.sh#L107-L219), [failure and performance checks](../../test/coupled_resume_suite.sh#L221-L285).

## Verification status and neighbors

No solver build, CTest run, or standalone suite was executed while authoring this page. Excluded reference data and `*.out` artifacts were not opened. Page submission validates documentation structure and Claims, not the physics or runtime suites.

- [Quickstart](../quickstart.md)
- [Build and backends](../operations/build-and-backends.md)
- [SWMM coupling](../integrations/swmm-coupling.md)
- [Checkpoint and resume](../workflows/checkpoint-and-resume.md)
- [SWMM state snapshots](../architecture/swmm-state-snapshots.md)
- [Snapshot inventory](snapshot-inventory.md)

---
type: architecture
title: Simulation ownership and lifecycle
description: End-to-end control flow from MPI and Kokkos startup through model initialization, timestep execution, checkpoint publication, and SWMM shutdown.
tags: [architecture, lifecycle, simulation, state]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-d44494ef3e497fea81240ef8
    resource: repo://CMakeLists.txt
  - id: openwiki-source-d364d949938a433276255c32
    resource: repo://src/main.cpp
  - id: openwiki-source-fbe082bc4fe57162918ea012
    resource: repo://src/output.h
  - id: openwiki-source-c26b636827b23cd5791d2bcb
    resource: repo://src/swmm_triton.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Simulation ownership and lifecycle

The executable is a thin process driver around `Triton::triton<value_t>`. The model owns configuration, raster state, partition metadata, forcing, timers, execution-space buffers, and output orchestration. Numerical kernels and I/O helpers implement work requested by that owner rather than driving independent simulation loops.

## Process and library lifetimes

CMake builds `triton.exe` from `src/main.cpp` and links Kokkos; optional configurations add yaml-cpp, GDAL, or bundled SWMM. The driver includes the model and supporting header implementations into this executable.

`main` initializes MPI, optionally splits ensemble communicators, and initializes Kokkos. It obtains rank/size from `ENSIFY_COMM_WORLD`, constructs the model with ordinary or ensemble-provided arguments, calls `initialize`, writes the run-log header, calls `simulate`, and records total simulation time. The model leaves scope before Kokkos finalization. Ensemble cleanup follows Kokkos cleanup, then MPI finalizes.

Sources: [build target and optional integrations](../../CMakeLists.txt#L61-L98), [driver](../../src/main.cpp#L40-L104).

## State ownership

The model stores typed configuration arguments and original configuration text alongside DEM/state matrices, hydrograph/runoff readers, local/global observation metadata, and partition descriptors. The public surface is construction, initialization, simulation, and the getters used by the driver; orchestration helpers and state are private.

Host pointer tables alias matrix-owned H/QX/QY/Manning/DEM/MAXH arrays as well as separately allocated auxiliary arrays. Device pointer tables hold separately allocated execution-space storage. The destructor deletes host auxiliaries, not matrix-owned data, and frees entries from the device tables. This distinction matters when changing the buffer layout: table membership alone does not imply that host storage is independently owned.

Sources: [model interface and ownership](../../src/triton.h#L37-L164), [host table](../../src/triton.h#L1955-L1995), [destruction](../../src/triton.h#L2155-L2196).

## Initialization dependency chain

1. Read configuration and set `simtime` from `sim_start_time`. An explicit nonnumeric configuration argument replaces the default path; a positive checkpoint selector chooses the generated checkpoint configuration instead.
2. Load forcing series. Hydrograph times become seconds; runoff additionally converts hourly rates and millimetres to SI rate units.
3. Read DEM geometry, then load sequential rasters when selected. Establish the static partition or restore saved dynamic partition dimensions.
4. Map sources, observations, and external boundaries to the partition. Distribute sequential input or read parallel input, then prepare runoff arrays.
5. In a coupled build, initialize SWMM only when `inp_filename` is nonempty. A missing filename is an explicitly supported uncoupled run, with a warning on rank zero. For a coupled resume, restore SWMM state or replay history according to the restart policy, then scatter restored node depths before live exchange.
6. Allocate host auxiliaries, construct pointer tables, and create/copy device arrays. Optional DEM-derived node rims are checked collectively before the per-rank SWMM buffers are appended and copied.

The geometry and partition therefore precede both spatial input mapping and coupling ownership; the host DEM precedes DEM-derived rim calculation. Resume state must reach the coupling buffers before the first post-resume exchange.

Sources: [initialization order](../../src/triton.h#L383-L426), [configuration and forcing](../../src/triton.h#L631-L738), [coupled initialization and restore](../../src/triton.h#L428-L528), [rim checks and coupling buffers](../../src/triton.h#L530-L625).

## Simulation setup and timestep loop

`simulate` starts its timer, derives a small dry-domain fallback timestep from forcing/output intervals, synchronizes ranks, and initializes output metadata. If observations are enabled it writes an initial sample. On checkpoint resume it then loads the persisted ghost ring and recopies H/QX/QY to device before the first flux calculation; this preserves boundary state that cannot be reconstructed from checkpoint interiors alone.

The loop continues while `simtime < sim_duration`. Each iteration increments counters; adaptive mode computes local and global timesteps, while fixed mode keeps the configured timestep. `compute_new_state` advances the state, then the loop advances simulation time and handles progress reporting, observation intervals, and raster checkpoints.

Within `compute_new_state`, ordering is:

1. X flux, Y flux, then cell update.
2. Runoff and point-source forcing when present.
3. Open-boundary mirroring and explicit external boundary conditions.
4. Coupled surface/sewer exchange and the rank-zero SWMM step when globally enabled.
5. Wet/dry enforcement, followed by multi-rank halo exchange and cross-halo enforcement.

This is an ordering contract, not a set of interchangeable callbacks. In particular, post-exchange depths must pass through wet/dry enforcement. Detailed numerical laws and communication layouts belong on the linked pages below.

Sources: [setup and loop](../../src/triton.h#L2200-L2288), [fallback timestep](../../src/triton.h#L2411-L2434), [flux/update](../../src/triton.h#L2478-L2493), [forcing/boundaries](../../src/triton.h#L2495-L2556), [coupling and halo](../../src/triton.h#L2570-L2687).

**Ensemble boundary caveat:** most model communication uses `ENSIFY_COMM_WORLD`, but `compute_init_dt` explicitly reduces over `MPI_COMM_WORLD` when the member size exceeds one. Do not infer complete ensemble-communicator isolation from the macro's existence.

Source: [fallback reduction](../../src/triton.h#L2429-L2433).

## Checkpoint publication and optional rebalance

At a raster output boundary, the model copies H/QX/QY and requested maximum depth to host and synchronizes. In a coupled run, rank zero flushes exchange history **before** output writes the checkpoint configuration. `out.write_output` dispatches raster output, writes the ghost ring, and on rank zero writes checkpoint configuration plus `cid`. The model then requests the SWMM snapshot. This is an ordered sequence of artifacts, not a claim of one atomic transaction over the entire checkpoint.

Performance reporting follows synchronization. Dynamic multi-rank runs save the active partition and may then rebalance at the configured checkpoint cadence; the final simulation time does not trigger a new partition. Restart protocols and their refusal/fallback rules are described separately.

Sources: [checkpoint sequence](../../src/triton.h#L2300-L2361), [output publication tail](../../src/output.h#L703-L719).

## Shutdown and failure boundaries

After the loop, enabled observations receive a final sample. The model fences device work, synchronizes multi-rank runs, stops timers, writes final timing output, and prints `Simulation ends`. Only after that message does coupled finalization run. The global coupling count controls whether `end_swmm` is called; inside it rank zero closes exchange history, ends SWMM, writes its report, and closes the engine. A `Simulation ends` line alone therefore precedes sewer-report finalization.

The driver subsequently records simulation timing and destroys the model before shutting down Kokkos/MPI. There is no enclosing recovery loop in `main`; initialization failures do not become a partially successful simulation. A rejected coupled snapshot restore explicitly uses `MPI_Abort` from the rank-zero path so peers do not hang at the next collective.

Sources: [normal shutdown order](../../src/triton.h#L2366-L2405), [SWMM finalizer](../../src/swmm_triton.h#L1370-L1377), [restore abort](../../src/triton.h#L482-L489), [driver lifetime](../../src/main.cpp#L87-L104).

## Continue by responsibility

- [MPI and device execution](parallel-execution.md): strip ownership, transfers, collective guards, and repartitioning limitations.
- [Shallow-water numerics](../concepts/shallow-water-numerics.md): flux and update equations.
- [SWMM coupling](../integrations/swmm-coupling.md): node mapping and exchange behavior.
- [Checkpoint and resume](../workflows/checkpoint-and-resume.md): artifact protocol and state restoration.
- [Configuration and inputs](../workflows/configuration-and-inputs.md): file resolution and forcing contracts.

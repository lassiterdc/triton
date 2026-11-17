---
type: architecture
title: MPI decomposition and device execution
description: How TRITON distributes raster rows, stages device state, exchanges halos, and rebuilds partitions, including coupled-run collective constraints.
tags: [mpi, parallelism, gpu, decomposition]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-99a4d4e1b6f34174c1134d7f
    resource: repo://src/constants.h
  - id: openwiki-source-8ca55e77f7850d2615133c5b
    resource: repo://src/Ensify.h
  - id: openwiki-source-f2da30b3e71ec16ecb6b6fd2
    resource: repo://src/kokkos_utils.h
  - id: openwiki-source-dc493dccd92de1bd10eaf41e
    resource: repo://src/mpi_utils.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-c0c9e3840da0f06d67f05b32
    resource: repo://test/test_swmm_guard_predicate_closure.py
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# MPI decomposition and device execution

`Triton::triton<T>` owns partition metadata, host/device buffers, and the order of communication and kernels. MPI distributes horizontal row strips; device execution parallelizes work inside each strip. These are separate layers, not alternative execution modes.

## Partition geometry and initial distribution

`MpiUtils::create_local_dims` assigns each rank `globalrows / size` real rows, gives the remainder to the earliest ranks, and retains the full raster width. Each strip adds `GHOST_CELL_PADDING` on both sides of both dimensions; the constant is one cell. `partition_data_t::part_dims` stores padded dimensions.

For sequential input, `partition_matrix_files` scatters DEM, Manning data, and applicable state/map rasters from rank zero. `scatter_exchange` copies rank zero's strip locally and sends overlapping padded row ranges to the other ranks. Its source offset advances by the preceding strip's real-row count, so neighboring strips share their boundary context. The integer variant handles runoff IDs. With one rank, the model uses the loaded matrices directly.

Sources: [partition geometry and scatter](../../src/mpi_utils.h#L40-L88), [overlapping scatter](../../src/mpi_utils.h#L188-L275), [initial distribution](../../src/triton.h#L1606-L1655), [padding constant](../../src/constants.h#L64-L66).

## Memory and device dispatch

`create_host_vectors` constructs indexed pointer tables over state matrices and auxiliary allocations. `create_device_vectors` builds separate execution-space allocations, copies initial state and forcing metadata, and synchronizes before returning. The halo allocation holds `12 * cols * GHOST_CELL_PADDING` values for bundled depth and both discharge components.

The active `gpu*` implementation in `kokkos_utils.h` uses Kokkos allocation, unmanaged byte views with `deep_copy`, and `Kokkos::fence`; the name `gpuMemcpyAsync` does not by itself promise asynchronous copies. `parallel_for` normally calls Kokkos. Defining `TRITON_HIP_LAUNCHER` or `TRITON_CUDA_LAUNCHER` instead selects the explicit kernel launcher with the corresponding compile-time launch width. `gputils.h` separately provides HIP/CUDA macro aliases; it should not be confused with the Kokkos wrapper actually included by `triton.h`.

Sources: [buffer ownership](../../src/triton.h#L1906-L2020), [initial transfers](../../src/triton.h#L2103-L2137), [dispatch and memory wrapper](../../src/kokkos_utils.h#L18-L83), [runtime aliases](../../src/gputils.h#L16-L49), [includes](../../src/triton.h#L22-L27).

## Step synchronization and halo exchange

Each rank computes a local stable timestep. Multi-rank runs reduce it using `MPI_Allreduce(MPI_MIN)`; the result is also bounded by the next raster-output time. After the local hydrodynamic update and any sewer exchange, `wet_dry` runs before the halo stage.

For multiple ranks, `halo_copy_from_gpu` packs boundary values. With `gpu_direct_flag`, the code fences and passes the device halo pointer directly to MPI. Otherwise it copies the bundle to host, exchanges there, and copies it back. Both paths unpack into state arrays and run `wet_dry_qy_halo` to enforce the cross-strip Y-momentum constraint. GPU-direct operation therefore requires an MPI installation capable of handling that execution-space pointer; the flag is a branch selection, not a capability probe.

`MpiUtils::exchange` uses nonblocking sends/receives followed by waits, first towards the next rank and then towards the previous rank. Edge ranks omit nonexistent neighbors. `USE_HALO` interprets the compact bundle differently from full-matrix exchange.

Sources: [global timestep](../../src/triton.h#L2440-L2474), [halo schedule](../../src/triton.h#L2649-L2687), [neighbor communication](../../src/mpi_utils.h#L125-L185).

## Coupled participation: global versus local guards

The per-step SWMM region is guarded by `global_num_of_swmm_links > 0`. Every rank in `ENSIFY_COMM_WORLD` must participate in its `MPI_Gatherv` and `MPI_Scatterv`, including a rank whose strip contains no coupled node. Only the device transfers and exchange kernel use the local `num_of_swmm_links > 0` guard, because zero-node ranks never append the SWMM pointer-table entries.

Collectives use `swmm_model.exchange_q.data()` and `new_depth.data()` directly rather than indexing those optional pointer-table entries. Rank zero alone reorders the gathered flows, logs the step, advances SWMM, and prepares redistributed depths. The coupling block precedes wet/dry and halo enforcement. Rank-zero checkpoint flush/snapshot operations likewise use the global count, not rank zero's strip population.

`ENSIFY_COMM_WORLD` is the ensemble member communicator in ensemble builds and `MPI_COMM_WORLD` otherwise. Communication must not accidentally escape the selected member.

Sources: [coupling schedule and guards](../../src/triton.h#L2557-L2649), [checkpoint guards](../../src/triton.h#L2315-L2334), [communicator selection](../../src/Ensify.h#L49-L78), [communicator macro](../../src/Ensify.h#L116-L120).

## Dynamic repartitioning and resume

At configured checkpoint intervals, multi-rank `dynamic` runs may call `new_domain_decomposition`, except at the final simulation time. The controller gathers accumulated `BALANCING_MPI_TIME`, compares each rank with the mean using a five-percent imbalance threshold, and adjusts row counts gradually. It enforces at least two real rows per strip and rejects a proposed total exceeding the raster height.

When a change is selected, the model frees auxiliary/device buffers, remaps sources, observations and boundaries, redistributes rasters, and rebuilds host/device vectors and output metadata. Sequential input reuses root-owned static rasters; parallel input gathers those rasters first. H, QX, QY and MAXH are gathered using the old output layout and scattered using the new partition.

**Coupled limitation:** this rebuild does not rerun SWMM node ownership processing or reappend/reallocate the conditional coupling buffers. Direct coupling-vector access keeps collectives independent of optional host-table slots, but does not make dynamic coupled repartitioning a validated combination. Treat it separately from static coupled MPI operation.

Before any post-checkpoint repartition, the current partition is written. On a dynamic multi-rank resume, rank zero reads the saved real-row counts, broadcasts them, and reconstructs padded dimensions before input distribution. Parallel input also retains a regular auxiliary partition for loading before conversion.

Sources: [checkpoint trigger](../../src/triton.h#L2351-L2361), [controller and rebuild](../../src/triton.h#L2719-L2825), [buffer reset and redistribution](../../src/triton.h#L2830-L2999), [restore](../../src/triton.h#L400-L423).

## Focused verification and boundaries

`test/test_swmm_guard_predicate_closure.py` checks the distinction between global model operations and local buffer access: rank-zero combined predicates agree, MPI/coupling calls are excluded from local-count blocks, and indexed SWMM buffer uses require a local guard. It is a source-style/brace-walk check, not a runtime MPI or GPU correctness proof; its documented limitations include block comments, string braces and preprocessing.

Source: [guard-check contract](../../test/test_swmm_guard_predicate_closure.py#L3-L103).

Continue with [simulation lifecycle](simulation-lifecycle.md), [shallow-water numerics](../concepts/shallow-water-numerics.md), [SWMM coupling](../integrations/swmm-coupling.md), [build/backends](../operations/build-and-backends.md), [checkpoint/resume](../workflows/checkpoint-and-resume.md), or [ensemble runs](../workflows/ensemble-runs.md).

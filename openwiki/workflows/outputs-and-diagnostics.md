---
type: workflow
title: Output artifacts and runtime diagnostics
description: Trace TRITON raster, observation, checkpoint-metadata and log writers, including parallel layouts and the meaning of runtime performance columns.
tags: [outputs, diagnostics, rasters, timing, checkpoints]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-99a4d4e1b6f34174c1134d7f
    resource: repo://src/constants.h
  - id: openwiki-source-d364d949938a433276255c32
    resource: repo://src/main.cpp
  - id: openwiki-source-fbe082bc4fe57162918ea012
    resource: repo://src/output.h
  - id: openwiki-source-a90f44df34a566dc244e7ddf
    resource: repo://src/supertimer.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-47ce21353e1f2c4629fd35a5
    resource: repo://test/coupled_resume_suite.sh
  - id: openwiki-source-c8264fcd41cacbe9ffe1c639
    resource: repo://test/paraboloid_ens1.cfg
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Output artifacts and runtime diagnostics

`Triton::triton::simulate` schedules output and host transfers; `Output::output<T>` owns artifact layout and writing. Raster/checkpoint cadence, observation cadence, and console cadence are separate. A successful visualization write is not by itself proof that a complete restart set exists; use [Checkpoint and resume](checkpoint-and-resume.md) for the durability and restore contract.

## Select fields, format and layout

`print_option` is inspected for individual characters: `h` selects `H` (water depth), `u` selects `QX`, and `v` selects `QY`. These discharge fields are not velocity rasters despite the `huv` spelling. `max_value_print_option` containing `h` additionally selects `MH`, the maximum-depth array.

| `output_format` | Files for each selected field |
|---|---|
| `ASC` | Binary restart raster in `bin/` followed by plain numeric ASCII in `asc/` |
| `BIN` | Binary raster in `bin/` |
| `GTIFF` | GeoTIFF in `gtiff/` plus a companion binary restart raster in `bin/`; compiled only with `TRITON_GDAL` |

`output_option=SEQ` uses gathered output; the dispatcher's other branch uses per-rank output. Configure the documented `SEQ` or `PAR` values rather than relying on that fallback. For sequential ASCII, the immediately preceding binary writer fills the gather buffer that ASCII subsequently consumes.

- **SEQ:** binary/GeoTIFF writers gather strips to member rank zero; the resulting raster omits exterior ghost cells. Sequential ASCII uses the same assembled interior and writes fixed-point values with six decimal places, without a GIS header.
- **PAR:** each rank writes a local interior binary/ASCII raster. Binary files begin with row and column counts stored as `T`, followed by row-major interior values of type `T`; these are not integer headers or a portable interchange encoding.
- **GeoTIFF:** bands are `Float32` even when the model uses double precision. Projection is passed through GDAL's `SetFromUserInput`; the sequential geotransform uses the DEM lower-left origin, cell size and total interior height to obtain the upper-left corner.

GeoTIFF output is therefore not a replacement for full-precision checkpoint binaries. The source's GDAL error paths report missing drivers, dataset creation failures and row-write failures, but do not provide an all-rank transactional output protocol.

## Root, naming and mosaics

Raster, observation, cfg and performance writers generally form their root by concatenating `project_dir + "/" + output_folder`. Use relative output folders consistently with the [Configuration and inputs](configuration-and-inputs.md) contract; the run-log helper's absolute-path support does not imply that all writers use that helper.

`get_mat_path` appends an extension to `outfile_pattern` and calls `snprintf` with, in order, root directory, format subdirectory, variable name, print/checkpoint ID, and rank. The usual `%s/%s/%s_%02d_%02d` thus produces names such as `bin/H_12_00.out`. The ID is a checkpoint index, not elapsed seconds. Even a gathered raster carries rank zero's suffix. Formatting uses a fixed 256-byte buffer; avoid long or incompatible custom patterns.

For `PAR + GTIFF`, an `MPI_Allgather` obtains local interior row counts. Each rank derives its tile's vertical position from the rows at and below its rank, writes its TIFF, and enters a barrier. Rank zero then writes `gtiff/{field}_{id:02d}.vrt`, stacking relative tile references in rank order. The VRT contains metadata and references, not copied raster data. Keep tiles beside it.

**Naming constraint:** VRT tile references are hard-coded as `{field}_{id:02d}_{rank:02d}.tif`; they do not use `outfile_pattern`. A custom tile naming pattern can therefore leave broken VRT references.

**Indexing caution from source inspection:** the sequential GeoTIFF loop reads gathered cells at `(i + ghost_padding, j + ghost_padding)`, but the parallel loop reads `arr.get_address_at(i,j)` starting at zero although its dimensions exclude the ghost ring. Do not infer GIS tile/interior equivalence solely from the intended layout; this page records writer behavior, not a runtime geospatial validation.

## Scheduling and observation series

The raster branch runs when `simtime >= print_interval * (print_id + 1)`, increments the ID and transfers H/QX/QY (and selected MAXH) to host before writing. In variable-step mode, `compute_global_dt` clips a step at the next raster boundary. There is no unconditional final raster write after the loop, so a finish time between print boundaries does not automatically gain a final checkpoint.

When `time_series_flag` is enabled:

1. `simulate` initializes observation metadata and writes an initial sample at its starting simulation time.
2. After each step, it checks `simtime - last_print_obs_time >= print_observation`; on a match it advances the observation clock by one interval and records the current state/time. This is an `if`, not an interpolation or catch-up loop.
3. It unconditionally writes one final observation sample after the simulation loop. This can repeat the final timestamp when a scheduled observation already wrote it.

Observation extraction transfers only the selected observation arrays. The writer gathers values with `ENSIFY_COMM_WORLD`, reorders them using the observation-index mapping, and rank zero appends comma-separated rows to `series/H_at_Xsec.txt`, `QX_at_Xsec.txt` and/or `QY_at_Xsec.txt`. Clean starts write headers (`Time(s)` and numbered point columns); nonzero checkpoint starts do not recreate headers. Appending on resume is not history trimming, so preserve/manage existing series deliberately.

## Checkpoint metadata and boundary state

Every `write_output` call writes a per-rank `bin/GR_{id}_{rank}.out` using the configured pattern, independently of `SEQ/PAR`. It stores local dimensions as `T`, then H/QX/QY triples for every cell on the ghost ring. These values supplement interior rasters: the resume path restores the ring before the first flux evaluation. A write-open failure warns that resume will fail; read failures, incompatible dimensions and detected truncation stop the run.

Rank zero then writes `cfg/config_{id}.cfg` and a `cid` file in the process working directory. The cfg writer starts with the original cfg text and replaces matching `sim_start_time=`, `checkpoint_id=`, `time_step=` and `it_count=` lines. Time and step values use `max_digits10` precision; it does not synthesize absent keys. The passed `time_step` value is the loop's `average_dt` accumulator, not a separately recomputed universal restart timestep.

For dynamic decomposition with multiple ranks, the checkpoint branch writes `domain_decomposition/domain_decomposition{id}.txt` with `%Rank,Rows,Cols` and interior strip dimensions before any next rebalance. In coupled runs, rank zero flushes exchange history before raster/cfg publication, then writes the SWMM snapshot after `write_output`, inside the I/O timer. These files have distinct failure/publication rules; `cid` alone is not a transaction marker for all of them. See [Checkpoint and resume](checkpoint-and-resume.md) and [SWMM coupling](../integrations/swmm-coupling.md).

## Three diagnostic surfaces

| Surface | Content and lifecycle |
|---|---|
| `std::cerr` | Initialization/progress, checkpoint and observation notices, warnings/errors, `Simulation starts`/`Simulation ends`; periodic progress uses `it_count % it_print` on rank zero |
| `{output_root}/log.out` | Rank-zero run header created after model initialization, then a timing line appended after `simulate` returns |
| Performance text files | Per-rank timer columns and an arithmetic `Average` row, written by rank zero |

The run header records hostname, Linux CPU model when available, Kokkos-selected CUDA/HIP device name (otherwise `none`), task/thread counts, compile-time backend labels and Git version. Its total-GPU field assumes one GPU per task for GPU builds; it is not an independent inventory of unique physical devices. The log path helper normalizes trailing slashes, accepts an absolute `output_folder`, and recursively creates parent directories; the main raster writers do not share that path construction.

`log.out` is not a capture of console diagnostics and is overwritten when the header is created. Its line labeled `TRITON total wall time [s]` receives `model.get_simulation_time()`, i.e. the `SIMULATION_TIME` accumulator, not the `TOTAL_TIME` accumulator. For ensemble stream redirection and its concurrency limits, see [Ensemble runs](ensemble-runs.md).

## Interpret performance accounting

`SuperTimer` uses `gettimeofday`, accumulates elapsed microseconds on `stop`, and converts its default units to seconds. `start` establishes another interval without clearing accumulated time; `restart` clears that category. The values below describe instrumentation boundaries, not benchmark results.

At raster checkpoints, `WRITE_PERFORMANCE` enables a GPU synchronization and member barrier, stops simulation/total timers, writes `performance/performance{id}.txt`, and restarts those timers. End of simulation synchronizes again and writes root `performance.txt` with `print_id=-1`. Checkpoint timing reports are cumulative, not per-checkpoint deltas; their own writes are outside the stopped simulation/total intervals. The final timing report also precedes SWMM end/report cleanup.

| Column | Accounting meaning |
|---|---|
| `Compute` | Bracketed surface kernels, dt work, and relevant host/device transfers; surface work is synchronized before entering the coupling timer |
| `MPI` | TRITON's bracketed timestep reduction/halo communication, separate from coupling collectives |
| `IO` | Bracketed observation/checkpoint writes, including exchange-history flush and SWMM snapshot at checkpoints |
| `Resize` | Dynamic repartition/rebalance bracket |
| `SWMM` | Parent bracket for coupled exchange, collectives and rank-zero solve/remapping/log append |
| `SWMM_XFER` | Local coupled host/device transfers and exchange kernel; only ranks with local nodes execute it |
| `SWMM_MPI` | Gather plus scatter brackets, excluding the rank-zero solve between them |
| `SWMM_STEP` | Rank-zero remaps, exchange-history append and `swmm_step`; an upper bound on solver-only time |
| `SWMM_OTHER` | Residual `SWMM - SWMM_XFER - SWMM_MPI - SWMM_STEP` |
| `Other` | Residual `Simulation - Compute - MPI - IO - Resize - SWMM` |
| `Simulation` | Simulation accumulator, including work not charged to a child category |
| `Init` | Residual `Total - Simulation` |
| `Total` | Accumulator started in the model constructor |

Do not add SWMM children to their parent again. Each `Average` cell is the plain mean across ranks, not the critical-path time. In particular `SWMM_STEP` is measured only on rank zero, so its average is diluted by rank count; inspect rank zero's row to assess that serial bracket. Residuals are calculated by subtraction, not independently measured or clamped.

The coupled resume suite checks the SWMM child-column header and nonnegative `SWMM_OTHER`, and checks accounting closure within the printed precision. That is an accounting assertion, not a performance comparison. See [Regression and resume](../testing/regression-and-resume.md).

## Source anchors

- `src/output.h:318-719`: initialization, field/format dispatch, checkpoint cfg and cid.
- `src/output.h:723-1098`: ASCII/binary and ghost-ring layouts.
- `src/output.h:1101-1438`: GeoTIFF geometry, VRT construction and filename formatter.
- `src/output.h:384-520`, `src/output.h:1440-1565`: observation mapping/appends and cfg rewriting.
- `src/output.h:1568-1727`, `src/output.h:1759-1994`: performance reports, partition metadata and run logs.
- `src/triton.h:2200-2403`, `src/triton.h:2451-2474`, `src/triton.h:2570-2645`: scheduling, cleanup ordering and coupling timers.
- `src/supertimer.h:176-292`, `src/constants.h:142-161`, `test/coupled_resume_suite.sh:259-279`: timer mechanics/categories and accounting assertions.

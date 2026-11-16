---
type: concept
title: Shallow-water timestep and forcing
description: Implementation-level state variables, directional flux accumulation, friction and wet/dry rules, timestep selection, and forcing/boundary sequencing.
tags: [numerics, shallow-water, timestep, forcing, boundaries]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-99a4d4e1b6f34174c1134d7f
    resource: repo://src/constants.h
  - id: openwiki-source-bdbe4d036fb8d6bb3ba39aff
    resource: repo://src/inflow.h
  - id: openwiki-source-c0337859aaa213a6523dbbc6
    resource: repo://src/kernels.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-4b8e7dce368774e92d99ea30
    resource: repo://test/CMakeLists.txt
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Shallow-water timestep and forcing

The live surface state is depth H and unit discharges QX/QY, not velocity rasters. The flux kernels recover velocities as `qx / h` and `qy / h` above the configured depth threshold. DEM supplies bed elevation, and NMAN supplies squared Manning coefficients after input preparation.

Sources: [state indices](../../src/constants.h#L82-L87), [velocity recovery](../../src/kernels.h#L106-L133), [Manning preparation](../../src/triton.h#L767-L783), [parallel Manning preparation](../../src/triton.h#L1011-L1015).

## Precision and thresholds

`value_t` is double unless `USE_SINGLE_PRECISION` selects float; the MPI datatype follows that choice. `EPS12` is `1e-12` for double and `1e-6` for float, despite its name. `MAX_VALUE` is the finite sentinel `1e10`, not the numeric type's maximum. `hextra` is a separate configured depth threshold used for resting momentum, velocity evaluation, and wet/dry decisions. Gravity is `_G_ = 9.81`, while `SQRTG` is a separately recorded constant used in characteristic speeds.

Sources: [precision macros](../../src/constants.h#L37-L51), [gravity constants](../../src/constants.h#L132-L133), [depth rules](../../src/kernels.h#L629-L657).

## Directional fluxes: assignment before accumulation

`compute_new_state` calls `flux_x`, then `flux_y`, then `update_cells`. X interfaces pair a cell with its eastern neighbor; Y interfaces pair a cell with the row above, using a northward normal. The kernels have different ghost-edge exclusions so the required interfaces are evaluated under row-strip decomposition.

Both directions form square-root-depth-weighted velocities, a wave speed from mean depth, and three characteristic eigenvalues. They split fluctuations by characteristic sign, apply entropy correction when left/right wave speeds straddle zero, and include a bed-step source correction. Partially submerged bed steps limit the effective bed difference; subcritical wet/wet interfaces additionally clamp the source contribution. Dry-cell cases can redirect mass fluctuations, and sufficiently small fluctuations are zeroed using `EPS12`.

The X kernel **assigns** the directional RHS slots, including zeros at wholly dry interfaces. The Y kernel **adds** its contributions with `+=`. Reversing the calls or making Y assign would discard X contributions. The arrays are therefore timestep workspaces with an ordering dependency, not independently interchangeable flux outputs.

Sources: [driver order](../../src/triton.h#L2478-L2493), [X interface and wave construction](../../src/kernels.h#L72-L219), [X redistribution and RHS writes](../../src/kernels.h#L221-L298), [Y orientation](../../src/kernels.h#L352-L378), [Y wave/source correction](../../src/kernels.h#L397-L510), [Y accumulation](../../src/kernels.h#L553-L572).

## Cell update and Manning friction

`update_cells` excludes ghost cells. It computes new depth as old depth minus the two accumulated depth RHS slots and clips negative results to zero. When the new depth is below `hextra`, both discharge components become zero.

Otherwise it first subtracts momentum RHS contributions to obtain `mx` and `my`. Friction is applied only when squared Manning exceeds `EPS12`, **old** depth `hn` is at least `hextra`, and the updated momentum magnitude exceeds `EPS12`. The exact coefficient is `tt = dt * g * n_squared * sqrt(mx*mx + my*my) / (hn*hn*cbrt(hn))`; each component uses the closed expression `-0.5 * (m - m*sqrt(1 + 4*tt)) / tt`. When that guard fails, momentum stays at the flux-updated value. Do not replace the old-depth denominator or the `sqrt(1 + 4*tt)` form when describing this implementation.

Source: [cell update](../../src/kernels.h#L597-L670).

## Adaptive timestep and fixed-mode boundary

`compute_dt_and_sqrt` refreshes `sqrt(h)` for each slot and sets its timestep to the sentinel unless `h > hextra`. Wet cells use `courant * dx / (max(abs(qx), abs(qy))/h + SQRTG*sqrt(h))`. `find_min_dt` reduces locally, using Kokkos reduction by default or `min_element` under `TRITON_REDUCE_1`. Multi-rank adaptive execution then applies an MPI minimum reduction and shortens the step to the next raster-output boundary.

If every candidate remains the sentinel, the model uses `init_dt`: one percent of the minimum applicable raster, observation, runoff, and source-series interval. The fallback setup contains a world-communicator reduction; see [parallel execution](../architecture/parallel-execution.md) and [lifecycle](../architecture/simulation-lifecycle.md).

Fixed mode initializes `global_dt` from `time_step` and bypasses both `compute_local_dt` and `compute_global_dt`. Consequently it also bypasses their output-boundary clamp and the square-root refresh bundled into local timestep computation. Host square-root storage starts at zero. This documents the actual branch, not an assertion that fixed mode has numerical parity with adaptive mode; fixed-step use needs separate verification.

Sources: [local timestep kernel/reduction](../../src/kernels.h#L1006-L1052), [fallback interval](../../src/triton.h#L2411-L2434), [global timestep](../../src/triton.h#L2440-L2474), [fixed/adaptive branch](../../src/triton.h#L2249-L2274), [initial square-root buffer](../../src/triton.h#L1917-L1921).

## Forcing semantics

Forcing is applied after the hydrodynamic cell update and before external boundaries and coupling.

| Input | Sampling and state effect |
|---|---|
| Point hydrographs | The driver chooses bracketing rows at current `simtime`; `compute_flow` linearly interpolates or uses one endpoint, then adds `flow * dt / (dx*dx)` to the source cell's depth. It does not inject momentum. |
| Runoff regions | The driver advances a runoff row index and holds that row's region intensity; the kernel adds `intensity * dt` to real-cell depth. This is not the hydrograph interpolation path. Negative/shallow resulting depths are clipped as applicable and momentum is zeroed below `hextra`. |

Hydrograph times are converted from hours to seconds. Runoff times undergo that conversion too, while rates additionally convert from per-hour to per-second and millimetres to metres. Spatial runoff IDs choose columns of the prepacked forcing array. Keep the sampling distinction when constructing inputs: the presence of time columns does not mean every forcing kind interpolates identically.

Sources: [forcing order/index selection](../../src/triton.h#L2495-L2533), [point update](../../src/kernels.h#L890-L924), [runoff update](../../src/kernels.h#L942-L992), [unit preparation](../../src/triton.h#L686-L713).

## External boundaries

Open-boundary mirroring copies interior H/QX/QY into exterior ghost cells; north/south propagation is restricted to the outermost ranks. Explicit boundary conditions run afterward, so their writes take precedence at selected cells. `extBC` requires axis-aligned endpoint selections on a raster boundary and enumerates their cells.

`compute_extbc_values` recognizes these types:

- **0:** propagate the selected interior state into ghost cells, including corner propagation.
- **1:** interpolate a water-surface elevation series at `simtime + dt/2`, then set depth to `max(stage - DEM, 0)`. This is a stage boundary, not a volumetric point hydrograph.
- **2:** derive normal-slope velocity from the supplied slope, local depth and squared Manning with a denominator floor.
- **3:** derive velocity from the supplied Froude number and `sqrt(g*h)`.

For nonzero types, orientation factors write both inner and ghost states and select/sign the normal discharge component; corner handling is explicit. Type 1 uses endpoint branches and interval search with strict comparisons: do not infer a general interpolation-library contract for all equality/out-of-range cases from its name.

Sources: [boundary placement checks](../../src/extbc.h#L204-L292), [kernel boundary selection](../../src/kernels.h#L1094-L1160), [type behavior and interpolation](../../src/kernels.h#L1169-L1258), [open mirroring](../../src/kernels.h#L1276-L1348), [driver precedence](../../src/triton.h#L2537-L2547).

## Wet/dry enforcement and partition seams

SWMM exchange runs **before** `wet_dry`, so the latter sees final post-exchange depths. For a wet cell, a dry adjacent higher bed can zero the corresponding discharge component: the X condition tests either neighbor, not a requirement that both be dry. The Y checks beside partition edges initially consider the local side; after halo exchange, `wet_dry_qy_halo` checks the cross-strip pair using refreshed depths and bed elevations. `wet_dry` also updates the running maximum depth.

This is distinct from `update_cells` zeroing momentum below `hextra`. Moving coupling after wet/dry would leave exchange-written depths unchecked for a step and make partition-seam behavior inconsistent.

Sources: [wet/dry and maximum depth](../../src/kernels.h#L687-L748), [seam correction](../../src/kernels.h#L761-L795), [coupling ordering rationale](../../src/triton.h#L2550-L2556), [post-coupling enforcement](../../src/triton.h#L2649-L2687).

## Verification routes

CTest registers paraboloid, circular dam-break and Allatoona simulations with corresponding raster comparison jobs; fixture dependencies arrange clean output, simulation, then comparison. These are integration regression routes, not isolated proofs of each formula or backend combination. See [regression and resume testing](../testing/regression-and-resume.md) for oracle boundaries and [configuration and inputs](../workflows/configuration-and-inputs.md) for file contracts. [SWMM coupling](../integrations/swmm-coupling.md) owns the separate sewer-exchange law.

Source: [simulation regressions and ordering](../../test/CMakeLists.txt#L27-L89).

---
type: integration
title: Surface-sewer coupling contract
description: Coupling-node declarations, grid and unit mapping, shared-cell surface exchange, collective execution, and fail-fast SWMM diagnostics.
tags: [swmm, coupling, hydraulics, mpi, conservation]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-d44494ef3e497fea81240ef8
    resource: repo://CMakeLists.txt
  - id: openwiki-source-3d4824998a7387e308702cb7
    resource: repo://external/swmm/src/solver/routing.c
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-c26b636827b23cd5791d2bcb
    resource: repo://src/swmm_triton.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-47ce21353e1f2c4629fd35a5
    resource: repo://test/coupled_resume_suite.sh
  - id: openwiki-source-c82914037a426374077d0a2b
    resource: repo://test/unit/test_coupled_cells_review.cpp
  - id: openwiki-source-58de73433e28ad47e92e203c
    resource: repo://test/unit/test_coupled_cells.cpp
  - id: openwiki-source-1bcf56ac81551ab2c8834323
    resource: repo://test/unit/test_coupled_list.cpp
  - id: openwiki-source-bff32be1ce50d69defd849cb
    resource: repo://test/unit/test_coupling_declaration.cpp
  - id: openwiki-source-56d2799c89ed33060f1c34bf
    resource: repo://test/unit/test_zero_node_refusal.py
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Surface-sewer coupling contract

TRITON computes surface/sewer exchange on the ranks owning the coupled cells; rank zero owns the SWMM engine. The interface is positional: every flow slot and returned depth must refer to the same ordered coupled node. Node selection, spatial mapping, unit conversion, and collective participation are therefore correctness contracts, not optional input conveniences.

## Build and activation

`TRITON_ENABLE_SWMM=ON` defines `TRITON_SWMM`, adds bundled solver include paths, and links `swmm5`. Runtime coupling uses `inp_filename`, `manhole_diameter`, `manhole_loss`, `swmm_coupling_nodes`, and optional `swmm_node_rim_from_dem`; snapshot controls belong to the restart workflow.

A coupled binary with an empty `inp_filename` explicitly runs surface-only and warns on rank zero. In contrast, an `.inp` that resolves to **no coupling node** is rejected after SWMM input/start validation. It must not masquerade as a successful coupled run that never advances the sewer. The startup refusal names the declaration used and directs surface-only callers to omit `inp_filename`.

Sources: [build switch/linking](../../CMakeLists.txt#L61-L98), [configuration fields](../../src/config_utils.h#L527-L536), [surface-only branch](../../src/triton.h#L428-L451), [zero-node refusal](../../src/swmm_triton.h#L1264-L1278).

## Declaring the coupled node set

The `swmm_coupling_nodes` branches are exclusive:

| Value | Selected nodes |
|---|---|
| Path to a list | Exactly the named nodes, reordered into `[JUNCTIONS]` order; not the union with `[INFLOWS]` |
| Exact string `ALL_JUNCTIONS` | Every parsed junction in file order |
| Missing or empty | Legacy derivation from qualifying `[INFLOWS]` FLOW lines, deduplicated case-insensitively and ordered by junction position |

List files contain one name per line, no header. Surrounding whitespace is removed; blank lines and lines beginning with `;` are skipped. Remaining line content is the name, so this is not the `.inp` tokenizer and inline list comments are not stripped. Repeated names after case folding are rejected with both line positions. A nonempty value other than the exact keyword is treated as a filename.

The `.inp` reader strips from the first semicolon, splits whitespace with quoted-token handling, and recognizes case-insensitive section prefixes such as `[JUNC`, `[COORDINATE`, `[INFLOW`, and `[POLLUT`. It reads repeated sections rather than stopping at blank lines. Legacy classification first gathers pollutant names; a constituent matching a pollutant is not FLOW even if named `FLOW`. Otherwise a case-insensitive `FLOW` prefix qualifies, including `FLOWRATE`.

Sources: [tokenizer and section reader](../../src/swmm_triton.h#L151-L243), [ordering and legacy FLOW derivation](../../src/swmm_triton.h#L245-L292), [resolution/list format](../../src/swmm_triton.h#L303-L362), [runtime declaration selection](../../src/swmm_triton.h#L823-L853).

## Engine validation and slot ownership

Rank zero resolves every selected name against the started engine and refuses unknown nodes or anything other than JUNCTION. Engine indices must strictly increase. Legacy mode additionally verifies each selected node has a FLOW inflow and that the selected count equals the engine's FLOW set. Declared modes intentionally disable those FLOW-membership/count restrictions: a declared junction need not have an external FLOW series.

`routing_setCouplingNodes` builds the engine node-index-to-exchange-slot map before live stepping or restoration. That same map controls exchange injection and depth export. A declared node's existing FLOW series remains an ordinary model input and is **added** to its exchange; it neither changes selection nor disappears. Uncoupled nodes still receive their ordinary external inflows.

Sources: [engine checks](../../src/swmm_triton.h#L1201-L1262), [slot activation/disclosure](../../src/swmm_triton.h#L1281-L1310), [slot-map construction](../../external/swmm/src/solver/routing.c#L1064-L1089), [flow addition](../../external/swmm/src/solver/routing.c#L486-L526), [depth export](../../external/swmm/src/solver/routing.c#L275-L315).

## Grid placement, units, and rims

Coordinates are used as read and must already share the DEM coordinate system; SWMM flow-unit selection does not transform map coordinates. Declared nodes missing a `[COORDINATES]` entry are refused; the legacy branch retains its older missing-coordinate behavior. Cell indices use the implemented `ceil(... + 1e-16) - 1` rules, with the Y calculation reversing from lower-left coordinates into raster rows. Out-of-domain positions are rejected. Partition row counts determine the owning rank, and ghost offsets determine the local flattened cell index.

The exchange kernel uses metres, seconds, and cubic metres per second internally. It books SWMM exchange as `-flow / FT3_TO_M3_FACTOR` in cubic feet per second, positive into the sewer; exported engine depths convert feet to metres. Engine elevation/full-depth getters for CFS, GPM and MGD models are converted from feet with `0.3048`; SI getter lengths are unchanged. Coordinates do not undergo this conversion.

Default exchange rim depth comes from the engine's full depth, so a crown-raised junction is not incorrectly coupled at its raw input depth. For SI models, the parsed depth token is retained when it agrees within the implemented relative tolerance, preserving benign last-bit behavior. US models always use converted engine depth. With `swmm_node_rim_from_dem=1`, the model replaces the exchange rim depth with DEM elevation minus engine invert; nonfinite/NODATA cells or cells at/below invert are collectively rejected. This changes the exchange-side rim array, not the engine getter contract.

The configured manhole diameter and loss coefficient are assigned to each selected node. Diameter greater than cell size emits a warning, not a hard refusal.

Sources: [coordinates and missing-coordinate boundary](../../src/swmm_triton.h#L863-L906), [grid ownership](../../src/swmm_triton.h#L939-L1029), [cell-index formulas](../../src/swmm_triton.h#L2117-L2125), [units/rim helpers](../../src/swmm_triton.h#L424-L459), [engine rim/invert retrieval](../../src/swmm_triton.h#L1205-L1235), [DEM rim override](../../src/triton.h#L530-L573), [diameter/loss assignment](../../src/swmm_triton.h#L920-L929).

## Exchange law and shared-cell limits

Let surface depth be `h`, sewer depth be `d`, and exchange rim depth be `r`. The helper's signed `flow` is positive **into the surface**:

- Wet surface with `d <= r`: negative weir-type drainage using manhole perimeter.
- Wet surface with `r < d <= r + h`: negative orifice drainage from surface to sewer.
- `d > r + h`: positive orifice return from sewer to surface.

Every exchange becomes a cell-depth increment `flow * dt / cell_area`; the booked SWMM rate reverses sign and converts volume units.

Several nodes may occupy one surface cell. `build_coupled_cell_index` groups them, and the device kernel launches one thread per **cell**, avoiding competing writes. Every node in a shared cell evaluates its raw flow against the same pre-exchange depth. If their net exchange would make depth negative, only surface-draining flows are scaled together, using the prior cell water plus same-step sewer returns; incoming flows are not rationed. Scaling is clamped to `[0,1]`, depth becomes zero on that negative-depth branch, and momentum is zeroed below `hextra`. Each node retains its own booked exchange.

A single-node cell takes the preserved single-node arithmetic path. Its tiny-depth clamp uses `EPS12` inside the `hextra` branch and books the exact removed cell volume. The several-node branch instead activates volume limiting only below zero; do not infer identical near-zero arithmetic between these paths.

Sources: [flow cases](../../src/swmm_triton.h#L2164-L2182), [cell grouping](../../src/swmm_triton.h#L461-L484), [single-node branch](../../src/swmm_triton.h#L2195-L2247), [shared-cell limiter and launch](../../src/swmm_triton.h#L2250-L2292).

## Gather, step, scatter, enforce

At each globally coupled step, local node-owning ranks transfer previous sewer depths to device, evaluate exchange, and return per-node rates to host. All ranks gather rates, including zero-local-node ranks. Rank zero reorders rank-packed rates into global node order, logs the step, calls checked `swmm_step`, and reorders returned depths for scatter. Every rank participates in that scatter. Only per-rank buffer access is guarded by the local node count; the collective region is guarded by the global node count.

This exchange precedes surface wet/dry enforcement and halo correction. Persistence records the same step stream, but snapshot and replay details belong to [checkpoint and resume](../workflows/checkpoint-and-resume.md) and [snapshot architecture](../architecture/swmm-state-snapshots.md).

**Combination limits:** the live collectives use `ENSIFY_COMM_WORLD`, but startup count/index gathers in `init_swmm` use `MPI_COMM_WORLD`. Do not assume coupled ensemble isolation. Dynamic repartition rebuilds generic device arrays without rebuilding SWMM ownership/coupling buffers; see [parallel execution](../architecture/parallel-execution.md).

Sources: [live exchange sequence](../../src/triton.h#L2557-L2687), [rank/global reordering](../../src/swmm_triton.h#L2128-L2140), [startup communicator choices](../../src/swmm_triton.h#L1050-L1074), [repartition rebuild](../../src/triton.h#L2719-L2750).

## Startup and runtime failures

Before `swmm_open`, rank zero rejects input/report/output names exceeding `MAXFNAME`, checking the input as given and its resolved form where applicable. `swmm_open`, `swmm_start`, and live `swmm_step` return codes reach `stop_on_swmm_error`. Nonzero results print the API call, code, and engine message and abort `ENSIFY_COMM_WORLD`, preventing peers from hanging at later collectives.

Startup calls additionally close SWMM to flush its report and echo actual `ERROR {digits}` lines; per-line input errors also include the rejected following input line. A report that cannot be read produces an explicit diagnostic. Step failures do not close mid-routing through this startup-only path. Normal finalization closes exchange history, ends SWMM, writes its report, and closes it on rank zero.

Sources: [path and startup checks](../../src/swmm_triton.h#L1152-L1192), [checked step](../../src/triton.h#L2623-L2634), [error/report helpers](../../src/swmm_triton.h#L486-L540), [abort behavior and finalization](../../src/swmm_triton.h#L1339-L1377).

## Tests and older README restrictions

The README's claims that coincident nodes cause an error and that coupling should be restricted to one or two MPI ranks are not the current implementation contract. Shared-cell mapping is accepted and tested. The coupled resume suite contains eight-rank SEQ/PAR comparison legs and a sixteen-rank completion/report regression where rank zero owns no manhole. These are executable scenarios, not a universal scaling or backend guarantee, and they were not run as part of writing this page.

Focused tests cover declaration precedence/list parsing, legacy token/section behavior, engine-order checks, rim/unit helpers, shared-cell volume accounting, and startup refusal when no nodes are coupled. The shared-cell review test targets conservation when the would-be depth is a tiny **nonnegative** value below `EPS12`. Its commentary describes an older zeroing condition; the current kernel limits only negative depth. Read the actual condition rather than treating that comment as a current failure report. The review assertion measures surface volume change plus booked sewer volume against a scale-aware tolerance, not full-network mass conservation.

Sources: [older README wording](../../README.md#L137-L142), [declaration assertions](../../test/unit/test_coupling_declaration.cpp#L88-L148), [parser/unit assertions](../../test/unit/test_coupled_list.cpp#L125-L158), [rim/unit assertions](../../test/unit/test_coupled_list.cpp#L275-L326), [shared-cell assertions](../../test/unit/test_coupled_cells.cpp#L127-L235), [review fixture and residual](../../test/unit/test_coupled_cells_review.cpp#L52-L115), [zero-node runtime fixture](../../test/unit/test_zero_node_refusal.py#L2-L35), [MPI suite](../../test/coupled_resume_suite.sh#L158-L203).

Continue with [build/backends](../operations/build-and-backends.md), [input configuration](../workflows/configuration-and-inputs.md), [surface numerics](../concepts/shallow-water-numerics.md), or [regression and resume testing](../testing/regression-and-resume.md).

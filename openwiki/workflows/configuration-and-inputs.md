---
type: workflow
title: Preparing configuration, rasters, and forcing
description: Prepare a cfg and compatible grid/forcing inputs using the actual parser, path, unit-conversion, coordinate-mapping, and boundary contracts rather than assuming comprehensive startup validation.
tags: [configuration, inputs, rasters, forcing]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-aa1ee9798c4a664b9278054e
    resource: repo://doc/configuration_reference.rst
  - id: openwiki-source-34648547d88511ebf4f3016d
    resource: repo://src/config_utils.h
  - id: openwiki-source-f7749a16559cdcc937155ed0
    resource: repo://src/dem_utils.h
  - id: openwiki-source-31bb4be3ac28853eaf5bca8e
    resource: repo://src/extbc.h
  - id: openwiki-source-bdbe4d036fb8d6bb3ba39aff
    resource: repo://src/inflow.h
  - id: openwiki-source-c0337859aaa213a6523dbbc6
    resource: repo://src/kernels.h
  - id: openwiki-source-cef92bf761507f540f5c5b62
    resource: repo://src/matrix.h
  - id: openwiki-source-fbe082bc4fe57162918ea012
    resource: repo://src/output.h
  - id: openwiki-source-8c8851b95d1dd205c6313010
    resource: repo://src/string_utils.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-bf72216494b5e09a47393d71
    resource: repo://test/allatoona.cfg
  - id: openwiki-source-6961ff65795652b927a9b6c4
    resource: repo://test/circular_dambreak.cfg
  - id: openwiki-source-bdbd350f7b09dfbcb292e92a
    resource: repo://test/paraboloid.cfg
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Preparing configuration, rasters, and forcing

## Preparation workflow

Choose a sample cfg matching the intended forcing, then prepare its referenced inputs in the launch directory's path layout. The Allatoona test cfg combines point hydrographs, a roughness raster, runoff zones, and external boundaries. Paraboloid uses initial H/QX/QY rasters and observations without hydrologic forcing; circular dambreak supplies initial depth with empty initial discharge paths. These are concrete examples, not schemas or validation guarantees.

For a new case:

1. Establish a projected, meter-based DEM grid and aligned ancillary rasters.
2. Select raster encoding (`input_format`) separately from distribution (`input_option`).
3. Set roughness and initial surface state; preserve the distinction between initialization files and a checkpoint restart.
4. Match source/zone counts and coordinate order to forcing columns; prepare times and rates in the reader's input units.
5. Add observations and explicit edge segments only where needed.
6. Choose simulation end time, print settings, and a complete output set suitable for any intended restart.

Evidence: [Allatoona cfg](../../test/allatoona.cfg), [paraboloid cfg](../../test/paraboloid.cfg), [dambreak cfg](../../test/circular_dambreak.cfg), [setup guidance](../../doc/simulation_setup.rst).

## Cfg syntax, duplicates, and defaults

`ConfigUtils::parse_cfg` trims each whole line, ignores empty lines and lines starting with `#`, splits on `=`, removes a surrounding pair of double quotes from the first value token, and inserts the pair into a `std::map`. **The first occurrence of a duplicate key wins** because insertion does not overwrite an existing key. Appending a changed setting to a cfg that already has it is not an override mechanism.

This is not an INI, YAML, or general quoted-string parser:

- Individual key/value tokens are not trimmed by `StringUtils::split`; use `key=value`, not spaces around the equals sign.
- Inline `#` handling tests the preceding character rather than tracking quoted-string state; do not assume an arbitrary quoted `#` is protected.
- Extra `=` tokens are concatenated without restoring separators. Avoid equals signs inside values.
- Malformed lines are indexed before a complete token-count check. The parser is not a safe diagnostic layer for arbitrary malformed input.

`args` returns an empty string for a missing key; `argsd` returns its explicit fallback. Many numeric settings use `atoi`/`atof`, so an absent or malformed value is not automatically reported as a missing required setting. Unknown keys remain unused rather than being checked against a schema. Names and format selectors are case-sensitive except where conversion is explicit: `print_option` is lowercased, but `max_value_print_option` is not.

Selected real defaults include `checkpoint_id=0`, `it_count=0`, `it_print=10000`, empty optional paths, `input_folder=input`, `output_folder=output`, and observation print interval falling back to the raster print interval. Empty/missing `input_option` becomes `SEQ` for legacy cfgs. SWMM-specific keys are parsed only in a coupled build; snapshot policy and DEM-rim booleans default off, while `swmm_coupling_nodes` defaults empty. Use [SWMM coupling](../integrations/swmm-coupling.md) for that separate input contract.

Evidence: [parser and defaults](../../src/config_utils.h#L227-L292), [typed cfg population](../../src/config_utils.h#L476-L582), [untrimmed split](../../src/string_utils.h#L197-L213).

## Paths: input filenames are not relative to the cfg

The selected cfg is opened as a supplied filename. DEM, roughness, forcing, source/observation, and boundary-list filenames are passed directly to file readers. Relative values therefore depend on the process working directory, not the cfg's parent directory and not an automatic `input_folder` prefix. Type-1 boundary files are an exception: the boundary parser prepends `extbc_dir` to their filename.

Output/checkpoint paths use an executable-derived project root. `get_root_dir` makes the invocation path absolute when needed, then strips two path components without symlink resolution. Several output and SWMM paths concatenate that root and `output_folder`; keep this layout separate from ordinary working-directory input paths. The configuration documentation's global-folder language is not evidence that every path is rebased uniformly.

The test cfgs deliberately use paths such as `../input/...`, matching their configured working directory. A cfg copied elsewhere does not carry its own path base with it. See [checkpoint and resume](checkpoint-and-resume.md) before moving output trees.

Evidence: [cfg opening](../../src/triton.h#L631-L650), [direct forcing and DEM opens](../../src/triton.h#L686-L753), [coordinate readers](../../src/config_utils.h#L295-L355), [boundary path join](../../src/config_utils.h#L387-L451), [project root](../../src/config_utils.h#L608-L645).

## Raster encoding and distribution

| Setting | Actual reader behavior |
|---|---|
| `input_format=ASC` | Selects text readers. Other values follow binary branches; this is not general support for additional format names. |
| `input_option=SEQ` | Reads a global DEM header and rank-zero global raster data, then partitions/scatters it. |
| `input_option=PAR` | Reads the global header from `header_filename` as ASCII regardless of raster encoding, then reads rank-suffixed raster pieces. |

A sequential ASCII DEM has the conventional six-line header ordered as ncols, nrows, lower-left x, lower-left y, cell size, and NODATA, followed by rows of elevation. The header loader selects fields by **line position**, taking each line's final space-separated token; it does not validate the header labels as a schema. Binary DEM header values are read as the configured template value type in the same ordered slots.

ASCII roughness, runoff, and initial-condition rasters are headerless. The ordinary binary matrix reader expects its own row/column header before values; the full binary DEM path instead skips the DEM header width. Binary inputs therefore depend on the configured numeric representation; they are not interchangeable with arbitrary GIS binaries. GeoTIFF is an output integration, not an input branch here.

PAR paths append the padded rank and fixed extension: `{dem_prefix}_{rank}.dem`, `{roughness_prefix}_{rank}.mann`, `{runoff_prefix}_{rank}.rmap`, and `{initial_field_prefix}_{rank}.out`. Supply prefixes rather than already-suffixed filenames. Rank input pieces exclude ghost cells; initialization adds those and exchanges data. Dynamic restarts can read the regular pieces before redistribution to saved row counts.

Evidence: [DEM header parsing](../../src/dem_utils.h#L162-L294), [global reads](../../src/triton.h#L717-L847), [PAR DEM/roughness](../../src/triton.h#L905-L994), [PAR initial and runoff reads](../../src/triton.h#L1018-L1121), [binary readers](../../src/matrix.h#L1197-L1255).

### Grid and validation limits

Match grid dimensions, resolution, origin, alignment, and physical units before launching. This is a setup obligation, not a comprehensive startup validator. Headerless ASCII reads check matrix dimensions; binary matrices check their stored dimensions. Header-skipping DEM reads allocate the expected shape and read directly. None of these readers establishes CRS or geospatial alignment for ancillary grids.

ASCII data split on literal spaces, with an explicit too-many-columns diagnostic suggesting repeated separators. Values containing a decimal point use `atof`; other tokens use `atoi`. Use ordinary decimal notation and single-space-separated rows, not arbitrary whitespace or bare exponent-only numeric formatting. Binary reads do not uniformly verify complete payload reads.

The DEM's NODATA value is retained as metadata, but the shown matrix path loads elevation values literally rather than building a masked grid. Prepare valid terrain instead of expecting a NODATA mask to exclude cells. No raster reprojection occurs in this input path.

Evidence: [ASCII and binary matrix readers](../../src/matrix.h#L1049-L1255), [DEM metadata](../../src/dem_utils.h#L149-L158), [DEM loading](../../src/triton.h#L717-L762).

## Roughness and initial state

A nonempty `n_infile` supplies spatial Manning coefficients; otherwise the solver fills a constant `const_mann` raster. It adds ghosts, copies edge roughness, and squares coefficients for the numerical friction path. Supply Manning n, not pre-squared values.

Optional `h_infile`, `qx_infile`, and `qy_infile` initialize depth and the two unit-discharge components. They do not represent velocities despite `u`/`v` appearing in output selectors and internal variable names. Empty initial fields are zero-filled for a clean start. A positive checkpoint ID instead replaces this state from saved binary outputs and ghost rings, with historical maximum depth handled by its own output option. Initial-condition files alone are not a complete coupled resume.

Evidence: [roughness and initial readers](../../src/triton.h#L764-L834), [clean-state fallback](../../src/triton.h#L1764-L1792), [checkpoint state replacement](../../src/triton.h#L1182-L1192), [source depth/volume use](../../src/kernels.h#L891-L921).

## Point hydrographs and coordinates

Point forcing uses `num_sources`, `hydrograph_filename`, and `src_loc_file`. Hydrograph rows are comma-separated time followed by source discharge columns, with `%` starting the ignored remainder. Startup converts time from hours to seconds but leaves point discharge unchanged; the intended discharge unit is cubic meters per second. The source-location reader consumes comma-separated x,y pairs in the same order as those columns, skipping `%` comment lines.

Coordinates use the DEM's lower-left origin and cell size. The concrete mapping is `col = ceil((x-xll)/dx + 1e-16)-1` and `row = ceil(nrows-(y-yll)/dx + 1e-16)-1`, followed by rank-row and ghost-padding offsets. Rows thus run from the top of the grid, unlike y coordinates increasing upward. Use interior cell coordinates rather than assuming arbitrary outer-edge points map into the domain. Out-of-domain source cells terminate startup.

The local source table retains original source-column indices after row-strip ownership assignment. At runtime the selected hydrograph interval is linearly interpolated at `simtime`, then adds `Q*dt/dx²` to the owning surface cell's depth. Count consistency, strictly ordered/nonduplicate timestamps, and usable time coverage are preparation requirements; the shown parser does not provide a full table-schema validation pass. Avoid coincident point sources unless their shared-cell update behavior has been separately established.

Evidence: [tabular reader and conversion](../../src/inflow.h#L175-L295), [source location parser](../../src/config_utils.h#L295-L355), [source ownership and bounds](../../src/triton.h#L1270-L1375), [coordinate formulas](../../src/triton.h#L2142-L2151), [interval selection](../../src/triton.h#L2513-L2533), [source kernel](../../src/kernels.h#L891-L921).

## Distributed runoff is interval-indexed, not the point interpolation path

`num_runoffs`, `runoff_filename`, and `runoff_map` define zone forcing. The runoff series uses hours and mm/hour on input; startup converts times to seconds and intensities to meters/second. Interior runoff-map values index zone columns starting at zero. The host table is laid out by zone then time, and the kernel directly uses `zone_id * number_of_times + current_row` before adding intensity times dt to depth.

Unlike point hydrographs, runoff uses the selected row's intensity, not linear interpolation. The simulation advances its runoff index when passing the next timestamp and holds the last row after the end. Prepare adequate temporal sampling and valid IDs in `[0, num_runoffs)`; the kernel shown has no range guard for map IDs. Streamflow and runoff may both be enabled, so avoid counting the same water twice.

Evidence: [conversion](../../src/triton.h#L700-L708), [zone table construction](../../src/triton.h#L1846-L1902), [runtime interval selection](../../src/triton.h#L2495-L2510), [runoff kernel indexing](../../src/kernels.h#L943-L975), [zone/unit guidance](../../doc/configuration_reference.rst#L294-L365).

## Observation mapping

`time_series_flag` enables parsing `observation_loc_file` using the same x,y reader. Observation cells use the same coordinate formulas and row-strip ownership mapping as sources; `print_observation` controls their output interval. A critical validation difference: the shown out-of-bounds observation branch prints an error but does **not** exit there, unlike the source branch. Check observation locations before launching rather than treating the message as a guaranteed clean refusal.

Evidence: [observation parse gate](../../src/config_utils.h#L555-L560), [observation mapping](../../src/triton.h#L1381-L1454), [interval default](../../src/config_utils.h#L539-L543).

## Global edges and explicit external boundaries

`open_boundaries=0` initializes exterior DEM ghosts as walls; a true setting copies boundary terrain/state. Explicit boundary segments are applied after the general open-boundary copy and can replace behavior on selected edges.

`num_extbc` controls how many rows of the comma-separated boundary declaration are consumed. Each row supplies type,x1,y1,x2,y2 and a type-dependent value. Nonzero types require the extra field; type 1 joins it to `extbc_dir` and reads a time-level table. Endpoint coordinates are mapped into cells, checked for edge membership/alignment by `ExtBC`, expanded into a straight edge segment, then assigned to rank-local cell lists. This is not arbitrary interior polyline rasterization.

| Type | Input and numerical meaning |
|---|---|
| 0 | No external parameter file; copy adjacent interior H/QX/QY into ghosts, including corner handling. |
| 1 | Time in hours and **water-surface level**, not discharge or depth. Times become seconds; the kernel interpolates at the timestep midpoint and computes nonnegative depth as level minus DEM elevation. |
| 2 | Scalar normal-slope parameter used with local depth and squared Manning coefficient to determine boundary velocity/discharge. |
| 3 | Scalar Froude number; boundary velocity scales as Froude times `sqrt(g*h)`. |

Do not infer complete numeric/range validation from these types. The endpoint checker enforces specific upper-bound, perimeter and alignment conditions, not a full geospatial schema; unsupported type codes and malformed/undersized tables are not a validated extension mechanism. Use ordered time series extending safely beyond the simulated midpoint times rather than relying on every equality endpoint case.

Evidence: [declaration parser](../../src/config_utils.h#L359-L472), [boundary object and endpoint checks](../../src/extbc.h#L138-L292), [hours conversion](../../src/extbc.h#L329-L337), [rank setup](../../src/triton.h#L1467-L1530), [application order](../../src/triton.h#L2537-L2546), [type behavior](../../src/kernels.h#L1169-L1262).

## Timing and verification boundary

Simulation control uses seconds after input conversion. `sim_duration` is the ending simulation time, not an extra duration added to `sim_start_time`; checkpoint restarts retain that distinction. `time_increment_fixed`, `time_step`, `courant`, and `hextra` affect the solver and wet/dry path; see [shallow-water numerics](../concepts/shallow-water-numerics.md). Output naming, selected fields, and print intervals are covered by [outputs and diagnostics](outputs-and-diagnostics.md).

No cfg execution or raster-payload validation was performed during page generation. Large sample rasters and excluded `*.out` files were not opened. Source readers establish the behavior described here; documentation recommendations do not imply startup checks absent from those readers.

- [Quickstart](../quickstart.md)
- [Shallow-water numerics](../concepts/shallow-water-numerics.md)
- [SWMM coupling](../integrations/swmm-coupling.md)
- [Checkpoint and resume](checkpoint-and-resume.md)
- [Outputs and diagnostics](outputs-and-diagnostics.md)

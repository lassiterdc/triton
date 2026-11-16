---
type: quickstart
title: Quickstart and task routing
description: Build and launch the supplied surface-model sample, find its outputs, and navigate the TRITON architecture, coupling, restart, ensemble and verification guides.
tags: [quickstart, navigation, build, simulation]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-ed77bc20b48d193bace3dde6
    resource: repo://.gitmodules
  - id: openwiki-source-a79e6b9a2f1425d8d8ab6675
    resource: repo://cmake/machine.cmake
  - id: openwiki-source-3c65b3aef9a0f2e2814d30bb
    resource: repo://cmake/machines/Linux/default_SERIAL.sh
  - id: openwiki-source-8a69e9dd86720528c61788e4
    resource: repo://cmake/util.cmake
  - id: openwiki-source-d44494ef3e497fea81240ef8
    resource: repo://CMakeLists.txt
  - id: openwiki-source-af7fe2cb8aeceee6c5cdf227
    resource: repo://external/CMakeLists.txt
  - id: openwiki-source-b52bbcc67304215b8765334d
    resource: repo://input/paraboloid/paraboloid.cfg
  - id: openwiki-source-34648547d88511ebf4f3016d
    resource: repo://src/config_utils.h
  - id: openwiki-source-8ca55e77f7850d2615133c5b
    resource: repo://src/Ensify.h
  - id: openwiki-source-d364d949938a433276255c32
    resource: repo://src/main.cpp
  - id: openwiki-source-fbe082bc4fe57162918ea012
    resource: repo://src/output.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-4b8e7dce368774e92d99ea30
    resource: repo://test/CMakeLists.txt
  - id: openwiki-source-bdbd350f7b09dfbcb292e92a
    resource: repo://test/paraboloid.cfg
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Quickstart and task routing

Start with a surface-only paraboloid case, then select the workflow matching your inputs or coupled model. CMake builds `triton.exe`; the root Makefile is not the solver build entrypoint. The executable initializes MPI and Kokkos, constructs one model per communicator, initializes it, runs the simulation, and finalizes the libraries.

This guide's commands are established by current build/wrapper source. They were **not executed during this wiki-generation session**; wiki submission/finalization is documentation validation, not a solver build or numerical verification result.

## Prerequisites

- CMake 3.16 or newer, a C++17-capable toolchain with MPI headers/libraries and an MPI launcher, and Python 3. The root CMake project requires the Python interpreter during configuration.
- Populated `external/kokkos` submodule. `external/yaml-cpp` is additionally needed for ensemble builds; both are declared in `.gitmodules`.
- A compiler/backend profile appropriate to the host. Profiles configure the compiler, flags, launcher and Kokkos settings before CMake's `project` step. Optional GDAL discovery enables GeoTIFF; the basic ASCII/binary sample does not require it.

See [Build configuration, backends, and deployment](operations/build-and-backends.md) for machine detection, GPU architectures, manual configuration, dependencies, containers and documentation builds. Do not copy an accelerator profile without matching the target environment.

## Build and run a Linux CPU example

From the repository root, the following explicitly chooses the checked-in generic Linux SERIAL profile rather than relying on host autodetection. SERIAL selects the local Kokkos backend; it does not remove MPI from the program.

```bash
cmake -S . -B build-quickstart \
  -DMACHINE="$PWD/cmake/machines/Linux/default_SERIAL.sh" \
  -DBUILD_TESTS=ON
cd build-quickstart
./triton_build.sh
./triton_run.sh ./input/paraboloid/paraboloid.cfg "mpirun -n 1"
```

This profile uses `mpic++`, Kokkos `NATIVE` architecture, and a default launcher of `mpirun -n 8`. The example overrides only the run wrapper's launcher with one rank. The generated build helper sources the copied `triton_env.sh` and builds with the detected physical-core count; CMake creates the build-tree `input` symlink to the source samples.

Run wrappers **from their build directory**. They source a relative environment filename and launch `./triton.exe`. The first argument to `triton_run.sh` is the cfg, and the optional second argument is the whole MPI command string. With no arguments, the wrapper selects `./input/paraboloid/paraboloid.cfg` and the configured launcher. README text describing Allatoona as the wrapper default is not the wrapper implementation.

The sample cfg references its DEM, observation locations and initial H/QX/QY through `input/paraboloid/` paths. It uses no source hydrographs or runoff, constant zero Manning coefficient, variable timesteps, static decomposition, sequential ASCII output and `print_option=huv`. Its ending simulation time is `13.4571` seconds; raster and observation intervals are `1.121425` and `4.4857` seconds. These are sample configuration values, not measured runtime.

**Do not substitute `test/paraboloid.cfg` into this launch without checking paths.** CTest runs that separate cfg from `build/test`, where its `../input/` references and `output_paraboloid` destination are intentional. Input filenames are not automatically rebased to the cfg's directory.

## Find results and diagnose the first run

The input sample omits `output_folder`, whose parser default is `output`. For the build-directory launch above, inspect:

| Artifact | What it represents |
|---|---|
| `output/asc/` | Selected H, QX and QY numeric ASCII rasters |
| `output/bin/` | Companion binary rasters and per-rank ghost-ring checkpoint side files |
| `output/series/` | Observation time series |
| `output/cfg/config_{id}.cfg` | Rewritten cfg at a raster checkpoint |
| `output/log.out` | Rank-zero machine/build header and appended simulation timing line; not a copy of stderr |
| `output/performance/`, `output/performance.txt` | Checkpoint and final per-rank timer accounting |
| `cid` in the launch directory | Most recently written checkpoint ID; not a transaction marker for an entire restart set |

Console diagnostics include `Simulation starts`, output notices, periodic progress controlled by `it_print`, and `Simulation ends`. Keep stderr when diagnosing failures. Selected `u` and `v` output characters mean QX/QY discharge components, not automatically calculated velocity maps.

Read [Output artifacts and runtime diagnostics](workflows/outputs-and-diagnostics.md) before interpreting filenames, GTIFF mosaics or timers, and [Checkpoint artifacts and coupled resume](workflows/checkpoint-and-resume.md) before restarting. Preserve binary companions and boundary state even when the desired presentation format is ASCII or GeoTIFF.

## Run focused regression checks

The build command above enables `BUILD_TESTS`, which generates `triton_ctest.sh`. From the same build directory:

```bash
./triton_ctest.sh --output-on-failure -R '^BIT4BIT-PARABOLOID$'
```

The comparator's required CTest fixtures schedule clean output, its simulation and then comparison. Those simulations use the configured `RUN_COMMAND`, **not** the one-rank override supplied to the earlier run wrapper; this profile's default is eight ranks. Configure the launcher for the test environment as described in the build guide.

The `BIT4BIT` name is not a guarantee of bitwise equality; consult [Regression, coupling, and resume verification](testing/regression-and-resume.md) for oracle limits, empty-output hazards, coupled-only tests and external benchmark requirements. No tests are claimed to have passed by this guide.

## Choose the next task

| Task | Read next |
|---|---|
| Select compiler, backend, machine profile, GDAL or container build | [Build configuration, backends, and deployment](operations/build-and-backends.md) |
| Trace startup, state ownership, timestep orchestration or shutdown | [Simulation ownership and lifecycle](architecture/simulation-lifecycle.md) |
| Understand row-strip partitioning, host/device state, halos or load balancing | [MPI decomposition and device execution](architecture/parallel-execution.md) |
| Inspect flux ordering, friction, wet/dry enforcement, timestep or forcing equations | [Shallow-water timestep and forcing](concepts/shallow-water-numerics.md) |
| Prepare a cfg, rasters, hydrographs, runoff zones or external boundaries | [Preparing configuration, rasters, and forcing](workflows/configuration-and-inputs.md) |
| Interpret output formats, series, checkpoint IDs, run logs or performance columns | [Output artifacts and runtime diagnostics](workflows/outputs-and-diagnostics.md) |
| Launch multiple members in one MPI job | [Ensemble execution with Ensify](workflows/ensemble-runs.md) |
| Configure coupled junctions, units, rims or shared-cell exchange | [Surface-sewer coupling contract](integrations/swmm-coupling.md) |
| Preserve a checkpoint and choose coupled snapshot/replay recovery | [Checkpoint artifacts and coupled resume](workflows/checkpoint-and-resume.md) |
| Change SWMM snapshot serialization or investigate binary-output continuity | [SWMM snapshot format and output continuity](architecture/swmm-state-snapshots.md) |
| Choose/run CTest or standalone coupled resume checks | [Regression, coupling, and resume verification](testing/regression-and-resume.md) |
| Extend field discovery, triage inventory differences or verify serializer coverage | [Snapshot inventory and closure checks](testing/snapshot-inventory.md) |

## Optional modes and operational cautions

- `TRITON_ENABLE_SWMM=ON` adds bundled SWMM and links `swmm5`. Start from the [coupling contract](integrations/swmm-coupling.md), not the README's older node-collision/rank-count restrictions.
- `ENSEMBLE_BUILD=ON` adds yaml-cpp and generates `triton_ensrun.sh` **instead of** `triton_run.sh`. The executable then takes a YAML ensemble manifest, which supplies each member's model arguments; consult the [ensemble guide](workflows/ensemble-runs.md) for rank coverage, output separation and coupled-mode limits.
- `TRITON_IGNORE_MACHINE_FILES=ON` skips profile provisioning. Its branch does not create `ENVFILE` even though wrappers source it; use direct build/executable commands in your provisioned environment rather than assuming the wrapper environment exists.
- `triton_clean.sh` removes much more than object files: its generated list includes CMake state, executable, wrappers, sample outputs, logs and the build-tree input link. Preserve results first; do not use it as a harmless refresh command.

## Source anchors

- `CMakeLists.txt:1-19`, `CMakeLists.txt:61-107`, `external/CMakeLists.txt`, `.gitmodules`: prerequisites and optional target wiring.
- `cmake/machine.cmake:25-87`, `cmake/machine.cmake:95-97`, `cmake/machine.cmake:211-224`, `cmake/machines/Linux/default_SERIAL.sh`: profile selection and environment materialization.
- `cmake/util.cmake:23-121`: generated launch/build/test/clean helpers and sample symlink.
- `input/paraboloid/paraboloid.cfg`, `test/paraboloid.cfg`, `test/CMakeLists.txt:1-12`, `test/CMakeLists.txt:65-89`: sample paths and CTest fixtures.
- `src/config_utils.h:545-546`, `src/main.cpp:66-104`, `src/output.h:524-719`, `src/triton.h:2200-2403`: default output, runtime lifecycle and artifact scheduling.

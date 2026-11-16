---
type: workflow
title: Ensemble execution with Ensify
description: Build and launch YAML-defined TRITON ensembles, understand per-member MPI communicators and arguments, and separate simulation outputs from redirected streams.
tags: [ensemble, mpi, configuration, execution]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-8a69e9dd86720528c61788e4
    resource: repo://cmake/util.cmake
  - id: openwiki-source-d44494ef3e497fea81240ef8
    resource: repo://CMakeLists.txt
  - id: openwiki-source-af7fe2cb8aeceee6c5cdf227
    resource: repo://external/CMakeLists.txt
  - id: openwiki-source-8ca55e77f7850d2615133c5b
    resource: repo://src/Ensify.h
  - id: openwiki-source-d364d949938a433276255c32
    resource: repo://src/main.cpp
  - id: openwiki-source-dc493dccd92de1bd10eaf41e
    resource: repo://src/mpi_utils.h
  - id: openwiki-source-fbe082bc4fe57162918ea012
    resource: repo://src/output.h
  - id: openwiki-source-c26b636827b23cd5791d2bcb
    resource: repo://src/swmm_triton.h
  - id: openwiki-source-5e2a25d978c214969fec8491
    resource: repo://src/triton.h
  - id: openwiki-source-4b8e7dce368774e92d99ea30
    resource: repo://test/CMakeLists.txt
  - id: openwiki-source-f9923e005ae2c3d620247d22
    resource: repo://test/ensembles_ensify.yaml
  - id: openwiki-source-c8264fcd41cacbe9ffe1c639
    resource: repo://test/paraboloid_ens1.cfg
  - id: openwiki-source-56af69d1f891deac5ab40ced
    resource: repo://test/paraboloid_ens2.cfg
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Ensemble execution with Ensify

Ensify runs multiple model instances inside one MPI launch. It assigns parent ranks to member communicators, replaces each model's command-line arguments, and redirects C++ output streams. The model still follows the normal initialize/simulate lifecycle; ensemble membership changes its rank/size and communicator rather than introducing a separate solver.

## Build and launch

Configure with `-DENSEMBLE_BUILD=ON`. The root build links `yaml-cpp` and defines `ENSEMBLE_BUILD`; `external/CMakeLists.txt` adds the bundled YAML dependency only for this mode. This is independent of the SWMM build switch.

CMake's `add_build_and_run_scripts` generates `triton_ensrun.sh` instead of the ordinary `triton_run.sh`. The launcher sources the generated environment file, takes the YAML filename as its first argument and an MPI command string as its second, and invokes `./triton.exe` with the YAML path. Defaults are `./ensembles_ensify.yaml` and the configured `RUN_COMMAND`. Configure-time copies put the sample manifest and both member cfg files in the build directory, alongside an `input` symlink.

From an ensemble build directory, launch the bundled example with:

```bash
./triton_build.sh
./triton_ensrun.sh ./ensembles_ensify.yaml "mpirun -n 8"
```

The explicit rank count matches the sample manifest; the wrapper does not derive it from YAML. For a direct launch, source `./triton_env.sh` and pass the manifest, not a model cfg, as the executable's first argument. See [Build and backends](../operations/build-and-backends.md) for environment/compiler setup.

## Manifest and model arguments

The top-level `ensembles` sequence contains five-field sequences in this order:

| Field | Meaning |
|---|---|
| `rank_beg` | First parent rank, inclusive |
| `rank_end` | Last parent rank, inclusive |
| `arguments_list` | Strings supplied after the model's executable name |
| `stdout_filename` | File attached to `std::cout` on every active rank |
| `stderr_filename` | File attached to `std::cerr` on every active rank |

The checked-in example is:

```yaml
ensembles: [
  [0, 3, ["./paraboloid_ens1.cfg"], "ens1.out", "ens1.err"],
  [4, 7, ["./paraboloid_ens2.cfg"], "ens2.out", "ens2.err"]
]
```

Each entry therefore receives four ranks. The decoder requires a sequence of exactly five elements and converts their types with yaml-cpp. Ensify takes only `argv[1]` as the manifest filename; later launcher arguments are not appended to member arguments.

Ensify constructs replacement model arguments from the original executable name followed by the entry's argument list. `main` passes those arguments to `Triton::triton`; its constructor uses `argv[1]` as the cfg path and, when present, `argv[3]` as a checkpoint override. Keep the intended model arguments inside YAML. Relative filenames retain the process working directory; Ensify does not change directories. See [Configuration and inputs](configuration-and-inputs.md).

## Communicator lifecycle

1. `main` initializes MPI and calls `ensify::init` before Kokkos initialization.
2. Every parent rank loads the manifest and iterates all entries in the same order.
3. For each entry, `MPI_Comm_split` partitions the parent communicator into active and inactive groups, ordered by parent rank. The active test is `rank_beg <= parent_rank <= rank_end`.
4. Active ranks retain that communicator, arguments, and stream files; inactive ranks immediately free their temporary communicator.
5. `main` queries rank/size from `ENSIFY_COMM_WORLD` and initializes/simulates the model. In ensemble builds this macro resolves to the retained member communicator; otherwise it resolves to `MPI_COMM_WORLD`.
6. After model destruction and Kokkos finalization, Ensify closes its streams and frees the retained communicator; then `main` finalizes MPI.

Surface-model halo exchange and output gathers use `ENSIFY_COMM_WORLD`, so member-local rank zero is the model's root. See [Parallel execution](../architecture/parallel-execution.md).

**Coupled-mode boundary:** not every SWMM operation is member-local. `swmm_triton::init_swmm` contains a node-count `MPI_Gather` and node-index `MPI_Gatherv` on literal `MPI_COMM_WORLD`, while nearby coupling broadcasts use `ENSIFY_COMM_WORLD`. Do not treat this ensemble pathway as evidence that independent coupled SWMM ensembles are isolated; the sample regression exercises surface-only paraboloid runs.

## Authoring invariants versus implemented checks

Author the manifest so every launched rank belongs to exactly one member, ranges do not overlap, endpoints fit the launch size, and all ranks see the same manifest. These are authoring requirements, not validations implemented by Ensify:

- It does not check parent communicator size against endpoints or detect missing coverage. An uncovered rank never receives a valid member communicator before `main` uses it.
- It does not reject overlapping ranges. A later matching entry overwrites the retained communicator, arguments, and stream objects without freeing the earlier active communicator.
- It does not check range ordering, enforce distinct paths, or test whether output streams opened successfully.

Missing the YAML argument raises `std::runtime_error`. YAML loading/conversion errors propagate; the explicit false-config check also throws. `mpicheck` returns on `MPI_SUCCESS` and otherwise formats the MPI error and throws `std::runtime_error` if control returns from MPI. `main` has no exception handler or coordinated recovery for these failures.

## Files and diagnostic limits

The member cfg files choose `output_ens1` and `output_ens2` with sequential ASCII output. Those cfg-owned output folders are separate from Ensify's `ens1.out`/`ens1.err` and `ens2.out`/`ens2.err`. Give each member distinct model output destinations and stream paths; Ensify does not namespace or rewrite them. See [Outputs and diagnostics](outputs-and-diagnostics.md).

Redirection attaches `std::cout`/`std::cerr` buffers, not process file descriptors. Each active rank independently opens the same member filename with an ordinary `std::ofstream`; there is no rank-zero-only writer or stream aggregation. Shared per-member logs can be truncated or interleaved across ranks and should not be treated as complete ordered diagnostics. C stdio and external runtime messages are not covered by this C++ stream-buffer redirection.

## Existing regression coverage

With `BUILD_TESTS` and `ENSEMBLE_BUILD` enabled, `ENSEMBLE_PARABOLOID` runs the generated launcher in the build directory, requires both `output_ens*/asc/QY_12_00.out` files to be nonempty, and compares them with `cmp -s`. This checks agreement of that selected raster between the two members; it does not establish manifest validation, log integrity, independent SWMM operation, or equality of every output. Unlike the ordinary simulation/comparator branch, this ensemble test has no clean-output fixture. See [Regression and resume](../testing/regression-and-resume.md).

## Source anchors

- `src/Ensify.h:20-100`: YAML conversion, split loop, streams, argument construction, failures and cleanup.
- `src/main.cpp:66-104`: initialization, member model construction and finalization ordering.
- `src/triton.h:363-378`, `src/mpi_utils.h:145-171`, `src/output.h:360-379`: model arguments and member-local MPI consumers.
- `src/swmm_triton.h:1057-1074`: world-communicator coupling gathers.
- `CMakeLists.txt:78-81`, `external/CMakeLists.txt:1-5`, `cmake/util.cmake:35-68`: build switch, dependency and launcher.
- `test/ensembles_ensify.yaml`, `test/paraboloid_ens1.cfg:14-18`, `test/paraboloid_ens2.cfg:14-18`, `test/CMakeLists.txt:14-25`: example and regression contract.

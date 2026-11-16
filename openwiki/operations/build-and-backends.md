---
type: build-guide
title: Build configuration, backends, and deployment
description: How CMake selects machine environments, compilers, Kokkos backends, optional integrations, and generated launch helpers; includes container and documentation boundaries.
tags: [build, cmake, kokkos, deployment]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-16T03:17:13.865Z
sources:
  - id: openwiki-source-ed77bc20b48d193bace3dde6
    resource: repo://.gitmodules
  - id: openwiki-source-a79e6b9a2f1425d8d8ab6675
    resource: repo://cmake/machine.cmake
  - id: openwiki-source-5fa12adbff020a936d595f6a
    resource: repo://cmake/machines/frontier/cray_HIP.sh
  - id: openwiki-source-336d381ba9686c37b12da0fb
    resource: repo://cmake/machines/Linux/default_CUDA.sh
  - id: openwiki-source-3c65b3aef9a0f2e2814d30bb
    resource: repo://cmake/machines/Linux/default_SERIAL.sh
  - id: openwiki-source-8a69e9dd86720528c61788e4
    resource: repo://cmake/util.cmake
  - id: openwiki-source-d44494ef3e497fea81240ef8
    resource: repo://CMakeLists.txt
  - id: openwiki-source-bb1ebe868e35e9e500714501
    resource: repo://Dockerfile
  - id: openwiki-source-af7fe2cb8aeceee6c5cdf227
    resource: repo://external/CMakeLists.txt
  - id: openwiki-source-012f2c78e3b1446dfc35803f
    resource: repo://Makefile
generated: { by: "omp", at: "2026-11-16T03:17:13.865Z" }
---

# Build configuration, backends, and deployment

## Build ownership and dependencies

The repository-root CMake project builds `triton.exe` from `src/main.cpp`, requires Python's interpreter during configuration, and requires C++17 without compiler extensions. It runs `set_environment()` **before** `project(triton CXX)` so machine settings participate in compiler discovery. Kokkos is always added and linked. `.gitmodules` declares Kokkos and yaml-cpp; initialize those submodules before configuring configurations that need them. yaml-cpp is added and linked only for `ENSEMBLE_BUILD`.

CMake, not the repository-root Makefile, builds the executable. Use an out-of-source build directory, configure it against the source root, then invoke `cmake --build` or the generated build helper. The root Makefile owns documentation and Docker convenience targets.

Evidence: [root build](../../CMakeLists.txt), [external dependencies](../../external/CMakeLists.txt), [submodules](../../.gitmodules), [Makefile](../../Makefile).

## Machine selection and precedence

Without `TRITON_IGNORE_MACHINE_FILES`, selection follows this sequence:

1. `MACHINE` comes from the CMake variable, otherwise `TRITON_MACHINE` in the environment, otherwise the host operating-system name.
2. An existing path supplied as `MACHINE` is used directly as a custom machine file.
3. When `MACHINE` equals the host OS name, Python obtains the FQDN. Each machine directory's `__machine__.cmake` may supply `FQDN_REGEX` and optional comma-separated `ENV_VALUES` constraints. One match selects that directory; multiple matches are fatal and require explicit `MACHINE`. No match leaves the OS directory selected.
4. The selected directory is searched for the filename stem `{compiler nickname}_{backend}`. Both selectors default to `default`. A missing file is fatal; there is no automatic search through alternative compiler/backend combinations.

The selected script is copied into the build directory as `triton_env` plus its original extension. Configuration sources the original script and parses its resulting environment. For imported `TRITON_*` settings, an already-defined CMake variable wins, then the calling process's corresponding environment variable, then the sourced script's value. `BACKEND=default` is explicitly eligible for replacement. `COMPILER` is special: its initial CMake value is saved as the profile nickname and removed from the cache, so `-DCOMPILER=gnu` selects a profile rather than directly naming a compiler executable. The actual compiler comes from the imported settings.

Machine scripts are executable configuration, not inert data: they can load modules and change paths. Non-`TRITON_` environment values from the sourced script are propagated into CMake's process environment. Required imported settings are `COMPILER`, `BACKEND`, and `RUN_COMMAND`.

Evidence: [machine selection and import](../../cmake/machine.cmake#L4-L314), [documented profile contract](../../doc/machine_file.rst).

## Compiler, flags, and backend

Machine processing resolves `CMAKE_CXX_COMPILER` with `find_program(COMPILER)`. It assigns `CMAKE_CXX_FLAGS` from `COMPILER_FLAGS` plus `COMPILER_FLAGS_APPEND`, and executable linker flags from the corresponding linker settings. Consequently, direct `CMAKE_CXX_FLAGS` input is not the authoritative flag channel through this macro. `DEBUG` adds `TRITON_DEBUG`.

Normal profile processing accepts CUDA, HIP, SYCL, OPENMP, OPENMPTARGET, THREADS, SERIAL, or `default`, then enables `Kokkos_ENABLE_{BACKEND}`. CUDA also enables `Kokkos_ENABLE_CUDA_CONSTEXPR`; `ARCH` enables `Kokkos_ARCH_{ARCH}`. Available profile files, rather than the accepted-name list alone, determine which combinations can be selected on a machine.

The generic Linux SERIAL profile chooses `mpic++`, `NATIVE`, and `mpirun -n 8`. The generic CUDA profile chooses `mpic++` and one MPI rank. Frontier's Cray HIP profile loads site modules, selects `CC`, `AMD_GFX90A`, `TRITON_HIP_LAUNCHER`, and an eight-rank `srun` launcher, with site-specific GDAL/PROJ/TIFF paths. These are profile defaults, not portable deployment guarantees.

Evidence: [compiler and backend application](../../cmake/machine.cmake#L283-L314), [Linux SERIAL](../../cmake/machines/Linux/default_SERIAL.sh), [Linux CUDA](../../cmake/machines/Linux/default_CUDA.sh), [Frontier HIP](../../cmake/machines/frontier/cray_HIP.sh).

### Manual configuration branch

`TRITON_IGNORE_MACHINE_FILES=ON` skips profile discovery, sourcing, and environment import. It supplies SERIAL when the backend is undefined or `default`, `mpic++` when the compiler is undefined, `mpirun -n 4` when the launcher is undefined, and empty architecture/flag defaults. It then applies the same compiler/flag assignments and Kokkos backend/architecture switches. Do not assume the profile-mode environment precedence operates in this branch.

There is a wrapper limitation: this branch does not materialize `ENVFILE`, while generated build/run/test scripts still emit a `source ./${ENVFILE}` line. For a manual build, invoke the executable and build tool directly in the environment you provisioned rather than assuming a generated environment script exists. No manual configuration or wrapper execution was tested for this page.

Evidence: [manual branch](../../cmake/machine.cmake#L25-L87), [wrapper generation](../../cmake/util.cmake#L23-L78).

## Optional integrations

| Switch or discovery | Build consequence |
|---|---|
| `TRITON_ENABLE_SWMM=ON` | Defines `TRITON_SWMM`, adds the bundled SWMM subdirectory if its CMake entry exists, adds solver include directories, and links `swmm5`; absence of that entry is fatal. |
| `TRITON_SWMM_FLOODING_DEBUG=ON` with SWMM enabled | Adds `SWMM_FLOODING_DEBUG` to `swmm5`. |
| `ENSEMBLE_BUILD=ON` | Adds yaml-cpp, defines `ENSEMBLE_BUILD`, and generates the ensemble launcher instead of the ordinary launcher. |
| GDAL found by `find_package(GDAL)` | Sets `ENABLE_GDAL`, adds include/library settings, and defines `TRITON_GDAL`. Missing GDAL produces a warning and disables this integration. |
| `BUILD_TESTS=ON` | Enables testing, adds the test subtree, and generates `triton_ctest.sh`. |

SWMM details here are limited to the allowed parent build files; its excluded build files were not inspected. See [SWMM coupling](../integrations/swmm-coupling.md), [ensemble runs](../workflows/ensemble-runs.md), and [regression and resume tests](../testing/regression-and-resume.md).

Evidence: [optional target wiring](../../CMakeLists.txt#L61-L107), [external selection and GDAL](../../external/CMakeLists.txt).

## Generated helpers and working directory

Run helpers from the build directory: their environment source, executable, input defaults, and test commands are relative to the current directory. `triton_build.sh` sources the environment and runs `cmake --build .` with the detected physical-core count. `triton_run.sh` accepts a cfg path as argument one and an MPI launcher string as argument two; defaults are the paraboloid cfg and configured `RUN_COMMAND`. The ensemble equivalent accepts a YAML manifest and copies the example manifest and member cfgs into the build tree. CMake also creates an `input` symlink to the source samples.

`triton_ctest.sh` forwards arguments to CTest. `triton_clean.sh` is broader than a compiler clean: its generated removal list includes CMake state, executable, helpers, external build products, sample output directories, ensemble logs, and the build-tree input link. Preserve wanted results before using it.

Evidence: [helper generation](../../cmake/util.cmake#L23-L121).

## Container and documentation deployment

`make docker_build` builds `triton:v1`; `make docker_run` mounts the host's `output` directory at `/app/triton/build/output` and launches the paraboloid sample. The Dockerfile uses separate builder and runner stages based on `mfisherman/mpich:4.2.3`, copies the source into `/app/triton`, installs CMake and GDAL packages, configures with `MACHINE=Linux`, and runs the generated build helper. `/project` wrappers change to the build directory before forwarding arguments; that directory is added to `PATH`. It does not initialize submodules itself, so the copied context must contain required dependencies. The recipe creates a ctest forwarding wrapper without explicitly enabling `BUILD_TESTS`; its existence alone does not establish that tests were configured.

`make doc` drives Doxygen XML followed by Sphinx HTML. It prefers `.venv_triton/bin/python` when present. These targets are separate from the solver compilation path.

This page records source inspection only; no compiler, accelerator, MPI launch, Docker build, or test run was performed.

## Related pages

- [Quickstart](../quickstart.md)
- [Parallel execution](../architecture/parallel-execution.md)
- [SWMM coupling](../integrations/swmm-coupling.md)
- [Ensemble runs](../workflows/ensemble-runs.md)
- [Regression and resume testing](../testing/regression-and-resume.md)

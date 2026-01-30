# TRITON: Two-dimensional Runoff Inundation Toolkit for Operational Needs

TRITON is an open-source, high-performance software framework for 2D flood simulation. Its core is a computationally efficient, physics-based hydraulic model that operates on a regular, structured grid and solves the full 2D shallow water equations.

## Features

- **Cross-platform and HPC ready**: Performance-portable via Kokkos. Runs on single or multiple CPUs (OpenMP + MPI) and supports GPU acceleration with Kokkos and MPI.
- **Flexible Forcing & Inputs** – Uses topographical data (e.g., DEM, LIDAR) on a uniform Cartesian grid and supports streamflow hydrographs, gridded runoff, or both as hydrological forcing.
- **Rich Output Options** – Produces water depth maps, 2D velocity maps, and unit discharge data, plus time series outputs at user-defined points and intervals.
- **Linux/Unix Native** – Built for Linux/Unix systems with input/output in ascii, binary and geotiff format.
- **SI Units Standard** – Operates using the International System of Units (SI).

![Simulation Output](doc/_static/TRITON_output_example.jpg)

## Repository Structure

```
triton/
├── doc/           # User guides, API references, and technical documentation
├── src/           # Core simulation source code
├── external/      # Kokkos and yaml-cpp Git submodules, bundled SWMM
├── input/         # Sample simulation input data files
├── test/          # Regression test suite based on CTest
├── cmake/         # CMake configuration modules and machine files
├── Makefile       # Commands to generate documentation and a TRITON Docker image
└── README.md      # Project overview and usage instructions
```

## Installation

TRITON can be built from source or run using a container.

### **Prerequisites**
CMake 3.16 or newer
C++17 compatible compiler
MPI libraries with C++ bindings
Optional: CUDA, HIP, or SYCL for GPU acceleration
Optional: GDAL for GeoTIFF support
Kokkos is included as a Git submodule. No system install is required.

### **Build Instructions**

```bash
# Clone TRITON
git clone --recursive https://code.ornl.gov/hydro/triton.git
cd triton

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build (use -jN for parallel compilation, e.g., -j4)
make -j4
```

**On success**, `triton.exe` is created in the build directory.

### **Ignoring Machine Files (Advanced)**

TRITON uses Machine files to automatically configure builds for specific HPC systems (e.g., Frontier, Perlmutter, Aurora). These files detect the system and set appropriate compiler flags, module loads, and backend options.

If you need **full manual control** over the build configuration (e.g., for custom GPU+OpenMP builds or troubleshooting), you can ignore Machine files:

```bash
cmake -DTRITON_IGNORE_MACHINE_FILES=ON \
      -DBACKEND=HIP \
      -DKokkos_ENABLE_HIP=ON \
      -DKokkos_ARCH_AMD_GFX90A=ON \
      -DCMAKE_CXX_COMPILER=CC \
      -DCMAKE_CXX_FLAGS="-O3 -fopenmp -DTRITON_HIP_LAUNCHER" \
      ..
```

**When to use this option:**
- Building with GPU+OpenMP (not covered by standard Machine files)
- Debugging build configuration issues on HPC systems
- Testing custom compiler flags or backends
- Replicating exact build configurations across different systems

**Note:** When ignoring Machine files, you must manually specify all required settings including:
- `-DBACKEND` (SERIAL, OPENMP, CUDA, HIP, SYCL, etc.)
- `-DCMAKE_CXX_COMPILER` (MPI compiler)
- Compiler and linker flags as needed

### **Using Docker (Optional)**
```bash
make docker_build
make docker_run

docker pull grnydawn/triton-mpich
```

### **SWMM Coupling (Optional)**

TRITON can be coupled with EPA's Stormwater Management Model (SWMM) for integrated surface-subsurface urban drainage simulation.

**SWMM is included with TRITON** - no separate download or compilation needed!

#### Build with SWMM Coupling

```bash
cd triton
mkdir build && cd build
cmake -DTRITON_ENABLE_SWMM=ON ..
make -j4
```

That's it! The bundled SWMM library is compiled automatically as part of the TRITON build process.

#### SWMM Configuration

Add the following parameters to your TRITON configuration (.cfg) file:

```cfg
# SWMM coupling parameters
inp_filename = "path/to/swmm_network.inp"     # SWMM input file
manhole_diameter = 1.2                         # Manhole diameter (m)
manhole_loss = 0.7                             # Manhole loss coefficient
```

The SWMM .inp file should define:
- **COORDINATES**: X, Y locations of SWMM nodes
- **JUNCTIONS**: Junction elevation and max depth
- **INFLOWS**: Nodes that exchange flow with the surface (TRITON)

#### Output

When SWMM coupling is enabled:
- A `swmm/` subdirectory is created in the output folder
- SWMM results are written as:
  - `*.rpt` - ASCII report file
  - `*.out` - Binary output file

#### Notes

- The SWMM model should preferably use the DYNWAVE flow routing solver
- Ensure manhole diameter is smaller than grid resolution to avoid numerical instabilities
- Multiple SWMM nodes can connect to the same TRITON cell (will cause an error)
- SWMM runs only on rank 0; exchange flow is computed on all ranks
- **Important**: When running with multiple MPI ranks, use 1 or 2 ranks for SWMM coupling (4+ ranks may timeout)

#### Flooding Consistency Debug Check (Optional)

For TRITON-SWMM coupled simulations, a diagnostic tool is available to check consistency between node-level and system-level flooding statistics. This helps identify mass balance discrepancies caused by the "tallnode" workaround.

**Build with flooding debug enabled:**
```bash
cd build
cmake -DTRITON_ENABLE_SWMM=ON -DTRITON_SWMM_FLOODING_DEBUG=ON ..
make -j4
```

**When enabled:**
- A summary is printed to stderr at simulation end
- A detailed report is written to `output/swmm/swmm_flooding_debug.txt`
- The report compares node flooding volumes with system flooding loss
- A discrepancy is expected when the tallnode workaround is active (`canPond=1` in dynwave.c)

**Note:** This is a diagnostic-only feature and does not change the physics or mass balance calculations.

## Running a Simulation

Run a sample case from the build directory using:
```bash
cd build

# run a pre-selected example case (Allatoona)
./triton_run.sh

# run Circular Dambreak
./triton_run.sh ./input/circular/circular_dambreak.cfg

# run Paraboloid
./triton_run.sh ./input/paraboloid/paraboloid.cfg

```

Simulation results will be stored in `output/`.

## Documentation

Full documentation is also available at: [TRITON Documentation](https://triton-ornl.readthedocs.io)

## Testing

TRITON includes regression tests:
```bash
./triton_ctest.sh
```

## Contributing

We welcome contributions!  

1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Commit your changes and open a Merge Request (MR).  

## License

TRITON is released under the **3-Clause BSD License**. See the [LICENSE](LICENSE) file for full terms and conditions.

External third-party libraries included or referenced by TRITON retain their own respective licenses, which are provided in the **licenses** subdirectory.

## Acknowledgments

Development of TRITON is supported by the U.S. Air Force Numerical Weather Modeling Program. TRITON used resources of the Oak Ridge Leadership Computing Facility at Oak Ridge National Laboratory, a U.S. Department of Energy user facility. Development is led by Oak Ridge National Laboratory, the University of Zaragoza (Spain), and Tennessee Technological University (Cookeville, TN).


## Contact

Questions, bug reports, or feature requests:
- Open a GitLab issue: <https://code.ornl.gov/hydro/triton/-/issues>
- Submit a support ticket: <https://triton.ornl.gov/contact/>
- Email: [Mario Morales Hernandez](mailto:mmorales@unizar.es), [Sudershan Gangrade](mailto:gangrades@ornl.gov)

# TRITON – Two-dimensional Runoff Inundation Toolkit for Operational Needs

TRITON is an open-source, high-performance software framework for 2D flood simulation. Its core is a computationally efficient, physics-based hydraulic model that operates on a regular, structured grid and solves the full 2D shallow water equations.

## Features

- **Cross-Platform & HPC Ready** – Runs on single or multiple CPUs (OpenMP+MPI) and supports GPU acceleration (CUDA+MPI) for maximum computational efficiency.
- **Flexible Forcing & Inputs** – Uses topographical data (e.g., DEM, LIDAR) on a uniform Cartesian grid and supports streamflow hydrographs, gridded runoff, or both as hydrological forcing.
- **Rich Output Options** – Produces water depth, 2D velocity maps, and unit discharge data, plus time series outputs at user-defined points and intervals.
- **Linux/Unix Native** – Built for Linux/Unix systems with input/output in ASCII or binary formats, along with tools for easy format conversion.
- **SI Units Standard** – Operates using the International System of Units (SI); users must convert to U.S. customary units if needed.

![Simulation Output](doc/_static/TRITON_output_example.jpg)

## Repository Structure

```
triton/
├── doc/           # User guides, API references, and technical documentation
├── src/           # Core simulation source code
├── tools/         # Tools for TRITON
├── external/      # Kokkos Git submodule
├── input/         # Sample simulation input data files
├── test/          # Regression test suite based on CTest
├── cmake/         # CMake configuration modules and machine files
├── Makefile       # Commands to generate documentation and a TRITON Docker image
└── README.md      # Project overview and usage instructions
```

## Installation

TRITON can be built from source or run using a pre-built container.

### **Prerequisites**
- CMake ≥ 3.16
- C++17 or later compiler (GCC, Clang, or Intel)  
- [MPI](https://www.mpi-forum.org/) (for distributed runs)  
- Optional: CUDA, HIP, or SYCL for GPU acceleration  

### **Build Instructions**
```bash
git clone --recursive https://code.ornl.gov/hydro/triton.git
cd triton
mkdir build && cd build
cmake ..
./triton_build.sh
```

### **Using Docker (Optional)**
```bash
make docker_build
make docker_run

docker pull grnydawn/triton-mpich
```

## Running a Simulation

Run a sample case using:
```bash
./trtion_run.sh
```

Simulation results will be stored in `output/`.

## Documentation

- [User Guide:T.B.D.]
- [API ReferenceT.B.D.]

Full documentation is also available at: [Triton Documentation](https://triton.ornl.gov/documentation)

## Testing

Triton includes regression tests:
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

Triton is released under the **3-Clause BSD License**. See [LICENSE](LICENSE) for more details.

## Acknowledgments

Triton was developed by researchers and engineers at **[Your Organization]** with support from:
- [Funding Agency / Grant Info]
- [Partner Institutions]

## Contact

For questions, bug reports, or feature requests:
- Open an issue via GitLab: [Issues Page](https://code.ornl.gov/hydro/triton/-/issues)  
- Email: T.B.D.

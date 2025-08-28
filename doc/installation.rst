.. _getting_started:

Getting-started
========================

This section provides instructions on how to download, build, run, and test the TRITON project.

Downloading TRITON
------------------

TRITON is a C++ project managed with Git. To obtain the source code, clone the `repository <https://code.ornl.gov/hydro/triton.git>`_ using the following command:

.. code-block:: bash

    git clone https://code.ornl.gov/hydro/triton.git
    cd triton
    git submodule update --init --recursive

Building TRITON
---------------

TRITON uses CMake as its build system. The project is designed to be built using a C++17 compliant compiler and requires MPI for C++ components. Optional tools and tests can also be built.

**Prerequisites:**

* **CMake:** Version 3.16 or higher.
* **C++ Compiler:** A C++17 compatible compiler (e.g., GCC, Clang, MSVC).
* **MPI:** MPI libraries with C++ bindings (e.g., Open MPI, MPICH).

**Optional:**

* CUDA, HIP, or SYCL for GPU acceleration
* GDAL (Geospatial Data Abstraction Library) for GeoTIFF images

**Build Steps:**

1.  **Create a build directory:** It's recommended to build TRITON out-of-source.

    .. code-block:: bash

        mkdir build
        cd build

2.  **Configure CMake:**

Run CMake to configure the project. TRITON's build system selects machine-specific configurations using the `MACHINE`, `COMPILER`, and `BACKEND` arguments:

* **MACHINE**: Target system nickname (e.g., `frontier`)
* **COMPILER**: Compiler nickname (e.g., `cray`)
* **BACKEND**: Kokkos backend name (e.g., `HIP`)

If not specified, defaults are chosen automatically based on your host operating system.

**Examples:**

.. code-block:: bash

   # create and change to the TRITON build directory
   mkdir build
   cd build
   
   # the default compiler and backend are selected for the host OS
   cmake ..
   
   # the default compiler and backend are selected for the Frontier system
   cmake .. -DMACHINE=frontier
   
   # the Cray compiler and default backend are selected for the host OS
   cmake .. -DCOMPILER=cray
   
   # the default compiler and HIP backend are selected for the host OS
   cmake .. -DBACKEND=HIP
   
   # the default compiler and HIP backend are selected for the Frontier system
   cmake .. -DMACHINE=frontier -DBACKEND=HIP
   
   # the Cray compiler and HIP backend are selected for the Frontier system
   cmake .. -DMACHINE=frontier -DCOMPILER=cray -DBACKEND=HIP

Machine files are located in `<TRITON top directory>/cmake/machines/<MACHINE>` and use the naming format: ``COMPILER_BACKEND.<sh|bat>``.

Common backends include: CUDA, HIP, SYCL, OPENMP, and SERIAL.

3.  **Build the Project:**
 
To ensure that the TRITON build and excution tasks uses the configurations that are defined in the TRITON machine file, TRITON's build system generates several shell scripts in the build directory.

To build TRITON, run `triton_build.[sh|bat]`:

.. code-block:: bash

    # assuming a Linux system

    cd build
    ./triton_build.sh

Once the build task is successfuly done, `triton.exe` will be generated in the build directory.

Running TRITON
--------------

As explained in the section above, TRITON's build system generates a shell script to run `triton.exe`.

.. code-block:: bash

    # assuming a Linux system

    cd build
    ./triton_run.sh

Note that `triton_run.sh` includes an MPI job launcher such as mpirun with preconfigured command-line arguments. You may need to modify these arguments to add or remove options to suit your environment.

The first argument of `triton.exe` specifies the path to the TRITON simulation configuration file. `triton_run.sh` includes a preselected path, which you may need to modify to match your simulation setup.

Testing TRITON
--------------

If you enabled TRITON ctest during the CMake configuration by adding `-DBUILD_TESTS=ON` command-line argument , you can run the project's tests using CTest.

.. code-block:: bash

    cd build
    ./triton_ctest.sh

See the :ref:`CMake Command-line Arguments <cmake_arguments>` section for more details on additional arguments and environment variable support.


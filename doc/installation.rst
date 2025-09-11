.. _installation:

Download, Build, and Run
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

**Prerequisites:**

* **CMake:** Version 3.16 or higher.
* **C++ Compiler:** A C++17 compatible compiler.
* **MPI:** MPI libraries with C++ bindings.

**Optional:**

* CUDA, HIP, or SYCL for GPU acceleration
* GDAL (Geospatial Data Abstraction Library) for GeoTIFF images

**Build Steps:**

1.  **Create a build directory:** It's recommended to build TRITON out-of-source.

    .. code-block:: bash

        mkdir build
        cd build

2.  **Run CMake:**

TRITON uses CMake as its build system. User can control TRITON build and execution through providing command-line arguments to `cmake`. Several examples are shown below:

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

.. note::
   In addition to the `MACHINE`, `COMPILER`, and `BACKEND` arguments, users can control
   compilation and execution by adding more arguments such as `COMPILER_FLAGS`.
   See the :ref:`CMake Command-line Arguments <cmake_arguments>` for more details.

3.  **Build TRITON:**
 
After completing the CMake configuration phase, run **triton_build.sh** in build directory
to compile TRITON:

.. code-block:: bash

    # assuming a Linux system

    cd build
    ./triton_build.sh

Once the compilation task is successfuly done, `triton.exe` will be generated in the build directory.

Running TRITON
--------------

To run TRITON, in addition to compiling the TRITON executable, several preparatory steps are required: `1) preparing input data`, `2) creating a simulation configuration file`, and `3) executing the TRITON executable` on your system. See the :ref:`TRITON Setup <configuration_reference>` for instructions on creating a simulation configuration file and the :ref:`Running TRITON <triton_run>` for details on running TRITON.

The following example commands demonstrate how to use **triton_run.sh** script and to run several pre-configured simulation cases

.. code-block:: bash

    # assuming a Linux system
    cd build

    # runs an pre-selected example case, Allatoona
    ./triton_run.sh

    # runs Circular Dambreak case 
    ./triton_run.sh ./input/circular/circular_dambreak.cfg

    # runs Paraboloid case
    ./triton_run.sh ./input/paraboloid/paraboloid.cfg

Testing TRITON Installation
-----------------------------

If you enabled TRITON ctest during the CMake configuration by adding `-DBUILD_TESTS=ON` command-line argument , you can run the project's tests using CTest.

.. code-block:: bash

    cd build
    ./triton_ctest.sh

See the :ref:`CMake Command-line Arguments <cmake_arguments>` for more details on additional arguments and environment variable support.


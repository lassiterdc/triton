.. _cmake_arguments:

CMake Command-line Arguments
==================================

Getting machine-specific configurations
-----------------------------------------

TRITON's build system retrieves machine-specific build configurations from a TRITON machine file based on the ``MACHINE``, ``COMPILER``, and ``BACKEND`` CMake command-line arguments.

If no argument is provided for ``COMPILER`` or ``BACKEND``, ``default`` is used instead. If ``MACHINE`` is not specified, it defaults to the host operating system name, such as `Linux`, `Windows`, or `Darwin`.

*TRITON machine files* are located in `<TRITON top directory>/cmake/machines/<MACHINE>` and follow the filename syntax: ``COMPILER_BACKEND.<sh|bat>``.  The ``MACHINE``, ``COMPILER``, and ``BACKEND`` parts correspond to the CMake command-line arguments. Please see :ref:`Machine Configuration File <machine_file>` section for details.

The following examples illustrate how to invoke CMake with different options:

.. code-block:: bash

   # assuming a Linux system

   # create and change to the TRITON build directory
   mkdir build
   cd build
   
   # the default compiler and backend are selected for the host OS
   # search for Linux/default_default.sh
   cmake ..
   
   # the default compiler and backend are selected for the Frontier system
   # search for frontier/default_default.sh
   cmake .. -DMACHINE=frontier
   
   # the Cray compiler and default backend are selected for the host OS
   # search for Linux/cray_default.sh
   cmake .. -DCOMPILER=cray
   
   # the default compiler and HIP backend are selected for the host OS
   # search for Linux/default_HIP.sh
   cmake .. -DBACKEND=HIP
   
   # the default compiler and HIP backend are selected for the Frontier system
   # search for frontier/default_HIP.sh
   cmake .. -DMACHINE=frontier -DBACKEND=HIP
   
   # the Cray compiler and HIP backend are selected for the Frontier system
   # search for frontier/cray_HIP.sh
   cmake .. -DMACHINE=frontier -DCOMPILER=cray -DBACKEND=HIP

The following list shows the backends supported by TRITON through Kokkos:

* CUDA
* HIP
* SYCL
* OPENMP
* OPENMPTARGET
* THREADS
* SERIAL

Controlling TRITON Compilation and Execution
--------------------------------------------

The following CMake arguments control how TRITON is compiled and executed:

``ARCH``
   This argument sets the CPU and GPU architecture, following the names defined in Kokkos. For example, ORNL’s Frontier GPU system uses ``AMD_GFX90A``.

``RUN_COMMAND``
   This argument sets the MPI job launcher, followed by additional arguments such as ``-N`` for specifying the number of compute nodes.

``COMPILER_FLAGS``
   This argument sets C++ compiler options.

``COMPILER_FLAGS_APPEND``
   This argument appends additional C++ compiler options to those specified in ``COMPILER_FLAGS``.

``LINKER_FLAGS``
   This argument sets linker options.

``LINKER_FLAGS_APPEND``
   This argument appends additional linker options to those specified in ``LINKER_FLAGS``.


Building TRITON Tools and CTest Cases
--------------------------------------------

You can enable optional components such as the TRITON tools and test cases by specifying additional CMake arguments during configuration.

`-DBUILD_TOOLS=ON`

When this argument is set, TRITON will build all tools located in the `tools` subdirectory. This is useful if you need utilities for preparing input data, analyzing results, or performing other auxiliary tasks.

Example:

.. code-block:: bash

   cmake .. -DBUILD_TOOLS=ON

`-DBUILD_TESTS=ON`

When this argument is set, TRITON will generate CTest test cases in the build directory. These tests can be executed to verify the installation and ensure that the model runs correctly on your system.

Example:

.. code-block:: bash

   cmake .. -DBUILD_TESTS=ON

After configuring with `-DBUILD_TESTS=ON` and building `triton.exe` using `triton_build.[sh|bat]`, you can run all tests in the build directory with:

.. code-block:: bash

   ./triton_ctest.sh

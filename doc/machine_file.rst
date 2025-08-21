.. _machine_file:

Machine Configuration File
============================

A machine configuration file includes machine-specific settings used during TRITON build and execution.

TRITON’s build system generates the name of the machine configuration file based on the ``MACHINE``, ``COMPILER``, and ``BACKEND`` CMake command-line arguments.

*TRITON machine configuration files* are located in ``<TRITON top directory>/cmake/machines/<MACHINE>`` and follow the filename syntax:
``COMPILER_BACKEND.<sh|bat>``
The ``MACHINE``, ``COMPILER``, and ``BACKEND`` parts correspond to the CMake command-line arguments.

Machine Configuration Variables
----------------------------------

The following variables control how TRITON is compiled and executed:

``TRITON_BACKEND``
   Sets the name of the backend. The supported backends are CUDA, HIP, SYCL, OPENMP, OPENMPTARGET, THREADS, and SERIAL.

``TRITON_ARCH``
   Sets the CPU and GPU architecture, following the names defined in Kokkos. For example, ORNL’s Frontier GPU system uses ``AMD_GFX90A``.

``TRITON_RUN_COMMAND``
   Sets the MPI job launcher, followed by additional arguments such as ``-N`` to specify the number of compute nodes.

``TRITON_COMPILER``
   Sets the executable path of the C++ compiler. Because TRITON uses MPI, it is common to specify an MPI compiler wrapper such as ``mpic++`` or ``CC`` on Cray systems.

``TRITON_COMPILER_FLAGS``
   Sets C++ compiler options.

``TRITON_COMPILER_FLAGS_APPEND``
   Appends additional C++ compiler options to those specified in ``TRITON_COMPILER_FLAGS``.

``TRITON_LINKER_FLAGS``
   Sets linker options.

``TRITON_LINKER_FLAGS_APPEND``
   Appends additional linker options to those specified in ``TRITON_LINKER_FLAGS``.

**Note:** These configuration names match the corresponding CMake command-line arguments, except they are prefixed with ``TRITON_``.

Using Environment Variables and Configuration Hierarchy
---------------------------------------------------------

All CMake arguments explained in the *CMake Command-line Arguments* section have corresponding variables in TRITON machine files, prefixed with ``TRITON_``. For example, the ``COMPILER_FLAGS`` CMake command-line argument can be set in a machine file as ``TRITON_COMPILER_FLAGS``.

For example, assuming you have a Linux machine file:

.. code-block:: bash

   export TRITON_COMPILER_FLAGS="-O3"

> **Note:** You must use ``export`` on Linux to ensure the variables are available to CMake.

You can also set these variables directly in your environment without using a machine file. For example, to specify ``COMPILER_FLAGS`` via an environment variable:

.. code-block:: bash

   # In a shell
   export TRITON_COMPILER_FLAGS="-O3"

TRITON applies configuration settings using the following priority order (from highest to lowest):

1. Values provided directly in CMake command-line arguments
2. Environment variables (e.g., ``TRITON_COMPILER_FLAGS``)
3. Machine configuration files

This hierarchy ensures that explicit user input overrides environment settings and defaults.

Example Machine File
--------------------

.. code-block:: bash
   
   #!/bin/bash
   
   export TRITON_BACKEND=SERIAL
   export TRITON_ARCH="NATIVE"
   export TRITON_COMPILER=mpic++
   export TRITON_RUN_COMMAND="mpirun -n 8"

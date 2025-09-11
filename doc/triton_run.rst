.. _triton_run:

Running TRITON
==================================

Before proceeding, make sure that `triton.exe` has been generated in the build directory.
See :ref:`CMake Command-line Arguments <cmake_arguments>` and
:ref:`Download, Build, and Run <installation>` for instructions on generating
the TRITON executable (`triton.exe`).

Running "triton_run.sh"
----------------------------------

**triton_run.sh** launches TRITON simulation. The syntax to use **triton_run.sh** is shown below:

.. code-block:: bash

   # in build directory
   ./triton_run.sh <path-to-simulation-configuration-file>

**triton_run.sh** sets the environment variables used during TRITON compilation and runs the MPI job launch command specified by the RUN_COMMAND argument. Modify the script as needed to match your system.

See the :ref:`TRITON Setup <configuration_reference>` for instructions on creating a simulation configuration file.

Instructions for Parallel Input
----------------------------------

For very large cases (>5 billion grid cells), preprocessing splits input files for parallel use.

1. Open ``scriptSplitASCII`` and configure parameters:

   - ``TRITON_DIR``: TRITON directory
   - ``NFILES``: number of splits (equals MPI ranks)
   - ``INPUT_DEM``: DEM input file
   - ``IS_MANN``: YES/NO flag for Manning file
   - ``INPUT_MANN``: Manning input (if YES)
   - ``IS_RMAP``: YES/NO flag for runoff map
   - ``INPUT_RMAP``: runoff map input (if YES)
   - ``OUTPUT_FORMAT``: ASC or BIN
   - ``ASCII2BIN_FOLDER``: for BIN, ascii2bin directory
   - ``ASCII2BIN_RMAP_FOLDER``: for BIN+RMAP, ascii2bin_rmap directory

2. Run the script.

3. Update ``cfg`` file:

   .. code-block:: text

      dem_filename="input/dem/bin/par/case03"
      header_filename="input/dem/bin/par/case03.header"
      n_infile="input/mann/bin/par/case03"
      runoff_map="input/runoff/bin/par/case03_runoff"
      input_format=BIN
      input_option=PAR

4. Run TRITON with the same MPI ranks as ``NFILES``.

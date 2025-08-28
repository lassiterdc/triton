_simulation_file:

TRITON Setup
============

The TRITON simulation will be purely controlled by the configuration file. This documentation provides an overview of the project, guidance on getting started, instructions for configuring and running simulations, details about tools and build options, and a comprehensive API reference.

Once you download the TRITON code. The following directory structure is expected.

Project Structure
-----------------

.. code-block:: text

   triton/
   ├── doc/           # User guides, API references, and technical documentation
   ├── src/           # Core simulation source code
   ├── external/      # Kokkos Git submodule
   ├── input/         # Sample simulation input data files
   ├── test/          # Regression test suite based on CTest
   ├── cmake/         # CMake configuration modules and machine files
   ├── Makefile       # Commands to generate documentation and a TRITON Docker image
   └── README.md      # Project overview and usage instructions


Supported File Types
--------------------

+-----------------------+-----------+
| File Type             | Format    |
+=======================+===========+
| DEM                   | ASC/BIN   |
+-----------------------+-----------+
| Manning's n           | ASC/BIN   |
+-----------------------+-----------+
| Streamflow Hydrograph | TXT       |
+-----------------------+-----------+
| Runoff Hydrograph     | TXT       |
+-----------------------+-----------+
| Runoff Map            | ASC/BIN   |
+-----------------------+-----------+
| Observation Locations | TXT       |
+-----------------------+-----------+

.. note::
   TRITON can also write outputs in **GeoTIFF** format. However, GeoTIFF
   files are not accepted as input.


TRITON Configuration **TO BE UPDATED WITH NEW CASE STUDY NAMES**
----------------------------------------------------------------

The TRITON simulation can be configured and controlled by the configuration file, located in the ``$TRITON/input/cfg`` subdirectory. The configuration file is a plain-text file used to control various simulation options such as duration, parameters, and paths to input/output data.

As a new user, we recommend exploring one of the pre-existing configuration files to get acquainted. To configure your own test case, use one of these files as a template and make appropriate changes as necessary.

A quick walk-through of the main contents of the configuration file is provided below.

Topography
~~~~~~~~~~

- **dem_filename**: path where the DEM is located.

  Example:

  .. code-block:: text

     dem_filename="input/dem/asc/dem_file.dem"

TRITON accepts topographical data in **Esri ASCII raster format**. This file contains the first six rows as header information, followed by a space-delimited matrix of ``nrows x ncols``. This matrix defines the size of the computational domain for the hydraulic simulations.

Refer to ``$TRITON/input/dem`` to explore provided DEM examples.

.. note::
   Avoid negative NODATA values (e.g., ``NODATA_value = -9999``) in the topographical data, as they will act as large depressions during simulations.

Surface Roughness (Manning's n Coefficient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **n_infile**: location of the Manning's roughness file.

  Example:

  .. code-block:: text

     n_infile="input/mann/asc/roughness_file.mann"

TRITON accepts spatially varying Manning's n coefficients to account for surface roughness. The roughness must be defined explicitly for each grid cell of the computational domain. This file follows the same template as the DEM file (Esri ASCII raster format), excluding the header.

- **const_mann**: user-defined constant Manning's n value.

  Example:

  .. code-block:: text

     const_mann=0.035

Alternatively, users may provide a constant Manning's n value across the entire domain.

Hydrologic Forcing
~~~~~~~~~~~~~~~~~~

TRITON can be driven by either streamflow hydrographs at specified locations or gridded runoff hydrographs (or both).

**Streamflow Hydrograph**

- **hydrograph_filename**: file location of streamflow hydrograph.

  Example:

  .. code-block:: text

     hydrograph_filename="input/strmflow/streamflow.hyg"

This file contains streamflow hydrograph data in tabular form. The first column is time (in hours), followed by columns of inflow sources (equal to the number of inflow locations) with discharge values (m\ :sup:`3`/s).

- **num_sources**: number of inflow points/locations.

  Example:

  .. code-block:: text

     num_sources=2

- **src_loc_file**: file with Cartesian coordinates of inflow locations.

  Example:

  .. code-block:: text

     src_loc_file="input/strmflow/location.txt"

This file lists coordinates of all inflow locations in the same order as in ``hydrograph_filename``.

**Runoff Hydrograph**

- **runoff_filename**: file location of runoff hydrograph.

  Example:

  .. code-block:: text

     runoff_filename="input/runoff/runoff.hyg"

This file contains runoff hydrograph data in tabular form. The first column is time (in hours), followed by columns of runoff sources (equal to the number of inflow locations), containing the discharge evolution (mm/hour).

- **num_runoffs**: number of distinct runoff areas.

  Example:

  .. code-block:: text

     num_runoffs=156

- **runoff_map**: raster map of distinct runoff areas.

  Example:

  .. code-block:: text

     runoff_map="input/runoff/runoff.rmap"

This file follows the DEM template (Esri ASCII raster format, no header). Each grid cell is assigned a non-negative integer defining runoff areas.

.. note::
   See Morales-Hernández et al., XX or input files of test case #3 for examples.

- **runoff_row_size**: number of rows in the runoff file.
  (To be removed in the final release.)

External Boundary
~~~~~~~~~~~~~~~~~

- **num_extbc**: number of external boundary conditions.

  Example:

  .. code-block:: text

     num_extbc=0

If ``num_extbc=0``, domain boundaries (N/E/S/W) are closed.
If ``num_extbc >= 1``, boundary conditions are defined in ``extbc_file``.

- **extbc_file**: file defining external boundary conditions.

  Example:

  .. code-block:: text

     extbc_file="input/extbc/boundary.extbc"

Columns:

- Column 1: boundary type (0-3)

  - 0: free flow (supercritical)
  - 1: level vs. time (requires table of time + water level)
  - 2: normal slope (requires slope value in last column)
  - 3: Froude number (requires Froude number in last column)

- Columns 2-5: X, Y coordinates of endpoints of boundary.

Simulation Control
~~~~~~~~~~~~~~~~~~

- **sim_start_time**: start time in seconds.
  Example: ``sim_start_time=0``

- **sim_duration**: simulation duration in seconds.
  Example: ``sim_duration=43200``

- **checkpoint_id**: ID to restart from a checkpoint.
  Example: ``checkpoint_id=24``

  The counter is stored in ``$TRITON/cid``. Increments after each print interval.

- **time_increment_fixed**: switch for timestep control.
  Example: ``time_increment_fixed=0`` (variable dt)
           ``time_increment_fixed=1`` (constant dt)

- **time_step**: timestep size (s) if constant dt is selected.
  Example: ``time_step=0.01``

- **Initial condition files** (optional):

  - ``h_infile`` = initial water depth
  - ``qx_infile`` = initial x-direction discharge
  - ``qy_infile`` = initial y-direction discharge

  Example:

  .. code-block:: text

     h_infile="input/inith/asc/case5.inith"
     qx_infile="input/initqx/asc/case5.initqx"
     qy_infile="input/initqy/asc/case5.initqy"

Output Control
~~~~~~~~~~~~~~

- **print_option**: governs output type.
  Example: ``print_option=h`` (depth only)
           ``print_option=huv`` (depth + discharges)

- **print_interval**: output interval in seconds.
  Example: ``print_interval=1800``

- **time_series_flag**: activate/deactivate hydrograph outputs.
  Example: ``time_series_flag=0`` (off)
           ``time_series_flag=1`` (on)

- **observation_loc_file**: file with coordinates of observation points.

Input/Output File Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

TRITON supports ASCII or binary formats. Controls:

.. code-block:: text

   input_format=ASC    # Options: BIN or ASC
   output_format=ASC
   outfile_pattern="%s/%s/%s_%02d_%02d"
   output_option=PAR   # Options: PAR or SEQ

- **SEQ**: sequentially gather subdomains into one file.
- **PAR**: parallel output (faster, distributed).

Miscellaneous Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

- **courant**: Courant-Friedrichs-Lewy (CFL) number.
  Example: ``courant=0.5`` (should not exceed 0.5)

- **hextra**: water depth tolerance (below this, velocities set to zero).
  Example: ``hextra=0.001``

- **gpu_direct_flag**: set to ``0`` unless multiple GPUs with CUDA-aware MPI are available.
  Example: ``gpu_direct_flag=0``

Instructions for Parallel Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

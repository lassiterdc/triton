.. _configuration_reference:

Configuration Reference
=======================

.. note::
   For a quick lookup of all configuration variables in alphabetical order,
   see :ref:`configuration_variable_index`.


A TRITON configuration file (``.cfg``) defines everything needed to launch a run:

- **Input data** – paths to DEMs, roughness maps, and hydrographs  
- **Numerical settings** – timestep control, CFL limit, tolerances  
- **Forcing and boundaries** – streamflow, runoff, and external conditions  
- **Outputs** – format, frequency, and file naming conventions  

Each case directory under ``input/`` contains its own configuration file.  
For example, the Allatoona case provides:

.. code-block:: text

   input/allatoona/allatoona.cfg

This file is a good starting template: adjust paths and parameters to match
your own domain.

.. important::
   Unless an absolute path is given, all file paths in the configuration are
   resolved **relative to your working directory** (where you launch TRITON),
   not the location of the ``.cfg`` file itself.

The sections below walk through the configuration parameters block by block, using
``allatoona.cfg`` as an example.

.. _configuration_overview:

Configuration Overview
----------------------

The configuration file specifies all aspects of a TRITON run, including:

- paths to input rasters and hydrographs  
- numerical parameters (time step, CFL, tolerances)  
- boundary and forcing conditions  
- output format and frequency  

Each case directory under ``input/`` contains its own configuration file. For 
example, the Allatoona case includes:

.. code-block:: text

   input/allatoona/allatoona.cfg

We recommend starting with this file as a template. By adjusting paths and 
parameters, you can quickly adapt it to your own domain.

The following sections walk through the main configuration blocks, using 
``allatoona.cfg`` as a concrete example.

.. note::
   Unless an absolute path is given, all file paths in the configuration are
   resolved relative to the directory where you launch the run (your working
   directory), not necessarily the location of the ``.cfg`` file.

.. _topography:

Topography
~~~~~~~~~~

The first and most important input is the **Digital Elevation Model (DEM)**, which 
defines the spatial extent and resolution of the simulation domain.

.. index:: single: dem_filename
.. index:: pair: configuration; dem_filename

- **dem_filename**: path to the Digital Elevation Model file.

  Example (Allatoona):

  .. code-block:: text

     dem_filename="input/allatoona/allatoona.dem"

DEM files can be provided in **Esri ASCII raster (ASC)** or **TRITON binary (BIN)**
format. Conversion utilities between ASCII and binary are provided in
the ``tools/`` directory (#SG - tosort).

**ASCII DEM format**

An ASCII DEM begins with a six-line header followed by the elevation matrix.  
For example, the Allatoona DEM starts with:

.. code-block:: text

   ncols         591
   nrows         673
   xllcorner     719559.01581497
   yllcorner     3765449.3800973
   cellsize      30
   NODATA_value  -9999
   305.2047 301.1886 297.5595 297.4224 302.5769 ...

- The six header lines define the grid dimensions, spatial reference, resolution, and NODATA value.  
- The matrix that follows contains elevations for ``nrows × ncols`` cells.  

.. important::
   - TRITON requires the DEM (and all other rasters such as Manning’s n and runoff maps) 
     to be in a **projected coordinate system** with units in **meters**.  
   - All rasters must have the **same grid size, extent, resolution, and alignment** 
     as the DEM. Any mismatch will cause TRITON to fail at startup.  

.. note::
   The NODATA field is required by the ESRI ASCII raster format. TRITON does not
   apply any special treatment to NODATA values — they are read literally as part
   of the matrix. If negative values such as ``-9999`` are present, they will be
   interpreted as very deep pits, which may destabilize a simulation.

.. _io_formats:

Input and Output Formats
~~~~~~~~~~~~~~~~~~~~~~~~

File format options build directly on the file types described above.

- **Inputs**: TRITON accepts raster inputs (DEM, Manning’s n, runoff maps) in  
  **ASCII (ASC)** or **binary (BIN)** format.  
- **Outputs**: TRITON can write results in **ASC**, **BIN**, or **GeoTIFF (GTIFF)**.  

  - ASCII is easy to read and debug.  
  - Binary is compact and faster to read/write.  
  - GeoTIFF is ideal for direct use in GIS software and supports georeferencing.  

.. important::
   If GeoTIFF (``GTIFF``) is chosen for outputs, you must also specify a 
   **projection** in the configuration file. Without it, the GeoTIFF will not 
   contain spatial reference information.

   Example:

   .. code-block:: text

      output_format=GTIFF
      projection="EPSG:32614"   # UTM Zone 14, WGS 84

   The ``projection`` parameter is required only for GTIFF outputs. It is ignored
   if the format is ASC or BIN.

.. index:: single: input_format
.. index:: pair: configuration; input_format

- **input_format**: format of input rasters.

  Example (Allatoona):

  .. code-block:: text

     input_format=ASC

.. index:: single: output_format
.. index:: pair: configuration; output_format

- **output_format**: format of raster outputs.

  Example (Allatoona):

  .. code-block:: text

     output_format=BIN

.. index:: single: output_option
.. index:: pair: configuration; output_option

- **output_option**: controls how outputs are written in parallel.

  - ``SEQ`` (sequential): gathers subdomains into a single file per variable.  
  - ``PAR`` (parallel): writes one file per subdomain (faster on clusters).  

  Example (Allatoona):

  .. code-block:: text

     output_option=SEQ

.. tip::
   ``output_option=PAR`` writes one file per subdomain and is faster on clusters.
   Many GIS tools can read tiled outputs directly, but if you prefer single files,
   write with ``SEQ`` or mosaic tiles during post-processing.

.. index:: single: outfile_pattern
.. index:: pair: configuration; outfile_pattern

- **outfile_pattern**: naming convention for output files. TRITON replaces the
  placeholders with case name, variable, and timestep indices. This rarely needs
  modification.

  Example (Allatoona):

  .. code-block:: text

     outfile_pattern="%s/%s/%s_%02d_%02d"

.. index:: single: projection
.. index:: pair: configuration; projection

- **projection** *(required for GTIFF only)*: EPSG code or projection string
  to assign spatial reference to GeoTIFF outputs.

  Example:

  .. code-block:: text

     projection="EPSG:32614"

.. _hydrologic_forcing:

Hydrologic Forcing
~~~~~~~~~~~~~~~~~~

TRITON can be driven by **streamflow hydrographs** at specified inflow locations,
by **runoff hydrographs** over distributed zones, or by a combination of both.  
The Allatoona example demonstrates both types of inputs.

**Streamflow Hydrograph**

.. index:: single: num_sources
.. index:: pair: configuration; num_sources

- **num_sources**: number of inflow points (streamflow sources).

  Example (Allatoona):

  .. code-block:: text

     num_sources=2

.. index:: single: hydrograph_filename
.. index:: pair: configuration; hydrograph_filename

- **hydrograph_filename**: path to the file containing streamflow hydrographs.

  Example (Allatoona):

  .. code-block:: text

     hydrograph_filename="input/allatoona/allatoona.hyg"

  Format:  
  - First column = time in hours  
  - Remaining columns = discharges (m³/s), one column for each source  
  - Number of columns after time must equal ``num_sources``  

  Example file (``.hyg``):

  .. code-block:: text

     0.0   12.3   8.1
     0.5   13.0   8.2
     1.0   16.7   9.0
     # time (h), Q1 (m³/s), Q2 (m³/s)

.. index:: single: src_loc_file
.. index:: pair: configuration; src_loc_file

- **src_loc_file**: file with the Cartesian coordinates of inflow locations.  
  The order of coordinates must match the column order in ``hydrograph_filename``.

  Example (Allatoona):

  .. code-block:: text

     src_loc_file="input/allatoona/allatoona.src"

  Example file (``.src``):

  .. code-block:: text

     720100.0  3765900.0
     721500.0  3767100.0
     # X Y in projected meters

.. note::
   Hydrographs are plain text tables with no headers. Units: time in **hours**, 
   discharge in **m³/s**.


**Runoff Hydrograph**

.. index:: single: num_runoffs
.. index:: pair: configuration; num_runoffs

- **num_runoffs**: number of distinct runoff zones in the domain.

  Example (Allatoona):

  .. code-block:: text

     num_runoffs=156

.. index:: single: runoff_filename
.. index:: pair: configuration; runoff_filename

- **runoff_filename**: file with the runoff hydrographs.

  Example (Allatoona):

  .. code-block:: text

     runoff_filename="input/allatoona/allatoona.roff"

  Format:  
  - First column = time in hours  
  - Remaining columns = runoff rates (mm/hour), one for each runoff zone  
  - Number of columns after time must equal ``num_runoffs``  

  Example file (``.roff``):

  .. code-block:: text

     0.0   0.0  0.0  0.0 ...
     0.5   0.1  0.0  0.0 ...
     1.0   0.3  0.1  0.0 ...
     # time (h), runoff (mm/hr) for each zone

.. index:: single: runoff_map
.. index:: pair: configuration; runoff_map

- **runoff_map**: raster assigning each grid cell to a runoff zone
  (integer IDs from 1…``num_runoffs``). Must align exactly with the DEM.

  Example (Allatoona):

  .. code-block:: text

     runoff_map="input/allatoona/allatoona.rmap"

.. important::
   - The runoff map must use the **same grid, resolution, and projection** as the DEM.  
   - Zone IDs must correspond exactly to the runoff hydrograph file columns.  
   - Units: time in **hours**, runoff in **mm/hour**.

.. note::
   Streamflow and runoff forcing can be used **independently** or **together**.  
   For example:  
   - A dam-break test may use only streamflow hydrographs.  
   - A watershed-scale flood may combine distributed runoff with a few streamflow 
     sources. 

   TRITON superimposes all sources during simulation.

.. _surface_roughness:

Surface Roughness (Manning’s n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TRITON accounts for surface resistance using Manning’s n coefficients. These can
be defined as a spatially varying raster map or as a constant value.

.. index:: single: n_infile
.. index:: pair: configuration; n_infile

- **n_infile**: path to a raster of Manning’s n values.

  Example (Allatoona):

  .. code-block:: text

     n_infile="input/allatoona/allatoona.mann"

  - Same grid, extent, and projection as the DEM  
  - Format: ASCII (``.asc``) or binary (``.bin``)  

.. index:: single: const_mann
.. index:: pair: configuration; const_mann

- **const_mann**: constant Manning’s n value for the entire domain.

  Example:

  .. code-block:: text

     const_mann=0.035

.. important::
   - If ``n_infile`` is provided, it overrides ``const_mann``.  
   - The Manning raster must align exactly with the DEM grid.  

.. note::
   Typical Manning’s n values:  
   - 0.01–0.03 → smooth channels  
   - 0.03–0.05 → natural rivers with vegetation  
   - >0.05 → floodplains with dense vegetation or urban roughness
.. _external_boundaries:

External Boundaries
~~~~~~~~~~~~~~~~~~~

By default, TRITON treats all domain edges as **closed**.  
To allow inflow/outflow, users must define external boundaries.

.. index:: single: num_extbc
.. index:: pair: configuration; num_extbc

- **num_extbc**: number of boundary segments.

  Example (Allatoona):

  .. code-block:: text

     num_extbc=2

.. index:: single: extbc_file
.. index:: pair: configuration; extbc_file

- **extbc_file**: file listing boundary segments.

  Example (Allatoona):

  .. code-block:: text

     extbc_file="input/allatoona/allatoona.extbc"

.. index:: single: extbc_dir
.. index:: pair: configuration; extbc_dir

- **extbc_dir**: optional directory for boundary files.

  Example (Allatoona):

  .. code-block:: text

     extbc_dir="input/allatoona/"

**Boundary File Format**

One segment per row, columns:

1. Type (integer code):  
   - ``0`` = free flow (supercritical outflow)  
   - ``1`` = level vs. time (requires time–level table)  
   - ``2`` = normal slope (requires slope value)  
   - ``3`` = Froude number (requires Froude value)  

2–5. X, Y coordinates of endpoints  

6+. Additional data depending on type  
   - Type 1: filename with time–level data  
   - Type 2: slope value  
   - Type 3: Froude number  

Example for type 1 referenced file:

.. code-block:: text

   0.0   255.0
   1.0   255.2
   2.0   255.5
   # time (h), water level (m)

.. important::
   - ``num_extbc`` must equal the number of rows in ``extbc_file``.  
   - Coordinates must be in projected meters (same CRS as DEM).  

.. note::
   If explicit boundaries are defined (``num_extbc > 0``), they override the 
   global ``open_boundaries`` switch.

.. _simulation_control:

Simulation Control
~~~~~~~~~~~~~~~~~~

This block sets the simulation timing and timestep control.

.. index:: single: sim_start_time
.. index:: pair: configuration; sim_start_time

- **sim_start_time**: start time in seconds.

  Example:

  .. code-block:: text

     sim_start_time=0

.. index:: single: sim_duration
.. index:: pair: configuration; sim_duration

- **sim_duration**: total simulation length in seconds.

  Example (Allatoona):

  .. code-block:: text

     sim_duration=432000   # 5 days

.. index:: single: checkpoint_id
.. index:: pair: configuration; checkpoint_id

- **checkpoint_id**: restart index.  
  - ``0`` = fresh start  
  - ``>0`` = restart from that checkpoint  

.. index:: single: time_increment_fixed
.. index:: pair: configuration; time_increment_fixed

- **time_increment_fixed**: timestep mode.  
  - ``0`` = adaptive timestep (default, CFL-controlled)  
  - ``1`` = fixed timestep (set by ``time_step``)  

.. index:: single: time_step
.. index:: pair: configuration; time_step

- **time_step**: timestep size (s), only used if ``time_increment_fixed=1``.

.. note::
   Adaptive timestepping (``time_increment_fixed=0``) is recommended.  

.. important::
   ``courant`` (see Miscellaneous Parameters) controls stability. Keep CFL ≤ 0.5.

.. _output_control:

Output Control
~~~~~~~~~~~~~~

This block controls outputs and logging.

.. index:: single: print_option
.. index:: pair: configuration; print_option

- **print_option**: raster output fields.  
  - ``h`` = water depth only  
  - ``huv`` = depth + discharges  

  Example (Allatoona):

  .. code-block:: text

     print_option=h

.. index:: single: print_interval
.. index:: pair: configuration; print_interval

- **print_interval**: time interval between raster outputs (s).  

  Example:

  .. code-block:: text

     print_interval=1800   # every 30 minutes

.. index:: single: time_series_flag
.. index:: pair: configuration; time_series_flag

- **time_series_flag**: write hydrograph time series at observation points.  
  - ``0`` = disabled  
  - ``1`` = enabled  

.. index:: single: observation_loc_file
.. index:: pair: configuration; observation_loc_file

- **observation_loc_file**: file with observation coordinates (X Y in meters).  

  Example file:

  .. code-block:: text

     720950.0  3766400.0
     721200.0  3766800.0
     # no headers

.. index:: single: print_observation
.. index:: pair: configuration; print_observation

- **print_observation**: switch for writing observation outputs.  

.. index:: single: it_print
.. index:: pair: configuration; it_print

- **it_print**: iteration interval for diagnostic log messages.  

.. note::
   - Raster outputs: controlled by ``print_option`` and ``print_interval``.  
   - Time series: require both ``time_series_flag=1`` and a valid observation file.  
   - Console/log verbosity: controlled by ``it_print``.

.. _initial_conditions:

Initial Conditions
~~~~~~~~~~~~~~~~~~

TRITON starts dry unless initial condition files are provided.

.. index:: single: h_infile
.. index:: pair: configuration; h_infile
.. index:: single: qx_infile
.. index:: pair: configuration; qx_infile
.. index:: single: qy_infile
.. index:: pair: configuration; qy_infile

- **h_infile**: initial depth raster  
- **qx_infile**: initial discharge (x)  
- **qy_infile**: initial discharge (y)  

Example:

.. code-block:: text

   h_infile="input/allatoona/allatoona.inith"
   qx_infile="input/allatoona/allatoona.initqx"
   qy_infile="input/allatoona/allatoona.inityq"

In Allatoona these are empty, so it starts dry.

.. note::
   These rasters must match the DEM grid. Typically used for warm starts,
   calibration, or restarts.


.. _misc_params:

Miscellaneous Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Advanced settings for stability and parallel performance.

.. index:: single: courant
.. index:: pair: configuration; courant

- **courant**: CFL number (≤ 0.5 recommended).  

  Example:

  .. code-block:: text

     courant=0.5

.. index:: single: hextra
.. index:: pair: configuration; hextra

- **hextra**: depth tolerance (m). Below this, velocities set to 0.

.. index:: single: gpu_direct_flag
.. index:: pair: configuration; gpu_direct_flag

- **gpu_direct_flag**: GPU-aware MPI flag.  
  - ``0`` = off (safe default)  
  - ``1`` = on (use only with CUDA-aware MPI)  

.. index:: single: domain_decomposition
.. index:: pair: configuration; domain_decomposition

- **domain_decomposition**: domain partitioning mode.  
  - ``static`` (default)  
  - ``dynamic`` (updates partitions periodically)  

.. index:: single: factor_interval_domain_decomposition
.. index:: pair: configuration; factor_interval_domain_decomposition

- **factor_interval_domain_decomposition**: update frequency for dynamic mode.  

.. index:: single: open_boundaries
.. index:: pair: configuration; open_boundaries

- **open_boundaries**: global switch for open domain edges (test use only).  

.. index:: single: it_count
.. index:: pair: configuration; it_count

- **it_count**: internal counter, usually left at 0.  

.. important::
   - Keep ``courant`` ≤ 0.5.  
   - Leave ``gpu_direct_flag=0`` unless using CUDA-aware MPI.  
   - ``static`` domain decomposition ensures reproducibility.  

.. tip::
   ``dynamic`` decomposition may improve scaling for large, localized floods,
   but can reduce strict reproducibility.


.. _quick_sanity_checks:

Quick Sanity Checks
-------------------

- DEM, Manning, and runoff map share the same grid and CRS.  
- Input rasters are ASC/BIN. If outputs are GTIFF, set ``projection="EPSG:xxxx"``.  
- Hydrograph columns = ``num_sources``; runoff columns = ``num_runoffs``.  
- Inflow order matches hydrograph column order.  
- Boundary count = ``num_extbc``; coords in projected meters.  
- ``courant <= 0.5``; adaptive dt recommended.  


.. note::
   Unless an absolute path is given, all file paths in the configuration are
   resolved relative to the directory where you launch the run (your working
   directory), not necessarily the location of the ``.cfg`` file.

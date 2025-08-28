.. _before_you_run:

Before You Run TRITON
=====================

Read this first for a quick guide to preparing a TRITON simulation.

What You Need
-------------

- **DEM**: cleaned, projected (e.g., UTM), with a correct NODATA value. *Required*.
- **Hydrologic Forcing**: inflow hydrograph at selected locations, gridded runoff matching the DEM extent, or both. *At least one is required*.
- **Manning’s n**: model can use a constant value or a spatially varying field.
- **Boundary Conditions**: applied only at the edges of the computational domain (see configuration file for details).
- **Configuration File (.cfg)**: defines paths, time settings, and solver parameters.
- **System**: Linux or container environment with CPU or GPU resources.

Prerequisites
-------------

- **CMake ≥ 3.16**
- **C++17 or later compiler** (e.g., GCC, Clang, Intel)
- **MPI**: required for distributed runs
- *Optional*: **CUDA, HIP, or SYCL** for GPU acceleration

Data Standards
--------------

- Use a projected coordinate system.
- SI units (meters, seconds, m³/s).

TRITON Input Overview
---------------------

.. image:: _static/input_overview.gif
   :alt: TRITON input overview
   :width: 100%

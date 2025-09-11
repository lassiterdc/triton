.. _introduction:

Introduction to TRITON
======================

The **Two-dimensional Runoff Inundation Toolkit for Operational Needs (TRITON)**  
is a 2D open-source flood simulation tool designed for modern high-performance computing (HPC).  
It is a computationally efficient, physics-based hydraulic model that operates on a structured grid  
and solves the full 2D shallow-water equations.

Key Features
------------

* **Multi-platform HPC support**
  - Single CPU or multiple CPUs (OpenMP + MPI)  
  - Single GPU or multiple GPUs (CUDA / HIP / Kokkos + MPI)  
  - Highest computational efficiency on GPUs  

* **Topographic inputs**  
  Runs on DEMs or LiDAR-derived topography on a uniform Cartesian grid.  

* **Flexible hydrologic forcing**  
  Accepts streamflow hydrographs, gridded runoff hydrographs, or both.  

* **Primary outputs**  
  Water depth and 2D discharge maps at user-defined time intervals.  

* **Additional outputs**  
  Time series and point-based hydrographs at observation locations.  

Example Output
--------------

.. image:: _static/TRITON_output_example.jpg
   :alt: Example TRITON output
   :width: 100%

What You Need
-------------

To set up and run TRITON, you will need:

- **DEM** – topographic grid of the domain (projected, e.g., UTM).  
- **Hydrologic Forcing** – inflow hydrographs, gridded runoff, or both.  
- **Manning’s n** – constant roughness or a spatially varying map.  
- **Boundary Conditions** – optional, applied at the edges of the domain.  
- **Configuration File (.cfg)** – defines paths, timing, and solver parameters.  
- **System** – Linux or container environment with CPU or GPU resources.  

See :doc:`simulation_setup` for details on supported file types, formats, and directory layout.

Input Overview
--------------

.. image:: _static/input_overview.gif
   :alt: TRITON input overview
   :width: 100%
   
   
   Prerequisites for Compiling TRITON
----------------------------------

- **CMake ≥ 3.16**  
- **C++17 or later compiler**  
- **MPI** – required for distributed runs  
- *Optional*: **CUDA, HIP, or SYCL** for GPU acceleration  



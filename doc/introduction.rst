.. _introduction:

Introduction to TRITON
==================================

The Two-dimensional Runoff Inundation Toolkit for Operational Needs (TRITON) is a 2D open-source flood simulation tool designed for modern high-performance computing (HPC). At its core, TRITON is a computationally efficient, physics-based hydraulic model that operates on a structured grid and solves the full 2D shallow-water equations.  

Key Features
------------

* **Multi-platform HPC support**
  
  - Single CPU or multiple CPUs (OpenMP + MPI)  
  - Single GPU or multiple GPUs (CUDA / HIP / Kokkos + MPI)  
  - Highest computational efficiency is achieved using GPU implementations.  

* **Topographic inputs**  
  Uses digital elevation models (DEM) or LiDAR as the base input on a uniform Cartesian grid.  

* **Flexible hydrologic forcing**  
  Can be driven by streamflow hydrographs, gridded runoff hydrographs, or both.  

* **Primary outputs**  
  Water depth and 2D unit discharge maps at user-defined time intervals.  

* **Additional outputs**  
  Unit discharge values and time series of simulated results at specified point locations.  

An example output is shown below:

.. image:: _static/TRITON_output_example.jpg
   :alt: Example TRITON output
   :width: 100%

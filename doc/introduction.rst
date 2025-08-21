.. _introduction:

Introduction to TRITON
==================================

The Two-dimensional Runoff Inundation Toolkit for Operational Needs (TRITON*) is a 2D open source flood simulation tool designed for modern high performance computing (HPC). The core of the tool is a computationally efficient, physics-based hydraulic model that operates on a regular/structured grid and solves the full 2D shallow water equations. The key features of TRITON are:

* It can operate on multiple computer platforms and utilize modern HPC environments.
* The users can take advantage of:
   * Implementation with a single central processing unit (CPU) or multiple CPUs (using OpenMP+MPI)
   * Implementation with a single graphics processing unit (GPU) or multiple GPUs (using CUDA+MPI)
Highest TRITON computational efficiency can be achieved by using GPU implementation.
* TRITON utilizes topographical data (e.g., digital elevation model [DEM], light detection and ranging [LIDAR]), as its base input, in a uniform (Cartesian) grid structure. The model can be driven by streamflow hydrographs at specified locations or gridded runoff hydrographs, or both which serves as the model’s hydrological forcing. The primary TRITON output includes water depth and 2D unit discharge maps at user defined time intervals. Other variables such as unit discharge values can be outputted. TRITON can also output timeseries of simulated results as user-defined point locations. An example is shown below:

.. image:: _static/TRITON_output_example.jpg

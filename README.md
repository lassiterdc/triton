**Two-dimensional Runoff Inundation Toolkit for Operational Needs (TRITON; Morales-Hernández et al.,in preparation)**

*Refer to User's Guide, located in doc subdirectory, for instructions on requirements, installation, and other details.* 

A 2D open source flood simulation tool designed for modern high performance computing (HPC). The core of the tool is a computationally efficient, physics-based hydraulic model that operates on a regular/structured grid and solves the full 2D shallow water equations. The key features of TRITON are:


*  It can operate on multiple computer platforms and utilize modern HPC environments. The users can take advantage of:
1.  Implementation with a single central processing unit (CPU) or multiple CPUs (using OpenMP+MPI)
2.  Implementation with a single graphics processing unit (GPU) or multiple GPUs (using CUDA+MPI)

Highest TRITON computational efficiency can be achieved by using GPU implementation.


*  TRITON utilizes topographical data (e.g., digital elevation model [DEM], light detection and ranging [LIDAR]), as its base input, in a uniform (Cartesian) grid structure. The model can be driven by streamflow hydrographs at specified locations or gridded runoff hydrographs, or both which serves as the model’s hydrological forcing. The primary TRITON output includes water depth and 2D velocity maps at user defined time intervals. Other variables such as unit discharge values can be outputted. TRITON can also output timeseries of simulated results as user-defined point locations.

*  TRITON is developed on Linux/Unix platform. The input/output files can be either in ascii or binary formats. A set of tools and instruction are provided for format conversion.

*  The model utilizes International System of Units (SI). Users who are more familiar with United States (US) customary units need to perform proper unit conversion themselves.


*  A set of Test Cases are provided :

1.  Case 01 : TaumSauk
2.  Case 02a: Paraboloid Case - Resolution 0.04m
3.  Case 02b: Paraboloid Case - Resolution 0.02m
4.  Case 02c: Paraboloid Case - Resolution 0.01m
5.  Case 02d: Paraboloid Case - Resolution 0.005m
6.  Case 03 : Runoff
7.  Case 04	: Harvey 30m 
8.  Case 05	: Harvey 10m


Morales-Hernández, M., M. B. Sharif, A. Kalyanapu, S. K. Ghafoor, T. T. Dullo, S. Gangrade, S.-C. Kao, and K. J. Evans (2020), TRITON: A multi-GPU open source 2D high resolution flood model, in preparation.
#!/bin/bash

module purge
module load PrgEnv-nvidia/8.6.0
module load cray-libsci/25.09.0
module load nvidia/25.3
module load cudatoolkit/25.3_12.8

export CRAYPE_LINK_TYPE=dynamic
export CRAY_CPU_TARGET=aarch64

GDAL_DIR=/ccs/proj/nwp501/software/gdal/3.11.3
PROJ_DIR=/ccs/proj/nwp501/software/PROJ/9.6.1
TIFF_DIR=/ccs/proj/nwp501/software/libtiff/4.7.0
SQLITE3_DIR=/ccs/proj/nwp501/software/sqlite3/3.50.4

export CMAKE_PREFIX_PATH="$GDAL_DIR;$PROJ_DIR;$TIFF_DIR:$SQLITE3_DIR:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$GDAL_DIR/lib64:$PROJ_DIR/lib64:$TIFF_DIR/lib64:$SQLITE3_DIR/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$GDAL_DIR/lib64/pkgconfig:$PROJ_DIR/lib64/pkgconfig:$TIFF_DIR/lib64/pkgconfig:$SQLITE3_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
export PATH="$GDAL_DIR/bin:$PROJ_DIR/bin:$TIFF_DIR/bin:$SQLITE3_DIR/bin:$PATH"

export TRITON_BACKEND=CUDA
export TRITON_ARCH=HOPPER90
export TRITON_COMPILER=CC
export TRITON_COMPILER_FLAGS=
export TRITON_LINKER_FLAGS=
export TRITON_DEBUG=OFF
export TRITON_RUN_COMMAND="srun -n 4"

#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load \
    PrgEnv-gnu \
    craype-network-ucx \
    cray-ucx \
    cray-mpich-ucx \
    cudatoolkit \
    craype-accel-nvidia80
module swap gcc-native/12.3
module -t list

GDAL_DIR=/ccs/proj/nwp501/software/gdal/3.11.3
PROJ_DIR=/ccs/proj/nwp501/software/PROJ/9.6.1
TIFF_DIR=/ccs/proj/nwp501/software/libtiff/4.7.0
SQLITE3_DIR=/ccs/proj/nwp501/software/sqlite3/3.50.4

export CMAKE_PREFIX_PATH="$GDAL_DIR;$PROJ_DIR;$TIFF_DIR:$SQLITE3_DIR:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$GDAL_DIR/lib64:$PROJ_DIR/lib64:$TIFF_DIR/lib64:$SQLITE3_DIR/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$GDAL_DIR/lib64/pkgconfig:$PROJ_DIR/lib64/pkgconfig:$TIFF_DIR/lib64/pkgconfig:$SQLITE3_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
export PATH="$GDAL_DIR/bin:$PROJ_DIR/bin:$TIFF_DIR/bin:$SQLITE3_DIR/bin:$PATH"

export CXX=CC

export TRITON_BACKEND="CUDA"
export TRITON_ARCH="AMPERE80"
export TRITON_COMPILER="CC"
export TRITON_COMPILER_FLAGS="-DACTIVE_GPU=1;-DTRITON_CUDA_LAUNCHER;-O3;--use_fast_math;-ccbin;${TRITON_COMPILER}"
export TRITON_LINK_FLAGS=""
export TRITON_DEBUG=OFF
export TRITON_RUN_COMMAND="srun -N 1 -n 4 --gpus-per-task=1"

export CRAYPE_LINK_TYPE=dynamic
export CUDA_DIR=${CUDA_HOME}
export CRAY_CPU_TARGET=${CPU}

unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS

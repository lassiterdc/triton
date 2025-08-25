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

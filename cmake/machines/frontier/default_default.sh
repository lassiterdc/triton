#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-amd

export TRITON_BACKEND="OPENMP"
export TRITON_ARCH="NATIVE"
export TRITON_COMPILER=CC
export TRITON_COMPILER_FLAGS="-O3 -fopenmp"
export TRITON_LINKER_FLAGS="-fopenmp"
export TRITON_DEBUG=OFF
export TRITON_RUN_COMMAND="srun -N 1 -n 8"

export CRAYPE_LINK_TYPE=dynamic


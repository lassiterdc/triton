#!/bin/bash

module purge
module load PrgEnv-cray

export TRITON_BACKEND=OPENMP
export TRITON_ARCH="x86-64"
export TRITON_COMPILER=CC
export TRITON_COMPILER_FLAGS="-O3 -fopenmp"
export TRITON_LINKER_FLAGS="-fopenmp"
export TRITON_DEBUG=OFF
export TRITON_RUN_COMMAND="srun -N 1 -n 8"

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=true
export CRAY_CPU_TARGET=x86-64

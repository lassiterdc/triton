#!/bin/bash

source ${MODULESHOME}/init/bash
module reset
module load PrgEnv-amd cmake craype-accel-amd-gfx90a

export ROCM_PATH=${CRAY_AMD_COMPILER_PREFIX}
export TRITON_BACKEND="HIP"
export TRITON_ARCH="AMD_GFX90A"

export TRITON_COMPILER=CC
export TRITON_COMPILER_FLAGS="-DTRITON_HIP_LAUNCHER -O3 -ffast-math -I${ROCM_PATH}/include -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -Wno-unused-result -Wno-macro-redefined"
export TRITON_LINKER_FLAGS="--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64"
export TRITON_DEBUG=OFF
export TRITON_RUN_COMMAND="srun -N 1 -n 8"

export CRAYPE_LINK_TYPE=dynamic
export MPICH_GPU_SUPPORT_ENABLED=1

unset HSA_XNACK

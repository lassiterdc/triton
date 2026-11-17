# Files

- [MPI decomposition and device execution](parallel-execution.md) - How TRITON distributes raster rows, stages device state, exchanges halos, and rebuilds partitions, including coupled-run collective constraints.
- [Simulation ownership and lifecycle](simulation-lifecycle.md) - End-to-end control flow from MPI and Kokkos startup through model initialization, timestep execution, checkpoint publication, and SWMM shutdown.
- [SWMM snapshot format and output continuity](swmm-state-snapshots.md) - Full-precision SWMM state serialization, compatibility checks, publication markers, and the binary-output positioning required to continue a restored run.

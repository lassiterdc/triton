# Files

- [Checkpoint artifacts and coupled resume](checkpoint-and-resume.md) - Select and preserve a surface or coupled checkpoint, understand snapshot-versus-replay routing, and interpret interrupted-write, retention, and clean-start policies.
- [Preparing configuration, rasters, and forcing](configuration-and-inputs.md) - Prepare a cfg and compatible grid/forcing inputs using the actual parser, path, unit-conversion, coordinate-mapping, and boundary contracts rather than assuming comprehensive startup validation.
- [Ensemble execution with Ensify](ensemble-runs.md) - Build and launch YAML-defined TRITON ensembles, understand per-member MPI communicators and arguments, and separate simulation outputs from redirected streams.
- [Output artifacts and runtime diagnostics](outputs-and-diagnostics.md) - Trace TRITON raster, observation, checkpoint-metadata and log writers, including parallel layouts and the meaning of runtime performance columns.

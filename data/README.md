# Data layout

This directory separates reproducibility metadata from large local data.

- `manifests/` is tracked and records source versions, hashes, and processing settings.
- `raw/` is ignored and holds immutable downloaded source data.
- `processed/` is ignored and holds reproducible normalized data.

Do not manually edit files in `raw/`. A future acquisition command will download
and verify them from a versioned manifest. A separate normalization command will
produce `processed/` outputs without modifying the inputs.

# Data layout

This directory separates reproducibility metadata from large local data.

- `manifests/` is tracked and records source versions, hashes, and processing settings.
- `raw/` is ignored and holds immutable downloaded source data.
- `processed/` is ignored and holds reproducible normalized data.

Do not manually edit files in `raw/`. The acquisition command downloads and verifies
them from a versioned manifest. A separate normalization command produces
`processed/` outputs without modifying the inputs.

Prepare MaleCNS v1.0 with:

```bash
uv run flybrain-data download
uv run flybrain-data build
uv run flybrain-data validate
```

`download` is resumable and verifies byte size plus SHA-256. `build` refuses to
overwrite an existing output directory. `validate` independently checks the locked
output hashes, Parquet row counts, CSR bounds, indices, weights, contact total, and
self-edge count.

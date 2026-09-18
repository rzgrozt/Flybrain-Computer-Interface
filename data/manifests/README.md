# Source manifests

MaleCNS source URLs and checksums will be added only after they have been verified
against official sources. A manifest must identify the dataset version, each source
artifact, its cryptographic checksum, and the normalization schema version.

- `malecns-v1.0.json` locks the generation-specific official source objects.
- `malecns-v1.0-normalized-v2.json` locks the deterministic normalized outputs.
- `optic-column-type-assignments-v1.0.json` locks the pinned official L1/R7/R8
  optic-column workbook used for retinotopic inference.
- `eyemap-zhao2025-v1.json` locks the Zhao et al. 2025 microCT eye-map inputs and
  the deterministic derived viewing-direction CSV hash.

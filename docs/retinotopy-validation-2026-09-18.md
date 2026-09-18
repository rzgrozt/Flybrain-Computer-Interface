# R1-R6 retinotopic column inference

Date: 2026-09-18. Dataset: `male-cns:v1.0`.

This increment establishes a connectivity-grounded mapping from reconstructed MaleCNS
R1-R6 photoreceptors to the official medulla column labels. It deliberately stops
short of mapping those columns to screen azimuth/elevation.

## Source and lock

The official supplemental resource is:

- upstream repository: https://github.com/flyconnectome/2025malecns
- pinned workbook:
  https://raw.githubusercontent.com/flyconnectome/2025malecns/67767d2233657983993ff6c2be48e836a935863c/supplemental_data/optic-column-type-assignments-v1.0.xlsx
- size: 111,565 bytes
- SHA-256:
  `d4af1cacb751036f7e84bfecc9bec79ca010066ac066559c29b566003ec080d3`

The lock lives at
`data/manifests/optic-column-type-assignments-v1.0.json`.
The workbook itself remains under ignored `data/raw/`.

Fetch and verify it with:

```bash
.venv/bin/python scripts/fetch_optic_columns.py
```

The parser uses Python's standard ZIP/XML libraries rather than adding an Excel
dependency.

## What the official workbook contains

The workbook has separate `Right OL` and `Left OL` worksheets plus a `READ ME`
worksheet. Across both optic lobes it contains:

- 1,772 official medulla columns;
- 892 right-side columns;
- 880 left-side columns;
- 1,764 L1 assignments;
- 1,299 R7 assignments;
- 1,329 R8 assignments.

The workbook's own README defines the column field as the column ROI name and lists
the L1/R7/R8 body IDs and column type. It does not define a conversion from the
integer column labels to visual-field azimuth/elevation.

Every official L1/R7/R8 body ID used here is present in the normalized MaleCNS
graph. Annotation consistency checks pass with zero missing IDs, zero type
mismatches and zero laterality mismatches. L1 laterality is encoded in its
`instance` field (`L1_R`/`L1_L`); R7 and R8 use `root_side`.

## Biological basis of the inference

Drosophila uses neural superposition. Six R1-R6 photoreceptors from neighboring
ommatidia that view the same point in visual space converge onto one lamina
cartridge, where R1-R6 provide direct input to L1-L3. L1 then projects into the
corresponding medulla column.

Useful references:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10213501/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6728174/

Therefore the first inference rule is intentionally simple and anatomical:

`R1-R6 source -> direct official L1 targets -> strongest contact-supported column`

Soma positions are not used.

## Confidence policy

The deterministic configuration is
`configs/retinotopy-v1.json`.

Each R1-R6 cell is classified as:

- `unique_high`: exactly one official L1 candidate with at least 5 contacts;
- `unique_low`: exactly one candidate with fewer than 5 contacts;
- `dominant`: multiple candidates, but the best has at least 0.80 of candidate
  contacts and a best-minus-second margin of at least 0.60 of candidate contacts;
- `ambiguous`: multiple candidates without a dominant winner;
- `unmapped`: no direct official-L1 candidate.

For ambiguous cells, the best candidate is recorded for audit but
`assigned_column` is left null. No arbitrary tie-breaking is exposed as a valid
mapping.

The 5-contact low/high boundary and 0.80/0.60 dominance thresholds are engineering
confidence labels, not biological constants. They are fixed in the versioned config.

## Canonical MaleCNS result

Run:

```bash
.venv/bin/python -m flybrain_interface.experiments.retinotopy \
  --data-directory data/processed/malecns-v1.0 \
  --workbook data/raw/optic-column-type-assignments-v1.0.xlsx \
  --output /tmp/retinotopy-v1.json
```

Measured result:

| Status | R1-R6 neurons |
| --- | ---: |
| unique_high | 3,203 |
| unique_low | 14 |
| dominant | 10 |
| ambiguous | 12 |
| unmapped | 138 |

Totals:

- reconstructed R1-R6: 3,377;
- assigned column: 3,227 (95.56%);
- high-confidence assignment (`unique_high + dominant`): 3,213;
- left assigned R1-R6: 1,084;
- right assigned R1-R6: 2,143;
- candidate-side mismatches: 0;
- covered official columns: 796.

Unmapped reasons:

- 125 R1-R6 have no direct L1 edge in the normalized graph;
- 13 have a direct L1 edge, but not to an L1 represented in the official column
  table.

Candidate counts are exceptionally clean:

- 3,217 cells have exactly one official L1 candidate;
- 20 have two candidates;
- one has three;
- one has seven;
- 138 have none.

The 12 ambiguous cases are retained explicitly rather than forced into a column.

## Six-photoreceptor cartridge check

A complete neural-superposition cartridge is expected to pool six R1-R6 inputs.
Among assigned reconstructed inputs:

- 796 columns have at least one assigned R1-R6;
- 240 columns have exactly six assigned R1-R6;
- 33 columns are above six assigned inputs.

The occupancy distribution also contains columns below six, which is expected to be
affected by incomplete/missing R1-R6 reconstruction and the deliberate exclusion of
ambiguous/unmapped assignments. Occupancy is therefore a useful structural check,
not a requirement that every MaleCNS column must equal six in this dataset.

## Follow-up: measured visual direction is now available

The official workbook itself does not define screen azimuth/elevation, so this
structural inference deliberately did not guess an orientation. A separate follow-up
validation now joins the MaleCNS p/q lattice to the Zhao et al. 2025 microCT eye-map
dataset using pinned primary data and explicit coordinate checks.

That follow-up establishes measured viewing directions for 1,678 official columns
(94.70% coverage) and gives measured directions to 3,184 of the 3,227 assigned
R1-R6 photoreceptors. The derived table is reproducible byte-for-byte from the pinned
source inputs.

A perspective virtual-screen sampler has also been validated with deterministic
horizontal and vertical moving bars. See
[`docs/spatial-vision-validation-2026-09-18.md`](spatial-vision-validation-2026-09-18.md)
for provenance, coordinate conventions, coverage, geometry gates and current
limitations.

Soma coordinates remain excluded from receptive-field inference. Missing eye-map
directions remain missing rather than being interpolated or fabricated.

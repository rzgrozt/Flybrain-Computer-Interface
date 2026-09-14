# MaleCNS v1.0 data pipeline

## Authoritative source

The source artifacts come from the [official MaleCNS download
page](https://male-cns.janelia.org/download/) and generation-pinned objects in the
`flyem-male-cns` Google Cloud Storage bucket. The dataset is CC-BY-4.0 and is
credited to the FlyEM Project Team at HHMI Janelia, the University of Cambridge
Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research.

The tracked source manifest records the release UUID, object generation, byte size,
GCS MD5 and CRC32C metadata, and independently computed SHA-256 for:

- curated body annotations;
- aggregate body-level neurotransmitter predictions;
- the full segment-to-segment connection-weight graph.

Raw files total approximately 1.1 GB and remain under ignored `data/raw/`.

## Selection policy

The release annotation table contains 211,577 unique segment IDs. The published
166,700-neuron census corresponds exactly to rows with a non-null `superclass`.
Normalization therefore retains every assigned superclass, rather than using the
narrower `status == "Traced"` filter, which would retain only 165,122 rows.

This is an explicit engineering selection over released annotations. It is not a
claim that unassigned segments are biologically irrelevant. The report preserves
the number of excluded annotation and connection rows.

## Normalized outputs

`neurons.parquet` is ordered by ascending released body ID and assigns a stable,
zero-based `neuron_index`. It retains classification, side, nerve, receptor, and
neurotransmitter fields. Ground-truth, predicted, and consensus transmitter fields
remain separate; normalization does not assign excitatory or inhibitory signs.

`edges.parquet` contains `source_index`, `target_index`, and positive
`synapse_count`, ordered by target then source. No minimum-weight threshold is
applied, and self-edges are retained.

The same graph is emitted as memory-mappable target-major CSR arrays:

- `csr_indptr.npy`: `int64`, one entry per target neuron plus one;
- `csr_indices.npy`: `int32` source-neuron indices;
- `csr_synapse_counts.npy`: `int32` unsigned-by-convention contact counts.

The current locked build contains:

| Quantity | Value |
|---|---:|
| Neurons | 166,700 |
| Directed edges | 25,582,938 |
| Synaptic contacts | 124,177,617 |
| Self-edges | 101 |
| Raw connection rows excluded by endpoint selection | 126,273,746 |

Normalized files total approximately 280 MB and remain under ignored
`data/processed/`.

## Reproducibility and safety

The downloader resumes partial transfers, verifies every completed source, and uses
generation-pinned URLs. The builder refuses to overwrite either completed or partial
outputs. DuckDB performs the 151,856,684-row endpoint join and external sort with a
configurable memory cap. PyArrow streams the sorted Parquet edges into NumPy `.npy`
arrays without loading the full graph into memory.

The independent validation command verifies locked file hashes and sizes, Parquet
row counts, CSR shape and monotonicity, source-index bounds, positive weights,
synaptic-contact total, and self-edge count.

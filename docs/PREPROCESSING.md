# Cell Ranger to BarcodeCNV

`barcodecnv preprocess` invokes the Python preprocessing module. It uses cellSNP-lite
for per-cell UMI allele counts and Beagle for population-reference phasing,
following `GermlineCalling` in the Julia BarcodeCNV implementation. Julia, R and
Eagle are not dependencies. Use `barcodecnv run` for preprocessing and inference together, or `preprocess`
followed by `infer` to reuse preprocessing when changing inference options.

```bash
uv run barcodecnv setup --genome hg38
uv run barcodecnv preprocess \
  --outs /path/to/cellranger/outs \
  --cells /path/to/cell_lineage_map.tsv \
  --one-block --out /path/to/prepared_sample
uv run barcodecnv infer /path/to/prepared_sample/prepared.h5 --out results
```

The map needs header columns `cell,barcode` (or
`cell_barcode,lineage_barcode`), optionally `block`. CSV and TSV are accepted.
Use `--one-block` only for one exchangeable cohort; otherwise supply blocks for
the permutation/bootstrap design. These are lineage labels, not inferred clones.
All selected cells must occur in the filtered 10x matrix. Empty labels and
repeated cell IDs are errors. Barcodes are kept as strings, including leading zeros.

The Cell Ranger `outs` directory must contain an indexed
`possorted_genome_bam.bam` and `filtered_feature_bc_matrix.h5` or the corresponding
MTX directory. Standard CB/UB tags are used. Each preprocessing invocation must
contain **one donor/genotype**; demultiplex multi-donor libraries first. This
initial provisioning targets human hg38; BAM, annotation, SNP catalogue, genetic
maps and phasing panel must use that build. Chromosome-name normalization does
not convert builds. The default allele catalogue is restricted to 1–22 and X;
X uses the available non-PAR panel. Expression genes retain their original scope.

By default, fit the B-cell panel downloaded by `barcodecnv setup`. To override
this, supply either a normal expression table with `--reference` (`gene,fraction` or
`gene,expression`) or a Julia-format panel with `--reference-panel`. The latter
fits a mixture to the selected cells and saves it under `reference_fit/` before
allele counting. See [REFERENCE_FITTING.md](REFERENCE_FITTING.md) for details.
The population haplotype panel remains separate from the expression reference.

For a single end-to-end command, replace `preprocess` with `run` in the example.
It writes intermediate files to `OUT/preprocessing/`, final tables/plots to
`OUT/inference/`, and overall status to `OUT/pipeline.json`. Both `run` and
`preprocess` report stages on the console. `infer` accepts the inference options
that previously belonged to `run`.

## Tools and resources

Use `barcodecnv setup --genome hg38` to download the B-cell expression panel,
annotation, SNP catalogue, genetic maps and phasing reference, and install private
cellSNP-lite, bcftools and Java tools. Beagle is downloaded as a pinned jar.
See [RESOURCES.md](RESOURCES.md) for platform support, storage requirements,
configuration discovery, cache validation and retry behaviour.

The setup config is discovered automatically. Explicit `--config`, the
`BARCODECNV_PREPROCESS_CONFIG` environment variable and existing `.preprocessing/`
configs take precedence over the shared download cache. Individual resource
options still override config values. `--chromosomes` defaults to the configured
set (1–22,X for full setup). The legacy `setup_preprocessing.py` can still link
an existing Julia-style cache and native tools, without downloading them.

## Processing and outputs

1. Validate cells, matrix, reference and chromosome resources before expensive work.
2. Restrict the common-SNP catalogue to requested chromosomes; run cellSNP-lite
   with `--minMAF 0 --minCOUNT 2` and UMI tags CB/UB. Its AD/DP matrices are retained.
3. Make one donor pseudobulk VCF: discard sites with OTH>0; call heterozygous when
   pooled AF is 0.1–0.9, or homozygous alternate when AF=1 and DP≥10. These are the
   Julia pipeline's hard filters, not a new probabilistic genotype caller. Only
   biallelic SNVs are included. Allelic imbalance/low depth can therefore exclude
   true heterozygotes; reference matching and germline genotype ascertainment remain
   limitations of this preprocessing.
4. Run Beagle sequentially per nonempty chromosome, with `impute=false`, bounded
   threads/heap (`--threads 8 --memory-gb 12`) and a recorded seed. Skip only
   chromosomes without pooled genotypes; tool failures abort the run. Index and
   concatenate chromosome outputs.
5. Match phased hets to count rows by chromosome, position, REF and ALT. For
   `GT=0|1`, H1=DP−AD and H2=AD; for `GT=1|0` the assignment reverses. Keep each
   cell separate for downstream resampling. SNPs absent from the panel or with
   mismatched alleles are not used. No matching phased counts is an error.
6. Use the production loader to write `prepared.h5` with cell-level expression,
   allele counts, lineage labels, genetic distances and normal expression reference.

Without GP, heterozygosity is conditional on the hard genotype call (`het=1`).
Without PQ, `phase_prob=0.99` is the existing Python loader assumption, **not** a
measured phasing confidence. GQ is not treated as phase quality. These assumptions
are recorded in `preprocessing.json`; phase is subsequently inferred in the CN HMM.

Outputs include `cells.tsv`, `alleles.tsv.gz`, intermediate VCFs/count matrices,
per-command logs, `prepared.h5`, `infer.sh` and `preprocessing.json`. The manifest
records commands, parameters, source paths, sizes/timestamps, selected checksums,
counts and complete/failed status. Existing output directories are never overwritten;
a failure retains logs and does not publish a completed bundle. No automatic
checkpoint reuse occurs. For deliberate reuse, `--cellsnp-dir` accepts counts for
exactly the same selected cells/BAM, and `--phased-vcf` accepts an existing
single-donor phased VCF. The caller is responsible for their provenance; cell
sets, matrix axes and allele identities are checked. `--chromosomes 22` is useful
for a small smoke test; it restricts the allele processing, not expression genes.

Automated preprocessing regression checks are in
[`tests/test_preprocessing.py`](../tests/test_preprocessing.py).

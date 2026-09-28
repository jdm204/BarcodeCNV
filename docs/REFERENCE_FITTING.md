# Fitting a normal expression reference

Provide either a finished `--reference reference.tsv` or a
`--reference-panel /path/to/panel`. The latter fits a mixture to expression pooled
across the **selected cells** in the cell map, before CN inference. Counts from
other cells in the matrix do not participate. No Julia runtime is used.

```bash
# Cell Ranger data -> fitted reference, phased counts, inference and reports
uv run barcodecnv run --outs /path/to/cellranger/outs \
  --cells cells.tsv --reference-panel /path/to/panel --out results

# Fit a reusable reference separately, using existing expression counts
uv run barcodecnv fit-reference --matrix filtered_feature_bc_matrix/ \
  --cells cells.tsv --genes genes.gtf.gz --reference-panel /path/to/panel \
  --genome hg38 --out fitted_reference

# Reuse fitted_reference/reference.tsv with --reference in later commands.
```

`preprocess`, count-input `infer`, and `prepare` also accept `--reference-panel`.
A prepared bundle already contains its reference: supplying reference overrides
alongside it is rejected. To change a reference, reload the count inputs or
preprocess again with the desired profile. `--reference` and `--reference-panel`
are mutually exclusive. `signal` still needs neither.

Panels use the existing Julia schema-1 bundle: `panel.toml`, `profiles.tsv`,
`expression.h5`, and `dataset.toml` with file checksums. Existing Julia panels
can be passed directly. The reader checks integrity, profile order, identifiers,
finite nonnegative expression fractions, and profile normalization. HDF5 matrix
dimensions are converted explicitly from Julia's on-disk convention. Raw symbol
CSV panel ingestion and automatic panel downloads are not provided by this port.

Ensembl versions are normalized while preserving PAR copies (`ENSG….2_PAR_Y` →
`ENSG…_PAR_Y`); other identifiers, including dotted symbols, are preserved. Joins
use gene IDs, never symbols. The assay `--genome` must match the panel genome
(default hg38). Every assay/panel-overlap gene must have a supplied annotation.
Absent panel genes remain unmeasured rather than becoming zero-expression genes.

## Fitting modes

The default `--reference-method global` adapts the Julia implementation of Numbat's
`fit_ref_sse`. Genes pass when their mean panel expression is strictly above
`--reference-min-cpm` (default 2) and their pooled observed expression is positive.
Observed counts and the predicted mixture are both normalized over these fitting
genes before calculating mean squared log-ratio error. This corrects the original
port's comparison of selected-gene observations with full-universe predictions.
For selected panel rows `M` and softmax weights `w`, the prediction is
`(M @ w) / sum(M @ w)`. The gradient includes this normalization. Individual panel
columns are not renormalized: mixture weights retain their meaning on the full
panel gene universe, and the saved profile is the full panel matrix times `w`.
All logits are free; Adam uses the same learning rate 0.05, betas (0.9, 0.999),
epsilon 1e-8, zero initialization and bias correction as Julia Optimisers.jl.
`--reference-iterations 2000` is a fixed update budget, not a convergence guarantee.
The final objective and logit-gradient magnitude are reported without declaring
convergence. The resulting profile covers the full panel gene universe.

The opt-in `--reference-method regional_consensus` fits chromosome-local windows
(default `--reference-bin-genes 200`) with a nuisance log-scale per window, then
combines mixture weights by a smoothed geometric median. Two layouts, offset by
half a window, are fitted separately; their consensus weights are averaged.
Regions with insufficient positive counts or no distinguishing profile shapes
are skipped. Both layouts must be informative, and all attempted fits must meet
the gradient tolerance. This ports the Julia procedure using SciPy BFGS; different
line-search steps mean the result is numerically close rather than bit-identical.
Zero-count genes remain in the window layout but not the local log objective.

Reference fitting uses the observed sample, so CN alterations can bias a global
mixture toward a profile that resembles those alterations. The regional mode
reduces sensitivity to regional scale changes but is not an ancestry model or
a guarantee against reference mismatch. Fitted mixtures and reference profiles
are treated as fixed inputs by subsequent CN inference; their uncertainty is
not integrated into CN probabilities.

## Saved results

`fit-reference` writes directly to its output directory. `preprocess` and
count-input `infer` write a `reference_fit/` subdirectory; end-to-end `run`
therefore stores the fit in `OUT/preprocessing/reference_fit/`. `prepare --out
sample.h5` uses a sibling `sample.h5.reference_fit/` directory.

- `reference.tsv`: full-universe gene fractions, reusable with `--reference`.
- `weights.tsv`: mixture weights alongside the original profile metadata.
- `fitting_gene_ids.txt`, `overlap_gene_ids.txt`, `absent_gene_ids.txt`: exact gene sets.
- `fit.json`: method/settings, panel provenance/checksums, fitting diagnostics,
  input paths, cell/annotation checksums, selected-cell count and pooled-count hash.

Reference-fitting regression checks cover filtered-gene mixture recovery, a
finite-difference gradient check, and saved Julia numerical expectations where
the full gene universe is retained. Legacy subset fits intentionally differ.
The checks are in [`tests/test_reference.py`](../tests/test_reference.py).

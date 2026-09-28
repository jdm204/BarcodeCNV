<img src="docs/assets/logo.png" alt="Two lineage barcodes beside copy-number heatmap blocks" width="420">

BarcodeCNV is a CNV-calling tool designed for static-barcode lineage tracing single cell RNA sequencing.
Here, static-barcode lineage tracing means transducing cells with inherited, expressed static nucleotide sequence barcodes (systems such as LARRY, CellTag, WILDseq, SPLINTR).

BarcodeCNV uses static lineage barcodes to pool sparse single-cell RNA counts, identify supported groups of genetically similar barcodes, and estimate their copy-number alterations. The aim is to support genotype-phenotype analysis while leaving weakly supported memberships unresolved. Grouping starts from self-centred, InferCNV-style smoothed expression and uses recursive binary splitting, with whole-cell bootstraps to assess split support and membership stability. Count-based CN contrasts and local smoothed haplotype-fraction (HF) differences can further subdivide these groups. An external normal expression reference, supplied directly or fitted as a mixture of available normal profiles, supports CN calling. A default 5% CN-independent depth-outlier component limits the influence of individual genes inconsistent with that reference. A dual-signal hidden Markov model (HMM) combines unsmoothed expression counts and phased allele counts to report uncertain CN states for each barcode and for directly pooled cells within each final group. A permutation diagnostic separately tests for barcode-associated regional signal.

# Usage

Cell Ranger preprocessing uses cellSNP-lite and Beagle; normal expression reference fitting is included in Python.

The methods document [pbpc.typ](docs/pbpc.typ) describes the current algorithm and
uncertainty summaries. From this directory, build it with
`typst compile docs/pbpc.typ docs/pbpc.pdf`; its figures are bundled in `docs/figures/`.

## Command line

For Linux x86_64, the [conda/mamba and pixi installation guide](docs/CONDA.md)
describes installation from our GitHub Pages channel without cloning this repo.
With that installation, use `barcodecnv` directly and run setup with `--tools path`.

From this directory, `uv sync --locked` installs the pinned environment.

For hg38 B-cell data, first fetch the resources and native tools, then run:

```sh
uv run barcodecnv setup --genome hg38
uv run barcodecnv run --outs /path/to/cellranger/outs \
  --cells cells.tsv --one-block --out results
```

This prints stages to the console, preprocesses the BAM/counts and then runs
inference, fitting the downloaded B-cell expression panel by default. Automatic
native-tool installation currently supports Linux x86_64. The hg38 phasing
resources are a substantial one-time download; see [RESOURCES.md](docs/RESOURCES.md)
for cache locations, disk requirements, retries and existing-tool overrides.
[PREPROCESSING.md](docs/PREPROCESSING.md) describes inputs and processing. The reusable bundle is
`results/preprocessing/prepared.h5`; final tables and plots are under
`results/inference/`. `results/pipeline.json` records overall completion or the
failed stage. Existing output directories are never overwritten. The prepared
bundle is retained if inference fails.

To run the stages separately:

```sh
uv run barcodecnv preprocess --outs /path/to/cellranger/outs \
  --cells cells.tsv --reference reference.tsv --one-block --out prepared_sample
uv run barcodecnv infer prepared_sample/prepared.h5 --out results
```

A pre-fitted `--reference` can be replaced by `--reference-panel /path/to/panel`.
The Python tool then fits the normal expression mixture and
saves the profile, weights and gene-selection audit. `fit-reference` exposes that
step on its own. See [REFERENCE_FITTING.md](docs/REFERENCE_FITTING.md) for panel format,
settings, the optional regional method and limitations.

The former inference-only `run` command is now named `infer`. If count tables
and phased alleles are already available, `infer` can load them directly:

```sh
uv run barcodecnv infer \
  --matrix filtered_feature_bc_matrix/ \
  --cells cells.tsv --genes genes.gtf.gz \
  --reference reference.tsv --alleles phased_counts.tsv.gz \
  --genetic-map genetic_maps/ --out results
```

Here the starting data are a 10x count matrix and annotation/phased-allele tables;
FASTQ alignment remains upstream;
BAM pileups/phasing are handled by `preprocess` or end-to-end `run`.
Allele counts are optional for expression-only analyses. Omit `--genetic-map`
when the allele table already includes per-locus genetic positions.

The command saves `results/prepared.h5`, the exact input bundle used, and records
its SHA-256 and original source paths in `run.json`. This snapshot is retained
if subsequent inference fails and can be reused without loading the source
tables again:

```sh
uv run barcodecnv infer results/prepared.h5 --out rerun
uv run barcodecnv signal results/prepared.h5 --out signal --permutations 199
```

`prepare` remains available for creating a reusable bundle separately. `signal`
also accepts `--matrix`, `--cells`, `--genes` and the optional loader arguments
directly; it does not require an external reference or fit HMMs. Choose either
a positional prepared bundle or count-input options; combining them is rejected.

Use `--help` on any command. `infer` and `run` default to 64 whole-cell bootstraps, 256
independent CN paths and 199 signal permutations. `--skip-signal` and
`--no-cn-refinement` omit those stages explicitly. Rolling haplotype-fraction
refinement is enabled by default; `--no-hf-refinement` restores the previous
expression/CN grouping. Use both refinement switches for expression-only
reporting groups. `--phase-iterations` changes
the fitting iteration limit; unconverged fits cannot report a completed result.
Output directories/files must be new. Failed inference runs retain `run.json` with their
failure status and any already completed diagnostic outputs.

Poisson–lognormal depth and joint barcode CN/phase inference are the defaults.
The two options can be varied independently; to reproduce the earlier NB and
shared conditional-phase workflow:

```sh
uv run barcodecnv infer sample.h5 --out nb_conditional \
  --depth-family nb --barcode-phase conditional --depth-outlier-probability 0
```

The depth likelihood includes a **5% CN-independent outlier mixture** by default.
This gives genes with large reference discrepancies an alternative explanation
without requiring a CN change. It is a prior mixture probability for each gene
observation, not a quota of discarded genes or a false-call probability. Allele
likelihoods, phase transitions and the CN transition prior are unchanged.

Use `--depth-outlier-probability P` to adjust it (`0 <= P < 1`). Consider lowering
it, for example to `0.01`, when you have evidence that your reference is well
matched; `0` disables the mixture. A larger value can also weaken genuine depth
signals, so increasing it is not an unconditional improvement.

```sh
uv run barcodecnv infer sample.h5 --out matched_reference \
  --depth-outlier-probability 0.01
```

The same setting is used during pooled phase/noise fitting, barcode inference
and group pseudobulk fitting. Inlier dispersion is refitted at each aggregation
level. The broad outlier component is a Poisson–lognormal distribution centered
at the diploid reference rate, with natural-log standard deviation 3, identical
across all CN states. It remains PLN when the inlier family is NB. This is a
contamination-mixture extension of the existing depth emission, not spatial
binning or likelihood tempering. `run.json` and HDF5 attributes record the
outlier probability and scale.

`pln` adapts Poisson–lognormal count likelihood. Its reference
rate is the **latent median**; NB uses the arithmetic mean. For PLN,
`sigma = sqrt(log1p(alpha))`; `alpha` remains the existing noise-search coordinate.
SciPy supplies mode location and adaptive QUADPACK quadrature. 
Numba compiles the integrand to a C callback; quadrature convergence
checks remain with SciPy. No approximate depth distribution is substituted.

`joint` sums and samples complete CN/phase paths in a 20-state HMM separately
for each barcode and fits dispersion using that joint likelihood. It avoids
the expected-log approximation but removes cross-barcode phase sharing for CN
calling. The pooled phase fit still aligns the rolling-HF grouping features.
`phase` in the output remains that common alignment; `barcode_phase` records
the separate joint marginals. With a symmetric phase prior, absolute haplotype
orientation can remain unidentified even when CN is well resolved. Probability
statements are still conditional on fitted noise and reference parameters.

### Input tables

CSV, TSV and gzip-compressed tables are accepted. IDs are strings, including
numeric-looking barcodes. Physical coordinates are one-based.

* **Expression:** 10x Matrix Market directory (`matrix.mtx`, `features.tsv`,
  `barcodes.tsv`, optionally gzipped), or a 10x HDF5 matrix. Only Gene Expression
  features enter RNA library totals. Whole-assay library sizes are computed
  before selecting annotated/reference-matched genes.
* **Cells:** `cell`, `barcode`, `block`. The existing `cell_barcode` and
  `lineage_barcode` aliases are accepted. Blocks define exchangeability for
  permutations (e.g. sample/batch). For one exchangeable cohort, `--one-block`
  explicitly substitutes a single block when that column is absent.
* **Genes:** GTF/GTF.gz, or `gene`, `chromosome`, `position` (alternatively
  `start`, `end`; their integer midpoint is used). Version suffixes are removed
  from gene IDs; ambiguous duplicates are rejected. Genes are sorted genomically.
* **Reference:** `gene`, `fraction`, for fractions relative to the whole assay;
  alternatively `gene`, `expression`, normalized over the entire supplied
  reference profile before gene selection. This must be a diploid reference
  suitable for the assay. It is not inferred from the cohort-median plotting
  reference. Omit it when preparing input solely for `signal`.
* **Alleles:** `cell`, `chromosome`, `position`, `h1`, `h2`, optionally `het`
  (default 1) and `phase_prob` (default 0.99, an explicit convention rather than
  a measured phase quality). Numbat `cell/CHROM/POS/AD/DP/GT/cM` is also accepted;
  `GT` must be phased `0|1` or `1|0`. Omit allele counts for expression-only data.
* **Genetic positions:** a PLINK four-column `.map` file or directory of maps. 
  Without a map, allele tables must contain consistent per-locus `genetic_cm` (or `cM`) values.

Only annotated cells are retained. Reference-supported annotated genes present
in the assay are used; the loader performs no CN-based gene selection. The
barcode/SNP count records are retained independently of gene count records.
The prepared `cell-counts-v1` bundle stores sparse CSC matrices (feature × cell),
explicit cell/gene/SNP labels, original library sizes and zero-based grid indices.
`CellBundle`, `read_bundle` and `write_bundle` in `barcodecnv.bundle` provide the
same boundary for programmatic loading.

### Outputs

* `prepared.h5`: reusable input snapshot when starting from count files.
  Bundle-input runs use the supplied file directly.
* `summary.png`: vertically stacked self-relative expression, external-reference
  expression, rolling haplotype fraction (when available), and CN calls.
  All panels use the same barcode/genomic order with called
  group boundaries. CN saturation represents conditional state probability;
  the separate strip shows the minimum expression/HF bootstrap split stability.
  Haplotype fractions are signed and centered across barcodes. Missing expression or HF coverage is
  white. No ground truth is used in this figure.
* `signal.png`, `signal.json`, `signal_null.csv`: observed regional count
  statistic, permutation histogram and Monte Carlo p-value, or an explicit
  `unassessable` result if the declared blocks allow no label exchanges.
* `groups.csv`, `order.csv`, `weighted_nodes.csv`, `core_nodes.csv`,
  `cn_nodes.csv`, `hf_nodes.csv`: reporting groups, ordering and decision audits. Group zero
  denotes unresolved membership. `phase_group` is a separate pooling decision.
  `pre_hf_group` preserves the groups before HF refinement; `hf_stability` is
  the additional split stability (empty when HF refinement was skipped).
* `barcode_cn_genes.csv.gz`: barcode/group IDs, gene coordinates, marginal MAP
  class and the four class probabilities. Includes unresolved barcodes.
* `group_cn_genes.csv.gz`: one row per resolved group and gene, with group size,
  `consensus_*` cell-weighted barcode probabilities, `pooled_*` group-pseudobulk
  posterior probabilities, and uncertain/conflicting member cell fractions.
* `group_cn_segments.csv.gz`: contiguous runs of pooled marginal MAP classes on
  the full gene/SNP grid, split at chromosome boundaries. Bounds are the first
  and last observed marker, not precise breakpoint estimates. Mean/minimum
  state probabilities summarize marker marginals, not whole-segment probability.
* `group_cn.png`: consensus, pseudobulk calls, uncertain member fractions and
  confident conflicts with pseudobulk calls in separate vertically stacked panels.
* `result.h5`: conditional probabilities (**barcode × marker × class**),
  labelled fine genomic grid, gene indices, shared phase probabilities,
  expression matrices (**gene × barcode**) and broad CN distance draws.
  `haplotype_features/` contains centered signed fractions and effective allele
  coverage (**region × barcode**), region centers and smoothing parameters.
  `group_calls/` stores group labels, sizes, consensus/pooled probabilities
  (**group × marker × class**), group phase and member-disagreement summaries.
  All groups and chromosomes follow the explicit stored labels. The gene tables
  are convenient subsets; the HDF5 retains SNP-only positions as well.
* `run.json`: parameters, numerical convergence, runtime and completion status.

The association diagnostic tests whether barcode labels explain regional count
structure within the declared blocks. Expression confounding can contribute;
it does not establish CN causation or general data quality. A nonsignificant
result does not prove the absence of useful barcode information. The statistic
and centring are fixed without barcode labels; permutations preserve barcode
cell counts within every block. Only the joint depth+allele statistic receives
the formal permutation p-value.

## Design and ownership

* `data.py`: validated, owned, read-only arrays of already prepared counts and
  reference expectations. Gene and SNP records map onto a common genomic grid;
  physical positions remain one-based, array indices are zero-based.
* `model.py`/`depth.py`: copy catalogue, count likelihoods and
  genomic/phase priors. This uses Poisson–lognormal depth by default
  and retaining NB as a comparator. For PLN the latent median is
  reference × dosage and log-scale noise is `sqrt(log1p(alpha))`. For
  NB the mean is reference × dosage and variance is `mean + alpha *
  mean**2`.
* `hmm.py`: numerical inference from supplied potentials, with no biological or
  barcode assumptions. Results own their arrays. Sampling produces independent
  conditional whole paths, not MCMC samples.
* `fitting.py`: caches emissions, fits shared phase and scalar dispersion, and
  returns explicit fitted values and diagnostics. No mutable global fit state.
* `depth_controls.py`: mixture configuration and cached CN-independent outlier
  likelihoods, shared by conditional and joint fitting.
* `joint.py`: combines CN and phase states for exact within-barcode inference,
  using the same numerical HMM engine and count likelihoods.
* `group_calls.py`: final group consensus and group-pseudobulk fits;
  `cn_reporting.py`: streaming gene tables and marginal-MAP segment exports.
* `loading.py`/`bundle.py`: file adapters and validated cell-count ownership;
  `signal.py`: label-blind features and permutation calibration;
  `bootstrap.py`/`grouping.py`: the existing PBPC2 expression decision rules;
  `haplotypes.py`: count-weighted rolling HF features and local splitting;
  `workflow.py`: orchestration and cell-weighted CN refinement;
  `plotting.py`/`cli.py`: presentation and filesystem output.

The count-inference entry points can also be used separately:

```python
from barcodecnv import (
    Model,
    FitOptions,
    DepthOptions,
    fit_phase_and_dispersion,
    infer_barcodes,
)

# pooled and barcodes are PreparedCounts with the same grid and SNP catalogue.
model = Model()
depth_options = DepthOptions(outlier_probability=0.05)
fit = fit_phase_and_dispersion(
    pooled, model, options=FitOptions(), depth_options=depth_options
)
posterior = infer_barcodes(
    barcodes, model, fit
)  # Inherits the pooled fit’s depth options.
paths = posterior.sample_paths(draws=256, seed=42)
```

`PreparedCounts` already contains expected diploid counts and genetic positions.
Those low-level calls do not group or pool cells. `CellBundle.prepared()` performs
explicit pseudobulking, and `barcodecnv.workflow.run_pbpc` owns the complete
expression-first orchestration. The smoothing port follows InferCNV 1.28.0
through step 14 and is checked against independently generated R fixtures.

## Preprocessing Cell Ranger outputs

Use [`barcodecnv preprocess`](docs/PREPROCESSING.md) to obtain a ready-to-run cell bundle
from an indexed Cell Ranger BAM, expression counts and cell-to-lineage map.
Supply a normal expression profile or a reference panel to fit.
See [PREPROCESSING.md](docs/PREPROCESSING.md) for local setup, inputs and assumptions.

## Statistical scope

The pooled CN and shared-phase fit uses structured variational updates
its product approximation is not the exact joint CN/phase posterior.
That fit supplies the common alignment used for HF features. Default barcode
CN calling instead integrates phase with a joint CN × phase HMM independently
within each barcode. The optional `conditional` route uses **expected log
likelihoods** under the pooled phase probabilities. The phase-switch process follows Numbat's
two-state continuous-time formulation.

Fitting first estimates pooled phase with heuristic dispersion, fits pooled
dispersion at fixed phase, and refits pooled phase. Default barcode inference
then fits dispersion using joint CN/phase evidence; the optional conditional
route instead holds pooled phase fixed. These are plug-in estimates; conditional
CN probabilities and paths do not integrate noise, reference or grouping
uncertainty. Nonconverged phase fits must not feed barcode inference.
Bootstrap membership stability and adaptive split statistics are heuristic
summaries, not calibrated posterior probabilities of clone identity. Reporting
group refinement does not silently change the groups used for phase fitting.

Default HF refinement follows CN refinement and only subdivides existing
groups; it cannot merge groups or recover already unresolved members. Phased
H1/H2 counts are pooled in 2 Mb base bins and smoothed over 10 Mb windows within
chromosomes. Whole-cell resampling and weak beta draws use shared base-bin
draws for overlapping windows, retaining their correlation. This conditions
on fitted MAP phase; phase uncertainty is not integrated. Cell bootstrap and
beta noise overlap as uncertainty sources, and the resulting split stability
is heuristic. No usable allele counts automatically skips this stage.

The HF audit records its parent group and the global barcode indices of that
parent; recursive `members` indices are local to the parent. Reported groups
and CN probabilities have distinct roles: changing this final refinement does
not refit barcode phase, noise or CN calls. The subsequent group reporting fit
is separate from those barcode results.

### Group CN reporting

After final grouping, descriptive consensus probabilities average barcode CN
class probabilities using cell counts as weights. These describe the inferred
composition of the group; they are not posterior probabilities of one shared
group state. A consensus MAP label is merely the class with the largest average.

For group calls, cells belonging to each final group are pseudobulked directly:
expression counts are summed per gene, H1/H2 counts per SNP, and whole-library
exposures are summed to obtain reference expectations. Distinct genes retain
separate records even if they map to the same genomic grid position. The group
pseudobulks get a new dispersion fit, followed by the joint CN/phase HMM; the
noise parameter is shared across group pseudobulks and is not copied from the
barcode fit. Gene-level count records are not multiplied across barcodes.

The model assumes one homogeneous CN/phase profile per group. Membership,
fitted group noise and reference are conditioned on, so pooled probabilities
can still be misleading if those assumptions fail. No group output feeds back
into barcode fitting or grouping. Group zero is excluded rather than pooled.
Group noise estimates, search convergence and boundary status are reported in
`run.json`, with the main fitted values also stored in the HDF5 group attributes.

At each position, `uncertain_cell_fraction` is the cell-weighted fraction of
barcodes whose maximum class probability is below 0.95. The
`consensus_conflict_cell_fraction` and `pooled_conflict_cell_fraction` instead
count cells belonging to confidently called barcodes (maximum probability at
least 0.95) whose MAP class differs from the indicated group call. These metrics
distinguish uncertain evidence from confident conflict; neither is a formal
test of group homogeneity. Cell fractions inherit each barcode's call, rather
than measuring CN states independently in each cell. Exact MAP ties use the
stored class order. Empty resolved-group results retain table headers.

The reporting API can also be used independently:

```python
from barcodecnv.group_calls import summarize_groups

group_calls = summarize_groups(
    bundle,
    model,
    posterior.classes,
    final_group_labels,
    depth_options=posterior.posterior.depth_options,
)
```

Here `posterior` is the `BarcodeFit` returned by `infer_barcodes` above. The
reporting function does not modify it. Both pseudobulk and consensus calls
should be read alongside the member-conflict summaries. An initial attempt
multiplying member barcode likelihoods was dropped after producing heavily
fragmented RBL1 calls; direct pseudobulking reduces this fragmentation.

## Development

```sh
cd barcodecnv
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Python 3.13 is selected by `.python-version`; `uv.lock` pins the development
environment. Runtime dependencies are NumPy, SciPy, Numba, pandas, h5py and
Matplotlib; development adds pytest and Ruff.

Ruff owns formatting, import sorting and linting for the package, tests, figure
scripts and validation scripts. Configuration lives in `pyproject.toml`, targeting
Python 3.13 with Ruff's default formatting style. To apply safe lint fixes and format:

```sh
uv run ruff check --fix .
uv run ruff format .
```

The rules cover basic correctness (`E4`, `E7`, `E9`, `F`), import ordering (`I`),
modern Python syntax (`UP`) and Bugbear (`B`). The immutable configuration
classes are allowed as defaults. `B905` is disabled because Numba's compiled
`zip` lacks `strict=` support; the lint cleanup preserves existing iteration
semantics. Unsafe automatic fixes are not enabled.

`.github/workflows/ci.yml` runs on pushes and pull requests, using the same
lint/format checks and pytest against the locked
Python 3.13 environment. It does not download preprocessing resources or run
external-data benchmarks.

### LLM Use

The design of the tool is by me and Chris Steel.
The intial Python implementation was ported by `codex:gpt6-astra` from earlier prototype Julia code written by me and `codex-gpt6-astra`.

# hg38 and B-cell setup

From the Python project directory:

```sh
uv sync --locked
uv run barcodecnv setup --genome hg38
uv run barcodecnv run --outs /path/to/cellranger/outs \
  --cells cells.tsv --out results
```

`setup` installs preprocessing tools and downloads hg38 resources plus the
**b-cells-v1 normal expression panel by default**. `run` and `preprocess` then
fit that panel to the selected cells when no explicit reference is given.
This initial distribution targets human B-cell data. Other built-in cell-type
panels are deferred; explicit `--reference` and `--reference-panel` overrides
remain available. No Julia checkout, R installation, root privileges or shell
activation is required. Python/uv and an internet connection are needed for setup.
Each run uses one donor and treats the selected cells as one cohort.

## Tools and data

Automatic tool installation currently supports **Linux x86_64 with glibc**.
A checksum-pinned micromamba 2.3.2 executable creates a private environment with
cellSNP-lite 1.2.3, bcftools 1.22 and OpenJDK 17 from conda-forge/Bioconda.
The environment is separate from the uv Python environment. Full resolved package
metadata and reported executable versions are recorded in `tools/installed.json`.
Dependencies/builds are solver-selected, not an immutable transitive environment
lock. No shell startup files or existing environments are modified.

The genomic resources match the existing Julia preprocessing:

| Resource | Source / version |
| --- | --- |
| Gene annotation | [GENCODE v44, GRCh38](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/) |
| Common SNP catalogue | [cellSNP 1000 Genomes phase 3, AF >= 5%, hg38](https://sourceforge.net/projects/cellsnp/files/SNPlist/) |
| Genetic maps | [Beagle PLINK GRCh38 maps](https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/) |
| Phasing reference | [1000 Genomes high coverage, 3,202 samples, 20220422 release](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/) |
| Beagle and bref3 converter | [Beagle 5.5, 27Feb25.75f](https://faculty.washington.edu/browning/beagle/beagle.html) |
| Normal expression panel | [b-cells-v1](https://gitlab.com/api/v4/projects/84721962/packages/generic/expression-panels/v1/b-cells-v1.tar.gz), the panel used by the Julia/Python reference fitter |

Only resources used by this workflow are downloaded; no genome FASTA, somatic-SNV
catalogues or RNA-editing databases are required here. Tools/data retain their
upstream licences and attribution. The package does not redistribute their binaries.
Bioconda is the [cellSNP-lite-recommended installation route](https://cellsnp-lite.readthedocs.io/en/latest/main/install.html).

The obsolete Beagle `1000Genomes_phased_v3/` URL is not used. Official EBI VCFs
are downloaded one chromosome at a time, checked against the published
`20220804_manifest.txt` MD5 values, and converted using bcftools and bref3.
Conversion retains biallelic polymorphic SNPs. X uses the updated v2 VCF and only
the GRCh38 non-PAR interval, 2,781,480–155,701,382, following the Julia pipeline.

Full setup transfers approximately 32.4 GB of source VCFs plus tools and smaller
resources, and can take substantial time. Allow approximately
15–20 GB free space: the completed phasing panel is about 9 GB, with additional
space for tools and one temporary chromosome VCF. The source VCF is removed after
successful conversion, retaining its source/checksum record. This is a one-time
shared setup cost, not a download repeated for each sample.

## Cache and reuse

The default cache root is `$XDG_DATA_HOME/barcodecnv`, or
`~/.local/share/barcodecnv` when XDG_DATA_HOME is unset. Setup writes the usable
configuration under `hg38/config.json`. Change the root with
`BARCODECNV_RESOURCE_DIR` or `setup --resource-dir /large/disk/barcodecnv`.
For an explicit custom directory, pass the printed `--config` path to `run` or set
`BARCODECNV_PREPROCESS_CONFIG` to it. Inference never downloads implicitly.

Configuration precedence is explicit `--config`, `BARCODECNV_PREPROCESS_CONFIG`,
working-directory `.preprocessing/config.json`, source-checkout
`.preprocessing/config.json`, then the shared cache. Existing local overrides
therefore continue to win. `setup_preprocessing.py` remains available to link an
existing installation; it is no longer needed on a new machine.

```sh
# Inspect destinations, sources and tool requirements without changing anything.
uv run barcodecnv setup --dry-run

# Small installation test: the saved config then defaults allele processing to 22.
uv run barcodecnv setup --chromosomes 22 --resource-dir /scratch/bcnv-test

# Fill in all remaining chromosomes later; completed chromosomes are reused.
uv run barcodecnv setup --resource-dir /scratch/bcnv-test

# Use existing native tools on PATH instead of installing the private environment.
uv run barcodecnv setup --tools path
```

`--tools path` is also the route for other Unix platforms; those installations have
not been tested here. Beagle/bref3 jars and all data are still fetched automatically.
Default allele chromosomes are 1–22 and X. A subset configuration is honoured by
`run`/`preprocess`; explicit `--chromosomes` overrides it and missing resources
fail before cellSNP processing. Expression genes are not restricted by this option.

Downloads use temporary files and publish only after checksum verification.
Interrupted HTTP downloads resume when the server provides a valid ETag or
Last-Modified validator; otherwise they restart safely. SHA-256 pins cover the
annotation, SNP catalogue, map archive, jars, micromamba and expression panel.
EBI's VCFs use its published MD5 pins plus locally recorded SHA-256 hashes.
Completed converted panels are rechecked against recorded SHA-256 hashes.
A modified cache entry fails with its path rather than being silently reused.
Only a completed setup publishes a new config; failed extensions preserve the
previous working config. A cache lock prevents concurrent writers. Individual
source records, conversion logs and `setup.json` support diagnosis/retry.

For reference fitting outside preprocessing, `--reference-panel b-cells-v1`
resolves the installed panel by name. The population phasing reference and normal
expression panel serve different purposes and must not be interchanged.

Automated resource-download and setup checks are in
[`tests/test_resources.py`](../tests/test_resources.py).

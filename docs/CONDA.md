# Conda installation and releases

The personal channel is hosted at
`https://jdm204.github.io/BarcodeCNV/channel`. Only **Linux x86_64** is supported;
other platforms are deferred. A package becomes available after its successful
Conda release workflow deployment.

## Install

```sh
conda create -n barcodecnv --override-channels --strict-channel-priority \
  -c https://jdm204.github.io/BarcodeCNV/channel \
  -c conda-forge -c bioconda barcodecnv
conda activate barcodecnv
barcodecnv setup --tools path --genome hg38
barcodecnv run --outs /path/to/cellranger/outs \
  --cells cells.tsv --one-block --out results
```

The same create command works with `mamba` in place of `conda`. To install a
specific release, replace `barcodecnv` with `barcodecnv=0.1.0`.
Use `--one-block` only for an exchangeable cohort; otherwise provide a `block`
column in the cell map.

For an application installation using pixi:

```sh
pixi global install \
  -c https://jdm204.github.io/BarcodeCNV/channel \
  -c conda-forge -c bioconda barcodecnv
barcodecnv setup --tools path --genome hg38
```

The conda package installs Python 3.13, the Python dependencies, cellSNP-lite,
bcftools and OpenJDK. Keep `--tools path` on the setup command so it uses these
executables instead of installing a second private environment. Setup downloads
the pinned Beagle/bref3 jars and hg38/B-cell resources separately, into the shared
cache. There are no package-install hooks downloading reference data. See
[RESOURCES.md](RESOURCES.md) for storage requirements and cache configuration.

## Release maintenance

All source configuration lives in this repository:

- `conda/meta.yaml`: conda-build recipe; reads the version and Python dependency
  constraints from `pyproject.toml`. Python 3.13 is the tested initial build.
- `.github/workflows/conda-release.yml`: builds and tests an installed package,
  checks installation from the generated channel, then deploys GitHub Pages.
- `conda/publish_channel.py`: merges packages without replacing existing files.
- `conda-channel` branch: generated packages and channel indexes. Do not merge
  this branch into `main`. Retaining it keeps older releases installable.

GitHub Pages must use **GitHub Actions** as its source under Settings → Pages.
The workflow uses the repository's `GITHUB_TOKEN`; no personal upload token or
Bioconda submission is needed. Repository policies must permit the workflow to
write the `conda-channel` branch and deploy to the `github-pages` environment.
This workflow owns the entire Pages site.

For a new release:

1. Update `pyproject.toml` and `uv.lock`, commit the changes, and check CI.
2. Create and push an annotated `vMAJOR.MINOR.PATCH` tag matching the version.
3. Follow the **Conda release** workflow through the deployment job.

For an existing tag such as `v0.1.0` that predates the workflow, use
Actions → Conda release → Run workflow on `main`, with `tag` set to `v0.1.0`.
This uses the workflow/recipe from the selected workflow revision and the source
from the exact tag. It never moves the tag. The workflow rejects version
mismatches and publishes only after tests pass.

The initial recipe build number is `0`. Increment `build.number` for a packaging
fix to the same upstream version. Published package filenames are immutable:
an identical file is reusable, but a different archive with the same name is
rejected. If only the Pages deployment fails, rerun the failed deployment job;
there is no need to rebuild. Rebuilding an already published version may produce
a different archive and require a new build number.

Reference downloads are deliberately excluded from release tests. The recipe
runs the Python test suite and checks that all three native executables can run;
publishing also tests the CLI and native-tool discovery in a fresh environment
installed through the generated channel metadata.

## Local build

Build from a clean staging directory so conda-build does not copy the checkout's
virtual environments or reference cache:

```sh
conda create -n bcnv-build --override-channels -c conda-forge \
  python=3.13 conda-build conda-index
conda activate bcnv-build
stage=$(mktemp -d)
git archive v0.1.0 | tar -x -C "$stage"
mkdir -p "$stage/conda"
cp conda/meta.yaml "$stage/conda/meta.yaml"
CONDA_CHANNEL_PRIORITY=strict conda-build "$stage/conda" \
  --override-channels -c conda-forge -c bioconda --no-anaconda-upload
```

This builds the tagged source using the current recipe. The recipe and runtime
dependencies are separate from the development environment in `uv.lock`.

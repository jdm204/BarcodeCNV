# Methods document figures

These assets belong to `../pbpc.typ`. Compile from the standalone `barcodecnv/`
directory with `typst compile docs/pbpc.typ docs/pbpc.pdf`. Compilation is self-contained.

## Current full RBL1 inference (28 September 2026)

- `rbl1-current-summary.png`: production barcode expression, HF and CN panels.
- `rbl1-current-group-cn.png`: production group consensus, pooled calls and
  member uncertainty/disagreement.
- `rbl1-current-run.json`: settings, input hash, fit diagnostics and runtime.
- `rbl1-current-comparison.json`: membership comparison with the earlier run.
- `rbl1-current-provenance.json`: original paths, asset hashes and scope.

These are outputs of a full inference rerun from the validated count bundle,
including regrouping and the barcode-association diagnostic. The existing
external reference is retained; reference fitting and BAM preprocessing were
not repeated. PLN, joint CN/phase and the 5% outlier model use current defaults;
the phase iteration budget was increased to 180. The final pooled phase fit
converged after 134 iterations. Two groups contain 234 and 442 cells; two
unresolved barcodes contain 19 cells. No complete CN truth is available.

The production PNGs are copied unchanged from the run. Recompute them using
the original bundle and command documented in `pbpc.typ`; the historical
rendering script below does not recreate this inference run.

## Saved fixed-group calling comparisons

- `calling-evaluation.csv`: 5% mixture rows across five histories and two focal
  reference conditions.
- `rbl1-group-calls.npz`: probability arrays and coordinates for the earlier
  fixed-group RBL1 calling comparison.
- `calling-evaluation.png`, `rbl1-group-calls.png`: scientific figures regenerated
  by `uv run python docs/figures/render_calling_evaluation.py`.
- `provenance.json`: original paths, hashes and evaluation scope.

These calling comparisons retain fixed memberships. They are distinct from the
current full RBL1 rerun. All use PLN, a joint CN/phase HMM and a 30 Mb persistence
length. Synthetic evaluation metrics require truth; real RBL1 panels show
conditional inference, without purity or accuracy claims.

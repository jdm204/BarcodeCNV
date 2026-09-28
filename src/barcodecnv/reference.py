"""Normal expression mixture fitting, ported from BarcodeCNV.Expression.

Global fitting adapts Numbat's fit_ref_sse: squared log-ratio error with free
softmax logits and the same fixed-budget Adam updates as Optimisers.jl. The
opt-in regional method uses SciPy BFGS with nuisance scales and a geometric median.
"""

import hashlib
import json
import logging
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from .loading import canonical_genes, read_10x, read_gene_coordinates, table

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferencePanel:
    id: str
    genome: str
    annotation: str
    genes: tuple[str, ...]
    profiles: tuple[str, ...]
    expression: np.ndarray
    metadata: pd.DataFrame
    provenance: dict

    def __post_init__(self):
        genes = tuple(canonical_genes(self.genes))
        if not all(str(x).strip() for x in (self.id, self.genome, self.annotation)):
            raise ValueError("panel identity, genome and annotation are required")
        if not genes or any(not g.strip() for g in genes):
            raise ValueError("empty panel gene IDs")
        if (
            not self.profiles
            or len(set(self.profiles)) != len(self.profiles)
            or any(not p.strip() for p in self.profiles)
        ):
            raise ValueError("profile IDs must be nonempty and unique")
        matrix = np.array(self.expression, dtype=float, copy=True)
        if matrix.shape != (len(genes), len(self.profiles)):
            raise ValueError("panel matrix axes disagree")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("panel values must be finite and nonnegative")
        if not np.allclose(matrix.sum(axis=0), 1, atol=1e-8, rtol=1e-6):
            raise ValueError(
                "panel fractions must sum to one on their measured gene universe"
            )
        matrix.flags.writeable = False
        object.__setattr__(self, "genes", genes)
        object.__setattr__(self, "expression", matrix)


def read_panel(directory):
    """Read the existing Julia schema-1 HDF5/TSV/TOML panel, checking hashes."""
    directory = Path(directory).resolve()
    manifest = tomllib.loads((directory / "dataset.toml").read_text())
    if manifest.get("complete") is not True or manifest.get("version") != 1:
        raise ValueError("incomplete or unsupported panel dataset")
    files = manifest.get("files", {})
    if not {"expression.h5", "profiles.tsv", "panel.toml"} <= files.keys():
        raise ValueError("panel integrity manifest omits required members")
    for name, digest in files.items():
        path = (directory / name).resolve()
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.is_relative_to(directory)
        ):
            raise ValueError("invalid panel dataset member")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError(f"panel checksum mismatch: {name}")
    spec = tomllib.loads((directory / "panel.toml").read_text())
    if spec["schema_version"] != 1 or spec["units"] != "fraction":
        raise ValueError("unsupported expression panel schema or units")
    metadata = pd.read_csv(
        directory / "profiles.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    fields = {
        "id",
        "study",
        "donor",
        "tissue",
        "cell_type",
        "cell_state",
        "capture",
        "platform",
        "processing",
        "n_cells",
    }
    if set(metadata.columns) != fields:
        raise ValueError("incomplete expression profile metadata")
    sizes = metadata.n_cells[metadata.n_cells != ""]
    if len(sizes) and (
        (pd.to_numeric(sizes) <= 0).any() or (pd.to_numeric(sizes) % 1 != 0).any()
    ):
        raise ValueError("known profile cell counts must be positive integers")
    with h5py.File(directory / "expression.h5") as f:
        profiles = tuple(f["profile_ids"].asstr()[:])
        if profiles != tuple(metadata.id):
            raise ValueError("profile metadata differs from matrix column order")
        genes = tuple(f["gene_ids"].asstr()[:])
        if len(f["gene_symbols"]) != len(genes):
            raise ValueError("gene labels and IDs differ")
        # HDF5.jl reverses dimensions on disk; do not guess orientation from shape
        # (square panels would silently pass an incorrect heuristic).
        matrix = f["expression"][:].T
    provenance = dict(
        source=str(directory), files=files, source_provenance=spec.get("provenance", {})
    )
    return ReferencePanel(
        spec["id"],
        spec["genome"],
        spec["annotation"],
        genes,
        profiles,
        matrix,
        metadata,
        provenance,
    )


def global_weights(matrix, observed, *, max_iter=2000):
    """Exact Optimisers.jl Adam parameterization, including bias correction/epsilon."""
    logits = np.zeros(matrix.shape[1])
    first = logits.copy()
    second = logits.copy()
    log_observed = np.log(observed)
    beta1_power, beta2_power = 0.9, 0.999
    for _ in range(max_iter):
        weights = softmax(logits)
        fitted = matrix @ weights
        if np.any(fitted <= 0):
            raise ValueError("nonpositive mixture prediction on fitting genes")
        gradient_weights = 2 * (matrix.T @ ((np.log(fitted) - log_observed) / fitted))
        gradient = (
            weights * (gradient_weights - weights @ gradient_weights) / len(observed)
        )
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + (1 - 0.999) * gradient**2
        logits -= (
            0.05
            * (first / (1 - beta1_power))
            / (np.sqrt(second / (1 - beta2_power)) + 1e-8)
        )
        beta1_power *= 0.9
        beta2_power *= 0.999
    return softmax(logits)


def fit_mixture(matrix, observed, *, max_iter=2000):
    k = matrix.shape[1]
    log_y = np.log(observed)

    def objective(x):
        weights = softmax(np.r_[x[: k - 1], 0.0])
        fitted = matrix @ weights
        residual = np.log(fitted) - log_y + x[-1]
        gradient_weights = matrix.T @ (residual / fitted) / len(observed)
        gradient = (weights * (gradient_weights - weights @ gradient_weights))[: k - 1]
        return 0.5 * np.mean(residual**2), np.r_[gradient, residual.mean()]

    initial = np.zeros(k)
    initial[-1] = np.mean(log_y - np.log(matrix @ np.full(k, 1 / k)))
    result = minimize(
        objective,
        initial,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": max_iter},
    )
    converged = bool(result.success and np.max(np.abs(result.jac)) <= 1e-7)
    return softmax(np.r_[result.x[: k - 1], 0.0]), converged


def consensus(votes):
    center = votes.mean(axis=1)
    for _ in range(1000):
        weights = 1 / np.maximum(np.linalg.norm(votes - center[:, None], axis=0), 1e-6)
        following = votes @ weights / weights.sum()
        if np.linalg.norm(following - center) < 1e-8:
            return following
        center = following
    return center


def contiguous_bins(chromosomes, width, offset=0):
    if width < 2 or not 0 <= offset < width:
        raise ValueError("invalid regional bin width or offset")
    bins = []
    tail = min(25, width)
    chromosomes = np.asarray(chromosomes)
    for chrom in dict.fromkeys(chromosomes):
        indices = np.flatnonzero(chromosomes == chrom)
        starts = sorted(
            set([0, *range(offset if offset else width, len(indices), width)])
        )
        local = [
            indices[start:end]
            for start, end in zip(starts, starts[1:] + [len(indices)])
        ]
        if len(local) > 1 and len(local[0]) < tail:
            local[1] = np.r_[local[0], local[1]]
            local.pop(0)
        if len(local) > 1 and len(local[-1]) < tail:
            local[-2] = np.r_[local[-2], local[-1]]
            local.pop()
        bins.extend(local)
    return bins


def regional_weights(matrix, observed, chromosomes, *, bin_genes=200, max_iter=2000):
    if bin_genes < 2:
        raise ValueError("regional bin_genes must be at least two")
    if matrix.shape[1] == 1:
        return np.ones(1), dict(informative_bins=[0, 0], failed_bins=0)
    results = []
    informative = []
    failures = 0
    for offset in (0, bin_genes // 2):
        votes = []
        for indices in contiguous_bins(chromosomes, bin_genes, offset):
            selected = indices[observed[indices] > 0]
            if len(selected) < 2:
                continue
            shapes = np.log(matrix[selected])
            shapes -= shapes.mean(axis=0)
            if np.max(np.abs(shapes - shapes[:, [0]])) <= 1e-8:
                continue
            weights, converged = fit_mixture(
                matrix[selected], observed[selected], max_iter=max_iter
            )
            failures += not converged
            votes.append(weights)
        if not votes:
            raise ValueError(
                "regional reference mixture is unidentified: no informative bins in a boundary layout"
            )
        informative.append(len(votes))
        results.append(consensus(np.array(votes).T))
    if failures:
        raise RuntimeError(
            f"regional reference fitting did not converge in {failures} bins; increase --reference-iterations"
        )
    return np.mean(results, axis=0), dict(
        informative_bins=informative, failed_bins=failures
    )


@dataclass(frozen=True)
class FittedReference:
    panel: ReferencePanel
    profile: dict
    weights: dict
    audit: dict


def fit_reference(
    panel,
    gene_ids,
    counts,
    genes,
    *,
    genome,
    method="global",
    min_cpm=2.0,
    max_iter=2000,
    bin_genes=200,
):
    if genome != panel.genome:
        raise ValueError(
            f"assay genome {genome} differs from panel genome {panel.genome}"
        )
    if method not in ("global", "regional_consensus"):
        raise ValueError("unknown reference fitting method")
    if not np.isfinite(min_cpm) or min_cpm < 0 or max_iter < 1 or bin_genes < 1:
        raise ValueError("invalid reference fitting settings")
    ids = canonical_genes(gene_ids)
    pooled = np.asarray(counts, dtype=float)
    if (
        pooled.shape != (len(ids),)
        or not np.isfinite(pooled).all()
        or np.any(pooled < 0)
    ):
        raise ValueError(
            "reference fitting needs one nonnegative finite pooled count per assay gene"
        )
    annotation = genes.set_index("gene")
    if not annotation.index.is_unique:
        raise ValueError("duplicate normalized annotation gene IDs")
    lookup = {g: i for i, g in enumerate(panel.genes)}
    overlap = [g for g in ids if g in lookup]
    absent = [g for g in ids if g not in lookup]
    selected = []
    rows = []
    for i, gene in enumerate(ids):
        if gene not in lookup:
            continue
        if gene not in annotation.index:
            raise ValueError(f"assay/reference gene absent from annotation: {gene}")
        j = lookup[gene]
        if panel.expression[j].mean() * 1e6 <= min_cpm or (
            method == "global" and pooled[i] == 0
        ):
            continue
        selected.append(i)
        rows.append(j)
    if len(selected) < 2:
        raise ValueError("fewer than two expressed genes overlap assay and reference")
    observed = pooled[selected]
    if observed.sum() <= 0:
        raise ValueError("no observed expression overlaps the reference")
    observed = observed / observed.sum()
    matrix = panel.expression[rows]
    details = {}
    if method == "global":
        weights = global_weights(matrix, observed, max_iter=max_iter)
        details["termination"] = "fixed_iteration_budget; not a convergence guarantee"
        predicted = matrix @ weights
        residual = np.log(predicted) - np.log(observed)
        gw = 2 * (matrix.T @ (residual / predicted))
        details["mean_squared_log_error"] = float(np.mean(residual**2))
        details["logit_gradient_max_abs"] = float(
            np.max(np.abs(weights * (gw - weights @ gw) / len(observed)))
        )
    else:
        chosen = annotation.loc[[ids[i] for i in selected]]
        positions = chosen.start if "start" in chosen else chosen.position
        # Match Julia's lexicographic chromosome, gene-start, ID ordering.
        order = sorted(
            range(len(chosen)),
            key=lambda i: (
                chosen.chromosome.iloc[i],
                int(positions.iloc[i]),
                chosen.index[i],
            ),
        )
        weights, details = regional_weights(
            np.maximum(matrix[order], 1e-12),
            observed[order],
            chosen.chromosome.iloc[order],
            bin_genes=bin_genes,
            max_iter=max_iter,
        )
    audit = dict(
        method=method,
        min_cpm=min_cpm,
        max_iter=max_iter,
        bin_genes=bin_genes,
        genome=genome,
        fitting_gene_ids=[ids[i] for i in selected],
        overlap_gene_ids=overlap,
        absent_gene_ids=absent,
        panel_id=panel.id,
        panel_annotation=panel.annotation,
        panel_provenance=panel.provenance,
        normalization="observed normalized on fitting genes; panel retains full measured universe",
        **details,
    )
    return FittedReference(
        panel,
        dict(zip(panel.genes, panel.expression @ weights)),
        dict(zip(panel.profiles, weights)),
        audit,
    )


def fit_from_files(matrix, cells, genes, panel_path, **options):
    counts, ids, cell_ids = read_10x(matrix)
    labels = table(cells).rename(columns={"cell_barcode": "cell"})
    if "cell" not in labels or labels.empty or labels.cell.duplicated().any():
        raise ValueError("reference fitting requires nonempty, unique selected cells")
    selected = pd.Index(cell_ids).get_indexer(labels.cell)
    if np.any(selected < 0):
        raise ValueError("annotated cells are missing from expression matrix")
    pooled = np.asarray(counts[:, selected].astype(np.float64).sum(axis=1)).ravel()
    fitted = fit_reference(
        read_panel(panel_path), ids, pooled, read_gene_coordinates(genes), **options
    )
    inputs = dict(
        matrix=str(Path(matrix).resolve()),
        cells=str(Path(cells).resolve()),
        genes=str(Path(genes).resolve()),
        selected_cells=len(selected),
        pooled_counts_sha256=hashlib.sha256(pooled.astype("<f8").tobytes()).hexdigest(),
    )
    for name, path in (("cells", cells), ("genes", genes)):
        with Path(path).open("rb") as handle:
            inputs[name + "_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return replace(fitted, audit={**fitted.audit, "assay_inputs": inputs})


def fit_for_args(args, matrix):
    if args.reference_panel is None:
        return None
    from .resources import resolve_expression_panel

    args.reference_panel = resolve_expression_panel(
        args.reference_panel, getattr(args, "config", None)
    )
    LOG.info("Fitting normal expression reference (%s)", args.reference_method)
    return fit_from_files(
        matrix,
        args.cells,
        args.genes,
        args.reference_panel,
        genome=args.genome,
        method=args.reference_method,
        min_cpm=args.reference_min_cpm,
        max_iter=args.reference_iterations,
        bin_genes=args.reference_bin_genes,
    )


def write_fit(directory, fitted):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        dict(gene=list(fitted.profile), fraction=list(fitted.profile.values()))
    ).to_csv(directory / "reference.tsv", sep="\t", index=False)
    metadata = fitted.panel.metadata.copy()
    metadata["weight"] = [fitted.weights[p] for p in metadata.id]
    metadata.to_csv(directory / "weights.tsv", sep="\t", index=False)
    audit = dict(fitted.audit)
    for name in ("fitting_gene_ids", "overlap_gene_ids", "absent_gene_ids"):
        values = audit.pop(name)
        (directory / f"{name}.txt").write_text("".join(g + "\n" for g in values))
        audit[name + "_count"] = len(values)
    (directory / "fit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False, default=str) + "\n"
    )


def add_reference_arguments(command, *, required=False, panel_only=False, genome=True):
    group = command.add_mutually_exclusive_group(required=required)
    if not panel_only:
        group.add_argument(
            "--reference",
            type=Path,
            help="pre-fitted gene/fraction or gene/expression profile",
        )
    group.add_argument(
        "--reference-panel",
        type=Path,
        help="expression panel directory or installed b-cells-v1; fit to selected cells",
    )
    if genome:
        command.add_argument(
            "--genome",
            default="hg38",
            help="assay genome; must equal panel genome (default: hg38)",
        )
    command.add_argument(
        "--reference-method", choices=["global", "regional_consensus"], default="global"
    )
    command.add_argument("--reference-min-cpm", type=float, default=2.0)
    command.add_argument(
        "--reference-iterations",
        type=int,
        default=2000,
        help="global Adam update budget / regional BFGS limit",
    )
    command.add_argument("--reference-bin-genes", type=int, default=200)

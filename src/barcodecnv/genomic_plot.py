"""Physical chromosome coordinates shared by all genome heatmaps."""

import numpy as np

# GRCh38 primary chromosome lengths (UCSC hg38), matching the resource FASTA index.
GRCH38_LENGTHS = dict(
    zip(
        [*[f"chr{i}" for i in range(1, 23)], "chrX", "chrY"],
        [
            248956422,
            242193529,
            198295559,
            190214555,
            181538259,
            170805979,
            159345973,
            145138636,
            138394717,
            133797422,
            135086622,
            133275309,
            114364328,
            107043718,
            101991189,
            90338345,
            83257441,
            80373285,
            58617616,
            64444167,
            46709983,
            50818468,
            156040895,
            57227415,
        ],
    )
)
MISSING_COLOR = "#e5e5e5"


def genome_layout(grid, *, genome="hg38", chromosome_sizes=None):
    """Chromosome offsets and lengths in bp; never infer length from gene density."""
    if chromosome_sizes is None:
        if genome not in ("hg38", "GRCh38"):
            raise ValueError(
                "provide chromosome_sizes for plotting genomes other than hg38/GRCh38"
            )
        sizes = GRCH38_LENGTHS
    else:
        sizes = {
            "chr" + str(c).removeprefix("chr"): n for c, n in chromosome_sizes.items()
        }
    layout = {}
    offset = 0
    chromosomes = np.asarray(grid.chrom)
    for chromosome in dict.fromkeys(grid.chrom):
        length = sizes.get("chr" + chromosome.removeprefix("chr"))
        if not isinstance(length, (int, np.integer)) or length <= 0:
            raise ValueError(
                f"provide a positive integer chromosome size for {chromosome}"
            )
        if grid.end[chromosomes == chromosome].max() > length:
            raise ValueError(
                f"marker coordinates exceed chromosome size for {chromosome}"
            )
        layout[chromosome] = (offset, length)
        offset += length
    return layout


def interval_heatmap(ax, values, chromosomes, left, right, layout, **options):
    """Plot disjoint half-open bp intervals; retain empty gaps as missing data."""
    values = np.asarray(values)
    chromosomes = np.asarray(chromosomes)
    left, right = np.asarray(left, float), np.asarray(right, float)
    rows = values.shape[0]
    artist = None
    for chromosome, (offset, length) in layout.items():
        ix = np.flatnonzero(chromosomes == chromosome)
        if not len(ix):
            continue
        starts, ends = left[ix], np.minimum(right[ix], length)
        if np.any(starts < 0) or np.any(ends <= starts):
            raise ValueError(f"invalid display intervals on {chromosome}")
        edges = np.unique(np.r_[starts, ends])
        positions = np.searchsorted(edges, starts)
        if len(np.unique(positions)) != len(ix) or np.any(edges[positions + 1] != ends):
            raise ValueError(f"overlapping display intervals on {chromosome}")
        # Explicit NaN cells keep gaps between covered HF bins blank. Marker
        # intervals already meet at midpoints and need no interpolation here.
        displayed = np.full((rows, len(edges) - 1, *values.shape[2:]), np.nan)
        displayed[:, positions] = values[:, ix]
        artist = ax.pcolormesh(
            (offset + edges) / 1e6,
            np.arange(rows + 1) - 0.5,
            np.ma.masked_invalid(displayed),
            shading="flat",
            rasterized=True,
            edgecolors="none",
            antialiased=False,
            **options,
        )
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_facecolor(MISSING_COLOR)
    ax.set_xlim(0, sum(length for _, length in layout.values()) / 1e6)
    return artist


def marker_heatmap(ax, values, grid, markers, layout, **options):
    """Each marker's colour extends to neighbouring midpoints, within its chromosome.

    No extrapolation to chromosome ends. Coincident genes share one interval and
    their finite values are averaged for display; stored values remain untouched.
    """
    markers, inverse = np.unique(markers, return_inverse=True)
    values = np.asarray(values)
    if len(markers) != values.shape[1]:
        source = np.moveaxis(values, 1, 0)
        totals = np.zeros((len(markers), *source.shape[1:]))
        counts = np.zeros_like(totals)
        supported = np.isfinite(source)
        np.add.at(totals, inverse, np.where(supported, source, 0))
        np.add.at(counts, inverse, supported)
        values = np.moveaxis(
            np.divide(
                totals, counts, out=np.full_like(totals, np.nan), where=counts > 0
            ),
            0,
            1,
        )
    else:
        # Reorder even when callers provide distinct markers out of grid order.
        values = values[:, np.argsort(inverse)]
    chromosomes = np.asarray(grid.chrom)[markers]
    left = grid.start[markers].astype(float) - 1
    right = grid.end[markers].astype(float)
    centers = (left + right) / 2
    for chromosome in dict.fromkeys(chromosomes):
        ix = np.flatnonzero(chromosomes == chromosome)
        midpoints = (centers[ix[:-1]] + centers[ix[1:]]) / 2
        right[ix[:-1]] = midpoints
        left[ix[1:]] = midpoints
    return interval_heatmap(ax, values, chromosomes, left, right, layout, **options)


def label_genome_axis(ax, layout, *, coordinate_axis=False):
    ax.set_xticks(
        [(offset + length / 2) / 1e6 for offset, length in layout.values()],
        [c.removeprefix("chr") for c in layout],
        fontsize=8,
    )
    ax.set_xlabel("Chromosome (genomic distance)")
    for offset, _ in list(layout.values())[1:]:
        ax.axvline(offset / 1e6, color="black", linewidth=0.4, alpha=0.5)
    if coordinate_axis:
        secondary = ax.secondary_xaxis("top")
        secondary.set_xlabel("Cumulative genomic position (Mb)")
        secondary.tick_params(labelsize=8)

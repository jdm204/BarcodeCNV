"""Owned prepared-count arrays; no preprocessing or reference estimation here."""

from dataclasses import dataclass

import numpy as np


def owned(value, dtype, ndim=1):
    source = np.asarray(value)
    if np.issubdtype(np.dtype(dtype), np.integer):
        if not np.all(np.isfinite(source)) or not np.all(source == np.floor(source)):
            raise ValueError("indices and counts must be finite integers")
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class Grid:
    chrom: tuple[str, ...]
    start: np.ndarray
    end: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "chrom", tuple(self.chrom))
        for name in ("start", "end"):
            object.__setattr__(self, name, owned(getattr(self, name), np.int64))
        if (
            not len(self.chrom)
            or len(self.start) != len(self.chrom)
            or len(self.end) != len(self.chrom)
        ):
            raise ValueError("grid axes must agree and be nonempty")
        if np.any(self.start < 1) or np.any(self.end < self.start):
            raise ValueError("invalid one-based inclusive coordinates")
        seen = set()
        for t, chrom in enumerate(self.chrom):
            if not isinstance(chrom, str) or not chrom:
                raise ValueError("chromosomes must be nonempty strings")
            if t and chrom == self.chrom[t - 1]:
                if self.start[t] <= self.end[t - 1]:
                    raise ValueError("grid intervals overlap or are unordered")
            else:
                if chrom in seen:
                    raise ValueError("chromosome blocks must be contiguous")
                seen.add(chrom)

    def __len__(self):
        return len(self.chrom)

    def same_as(self, other):
        return (
            self.chrom == other.chrom
            and np.array_equal(self.start, other.start)
            and np.array_equal(self.end, other.end)
        )


@dataclass(frozen=True)
class GeneCounts:
    group: np.ndarray
    marker: np.ndarray
    count: np.ndarray
    expected: np.ndarray
    weight: np.ndarray

    def __post_init__(self):
        _columns(self, ("group", "marker", "count"), ("expected", "weight"))


@dataclass(frozen=True)
class AlleleCounts:
    group: np.ndarray
    locus: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    het: np.ndarray

    def __post_init__(self):
        _columns(self, ("group", "locus", "h1", "h2"), ("het",))
        if np.any(self.het > 1):
            raise ValueError("heterozygosity weights must be in [0, 1]")


def _columns(obj, integers, floats):
    for names, dtype in ((integers, np.int64), (floats, np.float64)):
        for name in names:
            a = owned(getattr(obj, name), dtype)
            if np.any(~np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(obj, name, a)
    if len({len(getattr(obj, n)) for n in integers + floats}) != 1:
        raise ValueError("record columns must have equal lengths")


@dataclass(frozen=True)
class Loci:
    marker: np.ndarray
    position: np.ndarray
    population_phase: np.ndarray
    genetic_cm: np.ndarray

    def __post_init__(self):
        for name, dtype in (
            ("marker", np.int64),
            ("position", np.int64),
            ("population_phase", np.float64),
            ("genetic_cm", np.float64),
        ):
            a = owned(getattr(self, name), dtype)
            if np.any(~np.isfinite(a)):
                raise ValueError("SNP metadata must be finite")
            object.__setattr__(self, name, a)
        if (
            len(
                {
                    len(self.marker),
                    len(self.position),
                    len(self.population_phase),
                    len(self.genetic_cm),
                }
            )
            != 1
        ):
            raise ValueError("locus columns must agree")
        if (
            np.any(self.position < 1)
            or np.any(self.marker < 0)
            or np.any((self.population_phase < 0) | (self.population_phase > 1))
        ):
            raise ValueError("invalid SNP positions or phase probabilities")

    def __len__(self):
        return len(self.marker)


@dataclass(frozen=True)
class PreparedCounts:
    grid: Grid
    group_ids: tuple[str, ...]
    genes: GeneCounts
    alleles: AlleleCounts
    loci: Loci

    def __post_init__(self):
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        if (
            not self.group_ids
            or len(set(self.group_ids)) != len(self.group_ids)
            or any(not isinstance(g, str) or not g for g in self.group_ids)
        ):
            raise ValueError("group IDs must be unique nonempty strings")
        for a, bound in (
            (self.genes.group, len(self.group_ids)),
            (self.alleles.group, len(self.group_ids)),
            (self.genes.marker, len(self.grid)),
            (self.loci.marker, len(self.grid)),
            (self.alleles.locus, len(self.loci)),
        ):
            if np.any(a >= bound):
                raise ValueError("record index outside its labelled axis")
        loc = self.loci
        if np.any(np.diff(loc.marker) < 0):
            raise ValueError("loci must follow genomic grid order")
        if np.any(loc.position < self.grid.start[loc.marker]) or np.any(
            loc.position > self.grid.end[loc.marker]
        ):
            raise ValueError("SNP lies outside its grid interval")
        for i in range(1, len(loc)):
            if self.grid.chrom[loc.marker[i]] == self.grid.chrom[loc.marker[i - 1]]:
                if (
                    loc.position[i] <= loc.position[i - 1]
                    or loc.genetic_cm[i] < loc.genetic_cm[i - 1]
                ):
                    raise ValueError(
                        "SNP positions must increase and genetic map must not decrease"
                    )

    @property
    def shape(self):
        return len(self.group_ids), len(self.grid)

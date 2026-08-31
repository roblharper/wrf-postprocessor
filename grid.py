"""Build a regular evaluation grid over the domain extents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A regular (nx, ny, nz) grid: flat points plus the shape for cell building."""

    points: np.ndarray            # (N, 3) in x, y, z order
    shape: tuple[int, int, int]   # (nx, ny, nz)
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]  # the 1-D coordinate lines


def build_grid(
    bounds: dict[str, tuple[float, float]],
    resolution: tuple[int, int, int],
) -> Grid:
    """Regular grid over ``bounds`` (x/y/z min-max) at ``resolution`` (nx, ny, nz).

    Points are ordered ``ij`` (x outermost, z innermost) so the writer can build
    hexahedron cells with a simple index formula.
    """
    nx, ny, nz = resolution
    if min(nx, ny, nz) < 1:
        raise ValueError(f"resolution must be >= 1 per axis; got {resolution}.")

    xs = np.linspace(*bounds["x"], nx)
    ys = np.linspace(*bounds["y"], ny)
    zs = np.linspace(*bounds["z"], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    return Grid(points=points, shape=(nx, ny, nz), axes=(xs, ys, zs))

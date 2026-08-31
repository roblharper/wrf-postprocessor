"""Write the flowfield to binary VTK: one .vtu per timestep + a .pvd collection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import meshio


def _hexahedron_cells(shape: tuple[int, int, int]) -> np.ndarray:
    """Connect an (nx, ny, nz) ij-ordered point grid into hexahedron cells."""
    nx, ny, nz = shape

    def idx(i, j, k):
        return (i * ny + j) * nz + k

    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                cells.append([
                    idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                    idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
                ])
    return np.asarray(cells, dtype=np.int64)


def write_timestep(
    path: str | Path,
    points: np.ndarray,
    shape: tuple[int, int, int],
    fields: dict[str, np.ndarray],
) -> Path:
    """Write one binary .vtu: grid points, hexahedron cells, per-point ``fields``.

    ``fields`` maps a name to a per-point array (scalar (N,) or vector (N, 3)).
    """
    out = Path(path)
    cells = [("hexahedron", _hexahedron_cells(shape))] if all(n > 1 for n in shape) else []
    mesh = meshio.Mesh(points=np.asarray(points, dtype=np.float64), cells=cells,
                       point_data={k: np.asarray(v) for k, v in fields.items()})
    mesh.write(out, binary=True)
    return out


def write_collection(path: str | Path, entries: list[tuple[float, str]]) -> Path:
    """Write a .pvd collection so ParaView reads the (time, file) series as one.

    ``entries`` is a list of (time, relative_vtu_filename).
    """
    out = Path(path)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    lines += [f'    <DataSet timestep="{t}" file="{f}"/>' for t, f in entries]
    lines += ["  </Collection>", "</VTKFile>", ""]
    out.write_text("\n".join(lines))
    return out

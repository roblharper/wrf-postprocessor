"""End-to-end: a trained model in, a VTK flowfield time series out."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the postprocessor modules importable regardless of CWD.
_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))

import torch
import meshio

from wrf_pinn.models.mlp import MLP
from wrf_pinn.training.checkpoint import save_checkpoint

from model_loader import load_model
from denormalize import Recipe
from grid import build_grid
import run as postprocess


@pytest.fixture
def checkpoint(tmp_path):
    """Save a small untrained MLP as a checkpoint (weights are irrelevant here)."""
    torch.manual_seed(0)
    model = MLP()
    return save_checkpoint(tmp_path / "model.pt", model=model)


@pytest.fixture
def recipe_file(tmp_path):
    """A pre-processor-style metadata.json with an affine recipe."""
    columns = ["x", "y", "z", "t", "u", "v", "w", "theta", "p_prime", "source"]
    offset = [0.0, 0.0, 500.0, 1.8e9, 0.0, 0.0, 0.0, 300.0, -140.0, 0.0]
    scale = [5000.0, 5000.0, 500.0, 1800.0, 10.0, 10.0, 1.0, 5.0, 20.0, 1.0]
    meta = {"normalization": {"method": "minmax_pm1", "columns": columns,
                              "offset": offset, "scale": scale}}
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(meta))
    return p


def test_grid_shape_and_ordering():
    g = build_grid({"x": (0, 1), "y": (0, 2), "z": (0, 3)}, (4, 3, 2))
    assert g.shape == (4, 3, 2)
    assert g.points.shape == (4 * 3 * 2, 3)
    # ij ordering: z varies fastest
    assert g.points[0, 2] != g.points[1, 2] and g.points[0, 0] == g.points[1, 0]


def test_run_writes_vtu_per_timestep_and_a_pvd(checkpoint, tmp_path):
    model = load_model(checkpoint)
    out = tmp_path / "out"
    times = [-1.0, 0.0, 1.0]
    pvd = postprocess.run(model, out, resolution=(6, 5, 4), times=times)

    vtus = sorted(out.glob("*.vtu"))
    assert len(vtus) == len(times), "one .vtu per timestep"
    assert pvd.exists() and pvd.suffix == ".pvd"
    # the collection references every written file
    text = pvd.read_text()
    for v in vtus:
        assert v.name in text


def test_vtu_has_finite_state_fields_with_right_shape(checkpoint, tmp_path):
    model = load_model(checkpoint)
    out = tmp_path / "out"
    nx, ny, nz = 6, 5, 4
    postprocess.run(model, out, resolution=(nx, ny, nz), times=[0.0])

    mesh = meshio.read(sorted(out.glob("*.vtu"))[0])
    assert mesh.points.shape == (nx * ny * nz, 3)
    for name in ("u", "v", "w", "theta", "p_prime"):
        field = mesh.point_data[name]
        assert field.shape == (nx * ny * nz,)
        assert np.isfinite(field).all(), f"{name} has non-finite values"
    assert mesh.point_data["velocity"].shape == (nx * ny * nz, 3)


def test_recipe_denormalizes_into_physical_range(checkpoint, recipe_file, tmp_path):
    """With a recipe, coordinates land in physical extents, not [-1, 1]."""
    model = load_model(checkpoint)
    recipe = Recipe.from_metadata(recipe_file)
    out = tmp_path / "out"
    postprocess.run(model, out, resolution=(5, 5, 4), times=[0.0], recipe=recipe)

    mesh = meshio.read(sorted(out.glob("*.vtu"))[0])
    # z was offset 500, scale 500 -> physical z spans [0, 1000], well outside [-1, 1]
    z = mesh.points[:, 2]
    assert z.min() >= 0.0 - 1e-6 and z.max() <= 1000.0 + 1e-6
    assert z.max() > 1.0, "z should be denormalized to physical metres"


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pt")

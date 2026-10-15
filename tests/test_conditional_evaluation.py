"""Conditional checkpoint and source-unit evaluation contracts."""
from dataclasses import asdict
import json
import sys
from pathlib import Path

import meshio
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrf_pinn.data.conditional_case import STATE_VARS, COORD_NAMES
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
from model_loader import load_model
from evaluate import evaluate
import run


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    (root / "test").mkdir(parents=True)
    norm = dict(coord_names=list(COORD_NAMES), state_vars=list(STATE_VARS),
                coord_offset=[0]*4, coord_scale=[1]*4,
                state_offset=[10]*7, state_scale=[2]*7)
    metadata = dict(schema={"state_vars": list(STATE_VARS)}, normalization=norm,
                    cases={"train": []})
    (root / "metadata.json").write_text(json.dumps(metadata))
    targets = np.ones((2, 7), dtype=np.float32)
    targets[0, 0] = np.nan
    blob = dict(initial_coords=np.array([[0, 0, 0, 0]], dtype=np.float32),
                interior=np.array([[0, 0, 0, 1], [0, 0, 0, 2]], dtype=np.float32),
                targets=targets, boundary_coords=np.zeros((1, 1, 1, 4)),
                surface=np.zeros((1, 1, 5)), terrain=np.zeros((1, 3)), times=np.array([1, 2]))
    for var in STATE_VARS:
        blob.update({f"ini_mean_{var}": np.array([0.5]),
                     f"ini_modes_{var}": np.zeros((1, 1)), f"ini_coeffs_{var}": np.zeros((1, 1)),
                     f"bnd_mean_{var}_0": np.zeros(1), f"bnd_modes_{var}_0": np.zeros((1, 1)),
                     f"bnd_coeffs_{var}_0": np.zeros((1, 1))})
    np.savez(root / "test/tile0001_win01.npz", **blob)
    model = ConditionalModel(14, ConditionalModelConfig(hidden_width=5, hidden_layers=1, activation="silu"))
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    manifest = dict(model={"type": "conditional", "config": asdict(model.config), "latent_dim": 14},
                    state_vars=list(STATE_VARS), data={"datasets": [{"normalization": norm}]})
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "manifest": manifest}, checkpoint)
    return root, checkpoint, model


def test_evaluation_physical_errors_masks_exports_and_vtk(dataset, tmp_path):
    root, checkpoint, original = dataset
    model = load_model(checkpoint)
    assert model.config.activation == "silu"
    x, z = torch.randn(3, 4), torch.randn(1, 14)
    torch.testing.assert_close(model(x, z), original(x, z))
    report = evaluate(checkpoint, root, tmp_path / "out", batch_points=1, export_predictions=True)
    u = report["aggregate"]["u"]
    assert u["valid_count"] == 1 and u["excluded_target_count"] == 1
    assert u["bias"] == -2 and u["rmse"] == 2 and u["persistence_rmse"] == 1
    assert u["mse_skill_vs_persistence"] == -3
    assert report["calibration"]["status"] == "not_available"
    with np.load(tmp_path / "out/predictions/tile0001_win01.npz") as data:
        assert np.isnan(data["target"][0, 0])
        np.testing.assert_allclose(data["prediction"], 10)
    run.run(model, tmp_path / "vtk", resolution=(2, 2, 2), times=[0], conditioning=np.zeros(14))
    mesh = meshio.read(tmp_path / "vtk/flowfield_0000.vtu")
    assert set(STATE_VARS) <= set(mesh.point_data)


def test_reject_normalization_overlap_and_nonfinite(dataset, tmp_path):
    root, checkpoint, _ = dataset
    payload = torch.load(checkpoint, weights_only=False)
    payload["manifest"]["data"]["datasets"][0]["normalization"]["state_scale"][0] = 3
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="normalization differs"):
        evaluate(checkpoint, root, tmp_path / "out")
    payload["manifest"]["data"]["datasets"] = []
    payload["manifest"]["data"]["rank_case_ids"] = ["tile0001_win01"]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="overlap"):
        evaluate(checkpoint, root, tmp_path / "out")
    payload["manifest"]["data"]["rank_case_ids"] = []
    payload["model"]["net.2.bias"][0] = float("nan")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Nonfinite model"):
        evaluate(checkpoint, root, tmp_path / "out")


def test_legacy_requires_opt_in(dataset, tmp_path):
    _, _, model = dataset
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict()}, path)
    with pytest.raises(ValueError, match="no model manifest"):
        load_model(path)
    with pytest.warns(UserWarning, match="assuming tanh"):
        assert load_model(path, allow_legacy_conditional=True).config.activation == "tanh"

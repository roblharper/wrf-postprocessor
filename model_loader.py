"""Rebuild a trained MLP and restore its weights from a checkpoint."""

from __future__ import annotations

from pathlib import Path

from wrf_pinn.config.model import DEFAULT_MODEL, ModelConfig
from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.models.mlp import MLP


def load_model(
    checkpoint_path: str | Path,
    *,
    model_config: ModelConfig = DEFAULT_MODEL,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
):
    """Rebuild the MLP for ``model_config`` and load weights from ``checkpoint_path``.

    The checkpoint stores only ``model_state_dict``; the architecture is supplied
    here (defaults match training). Returns the model in ``eval`` mode.
    """
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}.")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise KeyError(f"{path} has no 'model_state_dict'; not a training checkpoint.")

    model = MLP(config=model_config, physics=physics)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model

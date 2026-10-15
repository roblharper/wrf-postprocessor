"""Rebuild a trained MLP and restore its weights from a checkpoint."""

from __future__ import annotations

from pathlib import Path
import warnings

from wrf_pinn.config.model import DEFAULT_MODEL, ModelConfig
from wrf_pinn.config.physics import DEFAULT_PHYSICS, PhysicsConfig
from wrf_pinn.models.mlp import MLP


def load_model(
    checkpoint_path: str | Path,
    *,
    model_config: ModelConfig = DEFAULT_MODEL,
    physics: PhysicsConfig = DEFAULT_PHYSICS,
    allow_legacy_conditional: bool = False,
):
    """Restore a self-describing conditional model or a legacy configured MLP.

    Conditional checkpoints without architecture metadata require explicit opt-in
    to the historical tanh convention. Returns the model in eval mode.
    """
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}.")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in payload:
        from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
        from wrf_pinn.data.conditional_case import STATE_VARS

        manifest = payload.get("manifest") or {}
        spec = manifest.get("model")
        state = payload["model"]
        if spec:
            if spec["type"] != "conditional":
                raise ValueError(f"Unsupported model type: {spec['type']}")
            config = ConditionalModelConfig(**spec["config"])
            latent_dim = spec["latent_dim"]
        elif allow_legacy_conditional:
            keys = sorted((k for k in state if k.startswith("net.") and k.endswith(".weight")),
                          key=lambda k: int(k.split(".")[1]))
            weights = [state[k] for k in keys]
            config = ConditionalModelConfig(hidden_layers=len(weights) - 1,
                                            hidden_width=state["net.0.weight"].shape[0],
                                            state_dim=weights[-1].shape[0])
            latent_dim = state["net.0.weight"].shape[1] - config.coord_dim
            warnings.warn("Legacy conditional checkpoint: explicitly assuming tanh, four coordinates, "
                          "and canonical variable order; training provenance is unavailable.", UserWarning)
        else:
            raise ValueError("Conditional checkpoint has no model manifest. Set "
                             "allow_legacy_conditional=True only for known tanh checkpoints.")
        names = tuple(manifest.get("state_vars", STATE_VARS))
        if len(names) != config.state_dim:
            raise ValueError("Checkpoint variable names do not match model output dimension.")
        model = ConditionalModel(latent_dim, config).to(dtype=state["net.0.weight"].dtype)
        model.load_state_dict(state)
        model.checkpoint_manifest = manifest
        model.state_names = names
        return model.eval()
    if "model_state_dict" not in payload:
        raise KeyError(f"{path} has no supported model weights.")

    model = MLP(config=model_config, physics=physics)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model

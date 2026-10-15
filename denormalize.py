"""Undo the pre-processor's affine normalization: value = offset + scale * normalized."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Recipe:
    """Per-column affine map from the pre-processor's ``metadata.json``."""

    def __init__(self, columns: list[str], offset: np.ndarray, scale: np.ndarray) -> None:
        if len(set(columns)) != len(columns) or offset.shape != (len(columns),) or scale.shape != (len(columns),):
            raise ValueError("Normalization names, offsets, and scales must have matching unique columns.")
        if not (np.isfinite(offset).all() and np.isfinite(scale).all()):
            raise ValueError("Normalization offsets and scales must be finite.")
        self._index = {name: i for i, name in enumerate(columns)}
        self._offset = offset
        self._scale = scale

    @classmethod
    def from_metadata(cls, metadata_path: str | Path) -> "Recipe":
        """Load the ``normalization`` block from a pre-processor metadata.json."""
        raw = json.loads(Path(metadata_path).read_text())
        norm = raw.get("normalization", raw)
        if "coord_names" in norm and "state_vars" in norm:
            return cls(
                columns=list(norm["coord_names"]) + list(norm["state_vars"]),
                offset=np.asarray(list(norm["coord_offset"]) + list(norm["state_offset"]), dtype=np.float64),
                scale=np.asarray(list(norm["coord_scale"]) + list(norm["state_scale"]), dtype=np.float64),
            )
        return cls(
            columns=list(norm["columns"]),
            offset=np.asarray(norm["offset"], dtype=np.float64),
            scale=np.asarray(norm["scale"], dtype=np.float64),
        )

    def denormalize(self, name: str, values: np.ndarray) -> np.ndarray:
        """Map one column's normalized values back to physical units by name."""
        if name not in self._index:
            raise KeyError(f"'{name}' not in normalization recipe columns.")
        i = self._index[name]
        return self._offset[i] + self._scale[i] * np.asarray(values)


def identity_denormalize(name: str, values: np.ndarray) -> np.ndarray:
    """No-op denormalizer for when no recipe is supplied (values already physical)."""
    return np.asarray(values)

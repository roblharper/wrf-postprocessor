"""Undo the pre-processor's affine normalization: value = offset + scale * normalized."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Recipe:
    """Per-column affine map from the pre-processor's ``metadata.json``."""

    def __init__(self, columns: list[str], offset: np.ndarray, scale: np.ndarray) -> None:
        self._index = {name: i for i, name in enumerate(columns)}
        self._offset = offset
        self._scale = scale

    @classmethod
    def from_metadata(cls, metadata_path: str | Path) -> "Recipe":
        """Load the ``normalization`` block from a pre-processor metadata.json."""
        raw = json.loads(Path(metadata_path).read_text())
        norm = raw.get("normalization", raw)
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

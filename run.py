"""Orchestrate: load a trained model, sweep a grid over time, write VTK flowfields.

    model checkpoint + domain extents + grid resolution
        -> evaluate the model on a regular grid at each requested time
        -> denormalize to physical units
        -> one binary .vtu per timestep + a .pvd collection

HRRR conditions are not used yet (an input slot for later).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from grid import build_grid
from vtk_writer import write_timestep, write_collection

log = logging.getLogger(__name__)

#: Model state columns written as fields (the model also outputs k_m, dropped).
STATE_NAMES: tuple[str, ...] = ("u", "v", "w", "theta", "p_prime")

#: Default evaluation bounds: the normalized [-1, 1] cube the model was trained on.
DEFAULT_BOUNDS = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0), "t": (-1.0, 1.0)}


def run(
    model,
    out_dir: str | Path,
    *,
    bounds: dict[str, tuple[float, float]] = DEFAULT_BOUNDS,
    resolution: tuple[int, int, int] = (32, 32, 16),
    times=(-1.0, 0.0, 1.0),
    recipe=None,
    device: str = "cpu",
    conditioning=None,
) -> Path:
    """Write a VTK flowfield time series; return the .pvd collection path.

    ``bounds`` and ``times`` are in the model's (normalized) coordinate space.
    ``recipe`` (a ``denormalize.Recipe`` or None) maps outputs and coordinates
    back to physical units; without it, values are written as-is.
    """
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid = build_grid(bounds, resolution)
    coords_phys = _physical_coordinates(grid.points, recipe)

    model.to(device).eval()
    entries: list[tuple[float, str]] = []
    for step, t in enumerate(times):
        fields = _evaluate(model, grid.points, float(t), recipe, device, torch, conditioning)
        name = f"flowfield_{step:04d}.vtu"
        write_timestep(out / name, coords_phys, grid.shape, fields)
        entries.append((_physical_time(t, recipe), name))
        log.info("wrote %s (t=%g, %d points)", name, t, grid.points.shape[0])

    pvd = write_collection(out / "flowfield.pvd", entries)
    log.info("wrote collection %s (%d timesteps)", pvd.name, len(entries))
    return pvd


def _evaluate(model, spatial_points, t, recipe, device, torch, conditioning=None) -> dict[str, np.ndarray]:
    """Run the model on (grid, t) and return denormalized per-point fields."""
    n = spatial_points.shape[0]
    coords = np.column_stack([spatial_points, np.full(n, t)]).astype(np.float32)
    with torch.no_grad():
        from wrf_pinn.models.conditional import ConditionalModel
        tensor = torch.as_tensor(coords, device=device, dtype=next(model.parameters()).dtype)
        if isinstance(model, ConditionalModel):
            if conditioning is None:
                raise ValueError("Conditional model export requires a case conditioning latent.")
            latent = torch.as_tensor(conditioning, device=device, dtype=tensor.dtype).reshape(1, -1)
            state = model(tensor, latent).cpu().numpy()
        else:
            state = model(tensor).cpu().numpy()

    fields: dict[str, np.ndarray] = {}
    for i, name in enumerate(getattr(model, "state_names", STATE_NAMES)):
        fields[name] = _denorm(recipe, name, state[:, i])
    fields["velocity"] = np.column_stack([fields["u"], fields["v"], fields["w"]])
    return fields


def _physical_coordinates(points: np.ndarray, recipe) -> np.ndarray:
    """Denormalize the x/y/z grid coordinates so VTK shows real extents."""
    return np.column_stack([
        _denorm(recipe, "x", points[:, 0]),
        _denorm(recipe, "y", points[:, 1]),
        _denorm(recipe, "z", points[:, 2]),
    ])


def _physical_time(t, recipe) -> float:
    return float(_denorm(recipe, "t", np.asarray([t]))[0])


def _denorm(recipe, name: str, values: np.ndarray) -> np.ndarray:
    """Apply the recipe when supplied; missing columns indicate incompatible data."""
    if recipe is None:
        return np.asarray(values)
    return recipe.denormalize(name, values)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from model_loader import load_model
    from denormalize import Recipe

    parser = argparse.ArgumentParser(
        description="Write a trained PINN's flowfield over time as binary VTK.",
    )
    parser.add_argument("checkpoint", type=Path, help="Trained model .pt checkpoint.")
    parser.add_argument("out_dir", type=Path, help="Output folder for .vtu + .pvd.")
    parser.add_argument("--recipe", type=Path, default=None,
                        help="Pre-processor metadata.json for denormalization.")
    parser.add_argument("--case", type=Path, help="Conditional case .npz supplying the latent.")
    parser.add_argument("--legacy-tanh", action="store_true",
                        help="Explicitly allow old conditional checkpoints with assumed tanh architecture.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bounds", type=float, nargs=6,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                        help="Normalized spatial bounds; default is the legacy [-1,1] cube.")
    parser.add_argument("--resolution", type=int, nargs=3, default=(32, 32, 16),
                        metavar=("NX", "NY", "NZ"), help="Grid resolution (default 32 32 16).")
    parser.add_argument("--times", type=float, nargs="+", default=[-1.0, 0.0, 1.0],
                        help="Normalized times to evaluate (default -1 0 1).")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    model = load_model(args.checkpoint, allow_legacy_conditional=args.legacy_tanh)
    conditioning = None
    if args.case:
        from wrf_pinn.data.conditional_case import read_conditional_case
        conditioning = read_conditional_case(args.case).z
    recipe = Recipe.from_metadata(args.recipe) if args.recipe else None
    bounds = (dict(zip(("x", "y", "z"), zip(args.bounds[::2], args.bounds[1::2])))
              if args.bounds else DEFAULT_BOUNDS)
    pvd = run(model, args.out_dir, bounds=bounds, resolution=tuple(args.resolution),
              times=args.times, recipe=recipe, conditioning=conditioning, device=args.device)
    print(f"Wrote flowfield time series to {pvd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# PINN post-processor

Turns a trained model into a flowfield. The pre-processor keeps data reading out
of the model; this keeps result exporting out of it. Feed a checkpoint, domain
extents, and a grid resolution; get the predicted flowfield over time as binary
VTK, ready to open in ParaView.

## What it produces

For a set of times it writes one binary `.vtu` per timestep plus a `.pvd`
collection that indexes them, so ParaView plays the series as an animation:

```
<out_dir>/
    flowfield.pvd            # time -> file collection
    flowfield_0000.vtu       # one per timestep
    ...
```

Each `.vtu` is a regular grid (hexahedron cells) carrying the predicted state as
per-point data: `u`, `v`, `w`, `theta`, `p_prime`, and a `velocity` vector.

## Pipeline

```
checkpoint + extents + resolution
    -> load model -> build grid -> evaluate over time -> denormalize -> write VTK
```

| Stage | File | Job |
|-------|------|-----|
| **Load** | `model_loader.py` | Rebuild the `MLP` and restore weights from a checkpoint. |
| **Grid** | `grid.py` | Regular grid over the domain extents at a chosen resolution. |
| **Denormalize** | `denormalize.py` | Undo the pre-processor's affine map (`value = offset + scale * normalized`) using its `metadata.json`. |
| **Write** | `vtk_writer.py` | Binary `.vtu` per timestep + a `.pvd` collection (via `meshio`). |
| **Orchestrator** | `run.py` | Wire the above + CLI. |

The model is continuous, so the grid resolution just chooses how densely to
sample it. Coordinates and state are written in **physical units** when a recipe
is supplied; without one, values pass through as-is.

## Usage

```bash
python run.py <checkpoint.pt> <out_dir> \
    [--recipe metadata.json] [--resolution NX NY NZ] [--times T ...] [-v]
```

`--recipe` is the pre-processor's `metadata.json`; with it, output is
denormalized to physical units. `--times` and the grid are in the model's
(normalized) coordinate space. HRRR conditions are not used yet (a slot for
later).

## Testing

```bash
pytest        # from this directory
```

`tests/test_end_to_end.py` saves a small model, runs the full path, and asserts:
one `.vtu` per timestep plus a `.pvd` that references them; grid shape and point
ordering; finite state fields of the right shape; and that a recipe denormalizes
coordinates into physical extents.

## Requirements

Python with `torch`, `numpy`, `meshio`, and the `wrf_pinn` package importable
(for the model architecture).

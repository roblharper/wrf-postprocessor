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

### Conditional checkpoint evaluation

With `wrf_pinn` installed (or its `src` on `PYTHONPATH`), evaluate a dataset's
`metadata.json` and `test/*.npz` using:

```sh
python evaluate.py model.pt /path/to/cases /path/to/results --device cpu --export-predictions
```

`evaluation.json` records source-unit errors, persistence skill, case/lead metrics,
checkpoint/metadata hashes and provenance notices. Prediction exports preserve
aligned targets, masks and normalized coordinates. `--batch-points` controls
inference memory. Units omitted by the dataset remain `unknown`.

New conditional checkpoints reconstruct their saved architecture. Historical
conditional checkpoints need `--legacy-tanh` to explicitly assume tanh and canonical
variable order; their training provenance cannot be verified. Legacy MLP
`model_state_dict` checkpoints remain supported by the VTK loader.

For conditional VTK export, supply `run.py --case case.npz --recipe metadata.json`.
Grid bounds/times must match the model's normalization: the legacy VTK defaults
are [-1,1]; conditional minmax_01 datasets use [0,1], so supply
`--bounds 0 1 0 1 0 1 --times 0.25 0.5 0.75 1` as appropriate for the case
(or explicit `bounds`/`times` in the Python API). A supplied recipe must contain every
exported field. Case evaluation derives its query points directly from the case.

These outputs evaluate deterministic predictions. Predictive intervals, coverage,
calibration and uncertainty plots are not implemented by this first slice.

### Conditioning representation audit

```sh
python conditioning_audit.py model.pt case.npz metadata.json audit.json --device cpu
```

Use `--legacy-tanh` only for known historical conditional checkpoints. This audit
flips matching POD mode/coefficient signs, verifies identical decoded fields, and
measures any prediction change. It also constructs mean-shifted fields with the
same latent to demonstrate information omitted by coefficient-only conditioning.
`--max-points` bounds deterministic inference sampling; `--mean-shift` is expressed
in normalized state units. Results carry hashes and source-unit labels.

The reference is the stored decoded POD field. This does not measure raw-field
truncation error, forecast accuracy, or predictive uncertainty. A shared basis
fitted only on training data and a matched retraining study remain separate work.

### Shared-basis reconstruction study

Make both `wrf_pinn/src` and the `wrf-preprocessor` module directory importable
through `PYTHONPATH`, then run:

```sh
python representation_study.py /path/to/conditioning_cases /path/to/study --ranks 0 2 4 8
```

The input contains `metadata.json`, `train/*.npz` and `test/*.npz` with POD fields,
initial grid/coordinates and boundary coordinates. Train/test case IDs, tile IDs
and window IDs must be disjoint; relative grid layouts must match. Metadata
split membership is checked when available. Optional `selection.json` records
original sources and hashes, with staged conditioning hashes verified before fit.

Each variable gets a fixed training-fitted mean/basis for its whole initial field
and another for its whole boundary field. `bases/*.npz` contains reusable arrays;
`study.json` records source/basis hashes, selection, source-unit RMSE, skill versus
the fixed training mean, and reconstructed spatial spread. Basis fitting never
uses held-out fields. Reported rank counts case-level coefficients per field and
is not equivalent to the current case-specific POD rank/budget.

Reference fields are decoded existing POD fields. This does not measure original
raw-field compression error. Existing normalization is retained and may include
test cases. Small within-simulation reconstruction gains do not establish
forecasting improvements, persistence skill, or calibrated uncertainty.

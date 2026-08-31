"""End-to-end smoke test: pre-process -> train -> post-process, on a known field.

Generates a FastEddy-shaped LES file whose fields are a fixed function of
position, runs it through the whole pipeline, then checks the VTK the
post-processor wrote against that same function, point for point.

    python end_to_end_demo.py [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import netCDF4

# ============================== CONFIG =======================================
# Where the sibling tools live. Edit these to match your layout; both are added
# to sys.path so their modules import regardless of where this file sits.
# wrf_pinn is imported as an installed package (pip install -e), not via path.
HERE = Path(__file__).resolve().parent
PREPROCESSOR_DIR = HERE.parent / "preprocessor"
POSTPROCESSOR_DIR = HERE

# LES tile the demo builds (FastEddy layout: x0/y0/z4 are independent axes).
X_EXTENT = (0.0, 3000.0)      # metres
Y_EXTENT = (0.0, 3000.0)
Z_EXTENT = (10.0, 1500.0)
LES_SHAPE = (7, 7, 7)         # nx, ny, nz
VALID_TIMES = (3600.0, 7200.0)  # two times so t normalizes to a real range

# Training + readout knobs.
EPOCHS = 400
GRID_RESOLUTION = (8, 8, 6)   # sampling density for the VTK readout (arbitrary)
PASS_TOL = 0.05               # rms error must be < PASS_TOL * (field range + 1)
# =============================================================================

for _d in (PREPROCESSOR_DIR, POSTPROCESSOR_DIR):
    sys.path.insert(0, str(_d))


def analytic_field(fx, fy, fz):
    """The known field as a function of fractional position (0..1) per axis."""
    tp = 2.0 * np.pi
    return {
        "u": 5.0 * np.sin(tp * fy),      # varies along y
        "v": 3.0 * np.sin(tp * fx),      # varies along x
        "w": 1.0 * np.sin(tp * fz),      # varies along z
        "theta": 2.0 * fx + fz,          # gradient in x, z
        "p_prime": 20.0 * (fx - fy),     # diagonal gradient
    }


def _fractional_grid(shape):
    """0..1 coordinates on a (z, y, x) grid, one linspace per axis."""
    nx, ny, nz = shape
    fz = np.linspace(0, 1, nz)[:, None, None]
    fy = np.linspace(0, 1, ny)[None, :, None]
    fx = np.linspace(0, 1, nx)[None, None, :]
    return np.broadcast_arrays(fx, fy, fz)


def write_fasteddy(path, valid_time):
    """A FastEddy LES file: independent x0/y0/z4 axes carrying analytic_field."""
    nx, ny, nz = LES_SHAPE
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", 1)
    for dim, n in (("x0", nx), ("y0", ny), ("z4", nz)):
        ds.createDimension(dim, n)
    ds.createVariable("x0", "f4", ("x0",))[:] = np.linspace(*X_EXTENT, nx)
    ds.createVariable("y0", "f4", ("y0",))[:] = np.linspace(*Y_EXTENT, ny)
    ds.createVariable("z4", "f4", ("z4",))[:] = np.linspace(*Z_EXTENT, nz)
    ds.createVariable("time", "f8", ("time",))[:] = [valid_time]

    fx, fy, fz = _fractional_grid(LES_SHAPE)
    field = analytic_field(fx, fy, fz)
    for var, key in (("uu", "u"), ("vv", "v"), ("ww", "w"),
                     ("theta", "theta"), ("pressure", "p_prime")):
        block = field[key].reshape((1, nz, ny, nx)).astype("f4")
        ds.createVariable(var, "f4", ("time", "z4", "y0", "x0"))[:] = block
    ds.close()


def write_dummy_hrrr(path):
    """Minimal HRRR anchor in metres: bbox covering the LES, both times."""
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", len(VALID_TIMES))
    ds.createDimension("y", 2)
    ds.createDimension("x", 2)
    xg, yg = np.meshgrid(np.linspace(X_EXTENT[0] - 100, X_EXTENT[1] + 100, 2),
                         np.linspace(Y_EXTENT[0] - 100, Y_EXTENT[1] + 100, 2))
    ds.createVariable("longitude", "f8", ("y", "x"))[:] = xg.copy()
    ds.createVariable("latitude", "f8", ("y", "x"))[:] = yg.copy()
    ds.createVariable("valid_time", "f8", ("time",))[:] = list(VALID_TIMES)
    ds.close()


def make_inputs(in_dir):
    """Write the LES tiles + dummy anchor the pipeline reads."""
    in_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(VALID_TIMES):
        write_fasteddy(in_dir / f"FE_NBL_demo_t{i}.nc", t)
    write_dummy_hrrr(in_dir / "hrrr_dummy.nc")


def preprocess(in_dir, case_dir):
    """Run the pre-processor; return (case_npy, metadata_json)."""
    import orchestrator
    orchestrator.run(in_dir, case_dir, chunk_size=50, test_fraction=0.0)
    npy = next(iter(case_dir.rglob("*.npy")), None)
    assert npy is not None, "pre-processor wrote no case"
    return npy, case_dir / "metadata.json"


def train(case_npy, metadata, checkpoint):
    """Train a small model data-only (PDE off) to reproduce the field."""
    import torch
    from wrf_pinn.data.case import read_case, CaseMetadata
    from wrf_pinn.models.mlp import MLP
    from wrf_pinn.training.train_pinn import train_pinn, TrainingSetup
    from wrf_pinn.training.checkpoint import save_checkpoint
    from wrf_pinn.config.conditions import ConditionsConfig, ConditionSpec
    from wrf_pinn.config.training import TrainingConfig, OptimizerConfig

    case = read_case(case_npy, CaseMetadata.load(metadata),
                     targets=("u", "v", "w", "theta", "p_prime"))
    off = lambda name: ConditionSpec(name, active=False, weight=0.0)
    conditions = ConditionsConfig(
        pde=off("pde"), boundary=off("boundary"), sensor=off("sensor"),
        simulation=ConditionSpec("simulation", active=True, weight=1.0),
    )
    cfg = TrainingConfig(epochs=EPOCHS, log_every=EPOCHS, device="cpu",
                         optimizer=OptimizerConfig(name="adam", learning_rate=1e-3))

    torch.manual_seed(0)
    model = MLP()
    history = train_pinn(model, TrainingSetup(case=case, conditions=conditions, training=cfg))
    save_checkpoint(checkpoint, model=model)
    return model, history.total[-1]


def postprocess(checkpoint, metadata, case_npy, out_dir):
    """Read out the flowfield over the exact box/time the case occupies."""
    import json
    from model_loader import load_model
    from denormalize import Recipe
    import run as postproc

    model = load_model(checkpoint)
    recipe = Recipe.from_metadata(metadata)

    # z and t are single-valued here (degenerate -> left unscaled), so read every
    # axis straight from the case rather than assuming a normalized [-1, 1] cube.
    columns = json.loads(Path(metadata).read_text())["normalization"]["columns"]
    data = np.load(case_npy)
    col = lambda name: data[:, columns.index(name)]
    bounds = {a: (float(col(a).min()), float(col(a).max())) for a in ("x", "y", "z", "t")}
    case_time = float(np.unique(col("t"))[0])
    return postproc.run(model, out_dir, recipe=recipe, bounds=bounds,
                        resolution=GRID_RESOLUTION, times=[case_time])


def verify(vtk_dir):
    """Check each VTK field reproduces analytic_field at every point."""
    import meshio
    ok = True
    for vtu in sorted(vtk_dir.glob("*.vtu")):
        mesh = meshio.read(vtu)
        pts = mesh.points

        def frac(axis):
            c = pts[:, axis]
            span = c.max() - c.min()
            return (c - c.min()) / span if span > 0 else np.zeros_like(c)

        expected = analytic_field(frac(0), frac(1), frac(2))
        for name, want in expected.items():
            got = mesh.point_data[name]
            rms = float(np.sqrt(np.mean((got - want) ** 2)))
            passed = rms < PASS_TOL * (float(np.ptp(want)) + 1.0)
            ok = ok and passed
            print(f"    {vtu.name:22s} {name:8s} rms={rms:.4f} "
                  f"(range {want.min():6.2f}..{want.max():6.2f})  "
                  f"[{'ok' if passed else 'OFF'}]")
    return ok


def main(argv=None):
    import tempfile
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else Path(tempfile.mkdtemp(prefix="e2e_"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] writing LES ({len(VALID_TIMES)} times) + dummy HRRR -> {out/'raw'}")
    make_inputs(out / "raw")

    print("[2/4] pre-processing -> normalized case + recipe")
    case_npy, metadata = preprocess(out / "raw", out / "cases")

    print("[3/4] training (data-only) to reproduce the field")
    _, loss = train(case_npy, metadata, out / "model.pt")
    print(f"      final data loss = {loss:.3e}")

    print("[4/4] post-processing -> VTK flowfield")
    pvd = postprocess(out / "model.pt", metadata, case_npy, out / "vtk")
    print(f"      wrote {pvd}")

    print("verifying VTK against the analytic field:")
    ok = verify(out / "vtk")
    print(f"\n{'PASS' if ok else 'FAIL'}: end-to-end round-trip ({out})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

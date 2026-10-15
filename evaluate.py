"""Evaluate conditional checkpoints against case targets in original data units.

Analysis belongs here; cluster submission remains in external drivers. This
module exports point-prediction evidence, not predictive uncertainty estimates.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import time

import numpy as np
import torch

from wrf_pinn.data.conditional_case import COORD_NAMES, STATE_VARS, read_conditional_case
from wrf_pinn.models.conditional import ConditionalModel
from denormalize import Recipe
from model_loader import load_model

log = logging.getLogger(__name__)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def moments(pred, target, persistence, valid):
    p, y, b = (a[valid].astype(np.float64) for a in (pred, target, persistence))
    if not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(b).all()):
        raise ValueError("Nonfinite predictions or persistence values at valid targets.")
    error = p - y
    return np.array([len(y), len(valid) - len(y), error.sum(), np.abs(error).sum(),
                     (error ** 2).sum(), ((b - y) ** 2).sum(), y.sum(), (y ** 2).sum()])


def metrics(values):
    n, excluded, bias, absolute, sse, baseline, total, squares = values
    variance_sum = max(0.0, squares - total ** 2 / n) if n else 0.0
    return {"valid_count": int(n), "excluded_target_count": int(excluded),
            "bias": float(bias / n) if n else None,
            "mae": float(absolute / n) if n else None,
            "rmse": float(np.sqrt(sse / n)) if n else None,
            "persistence_rmse": float(np.sqrt(baseline / n)) if n else None,
            "r2": float(1 - sse / variance_sum) if variance_sum else None,
            "mse_skill_vs_persistence": float(1 - sse / baseline) if baseline else None}


def evaluate(checkpoint, cases_dir, out_dir, *, device="cpu", batch_points=65536,
             export_predictions=False, allow_legacy_conditional=False):
    """Stream test cases and retain per-case/lead metrics and optional full fields."""
    if batch_points < 1:
        raise ValueError("batch_points must be positive.")
    checkpoint, cases_dir, out_dir = map(Path, (checkpoint, cases_dir, out_dir))
    metadata_path = cases_dir / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    norm = raw["normalization"]
    if (tuple(raw["schema"]["state_vars"]) != STATE_VARS or
            tuple(norm["state_vars"]) != STATE_VARS or tuple(norm["coord_names"]) != COORD_NAMES):
        raise ValueError("Dataset variable/coordinate order does not match the conditional reader.")
    paths = sorted((cases_dir / "test").glob("*.npz"))
    if not paths:
        raise ValueError("No test cases found.")
    names = [p.stem for p in paths]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate case IDs.")
    overlap = set(names) & set(raw.get("cases", {}).get("train", []))
    if overlap:
        raise ValueError(f"Evaluation cases occur in the training split: {sorted(overlap)[:5]}")
    model = load_model(checkpoint, allow_legacy_conditional=allow_legacy_conditional)
    if not isinstance(model, ConditionalModel):
        raise ValueError("Case evaluation currently supports ConditionalModel checkpoints.")
    manifest = model.checkpoint_manifest
    datasets = manifest.get("data", {}).get("datasets", [])
    if set(names) & set(manifest.get("data", {}).get("rank_case_ids", [])):
        raise ValueError("Evaluation case IDs overlap checkpoint training case IDs.")
    if datasets and not any(d.get("normalization") == norm for d in datasets):
        raise ValueError("Evaluation normalization differs from the checkpoint training recipe.")
    if tuple(model.state_names) != STATE_VARS:
        raise ValueError("Checkpoint variable order differs from the dataset.")
    recipe = Recipe.from_metadata(metadata_path)
    units = raw["schema"].get("units", {})
    notices = ["Calibration skipped: this checkpoint produces point predictions, not predictive intervals."]
    if not datasets:
        notices.append("Checkpoint dataset provenance unavailable; training/evaluation compatibility is unverified.")
    if any(units.get(v, "unknown") == "unknown" for v in (*COORD_NAMES, *STATE_VARS)):
        notices.append("Some units are unknown; denormalized values retain source-data units, not verified SI units.")
    if raw.get("provenance", {}).get("normalization_fit_scope") == "all_cases_including_test":
        notices.append("Normalization was fitted using test cases; this is not training-only preprocessing.")
    elif raw.get("provenance", {}).get("normalization_fit_scope") != "train_only":
        notices.append("Preprocessing fit scope is unknown in this legacy dataset.")
    for notice in notices:
        log.warning(notice)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    dtype = next(model.parameters()).dtype
    aggregate = np.zeros((len(STATE_VARS), 8), dtype=np.float64)
    records, lead_totals = [], {}
    start = time.monotonic()
    with torch.no_grad():
        for case_number, path in enumerate(paths, 1):
            case = read_conditional_case(path)
            if case.z.size != model.net[0].in_features - model.config.coord_dim:
                raise ValueError(f"Conditioning latent dimension mismatch in {case.name}.")
            if not np.isfinite(case.z).all():
                raise ValueError(f"Nonfinite conditioning latent in {case.name}.")
            pred = np.empty_like(case.targets, dtype=np.float64)
            z = torch.as_tensor(case.z, device=device, dtype=dtype).reshape(1, -1)
            for first in range(0, len(pred), batch_points):
                coords = torch.as_tensor(case.interior[first:first + batch_points], device=device, dtype=dtype)
                pred[first:first + batch_points] = model(coords, z).cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite model predictions in {case.name}.")
            initial = case.initial.transpose(0, 2, 1).reshape(len(STATE_VARS), -1).T
            lookup = {tuple(xyz): i for i, xyz in enumerate(case.initial_coords[:, :3])}
            if len(lookup) != len(initial):
                raise ValueError(f"Ambiguous initial spatial coordinates in {case.name}.")
            try:
                index = [lookup[tuple(xyz)] for xyz in case.interior[:, :3]]
            except KeyError as exc:
                raise ValueError(f"Initial and target grids do not align in {case.name}.") from exc
            persistence = initial[index].astype(np.float64)
            target = case.targets.astype(np.float64)
            for j, var in enumerate(STATE_VARS):
                pred[:, j] = recipe.denormalize(var, pred[:, j])
                target[:, j] = recipe.denormalize(var, target[:, j])
                persistence[:, j] = recipe.denormalize(var, persistence[:, j])
            match = re.fullmatch(r"(tile\d+)_(win\d+)", case.name)
            for lead, t in enumerate(np.unique(case.interior[:, 3]), 1):
                selected = case.interior[:, 3] == t
                physical_time = float(recipe.denormalize("t", np.array([t]))[0])
                for j, var in enumerate(STATE_VARS):
                    m = moments(pred[selected, j], target[selected, j], persistence[selected, j],
                                case.target_mask[selected, j] > 0)
                    aggregate[j] += m
                    lead_totals.setdefault((lead, var), np.zeros(8))[:] += m
                    records.append({"case_id": case.name, "tile": match[1] if match else None,
                                    "window": match[2] if match else None,
                                    "lead_index": lead, "time": physical_time,
                                    "time_units": units.get("t", "unknown"),
                                    "variable": var, "units": units.get(var, "unknown"), **metrics(m)})
            if export_predictions:
                folder = out_dir / "predictions"
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / f"{case.name}.npz", case_id=case.name,
                                    state_vars=np.array(STATE_VARS), coords_normalized=case.interior,
                                    prediction=pred, target=np.where(case.target_mask > 0, target, np.nan),
                                    target_mask=case.target_mask, persistence=persistence)
            elapsed = time.monotonic() - start
            log.info("evaluated %d/%d cases; elapsed %.1fs; ETA %.1fs", case_number, len(paths),
                     elapsed, elapsed / case_number * (len(paths) - case_number))
    report = {
        "schema_version": 1, "case_count": len(paths), "case_ids": names,
        "checkpoint_sha256": file_hash(checkpoint), "dataset_metadata_sha256": file_hash(metadata_path),
        "checkpoint_manifest": manifest, "dataset_id": raw.get("dataset_id"),
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device)},
        "notices": notices, "calibration": {"status": "not_available", "reason": "point_predictions_only"},
        "aggregate": {v: {"units": units.get(v, "unknown"), **metrics(aggregate[j])}
                      for j, v in enumerate(STATE_VARS)},
        "by_lead": [{"lead_index": lead, "variable": var, **metrics(m)}
                    for (lead, var), m in sorted(lead_totals.items())],
        "by_case_lead": records,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    log.info("wrote %s", out_dir / "evaluation.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("cases_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-points", type=int, default=65536)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--legacy-tanh", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(args.checkpoint, args.cases_dir, args.out_dir, device=args.device,
             batch_points=args.batch_points, export_predictions=args.export_predictions,
             allow_legacy_conditional=args.legacy_tanh)


if __name__ == "__main__":
    main()

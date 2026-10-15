"""Audit information omitted by case-specific coefficient-only conditioning.

Equivalent POD sign changes preserve decoded fields, but change the latent.
Mean shifts preserve the latent, but change decoded fields. Neither test is an
accuracy benchmark or a calibrated uncertainty estimate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wrf_pinn.data.conditional_case import STATE_VARS, decode_initial, decode_boundary, latent_vector
from wrf_pinn.models.conditional import ConditionalModel
from denormalize import Recipe
from evaluate import file_hash
from model_loader import load_model


class Blob(dict):
    @property
    def files(self):
        return list(self)


def equivalent_sign_flip(blob):
    """Flip matching coefficient/mode columns; preserve decoded conditioning."""
    out = Blob({k: v.copy() for k, v in blob.items()})
    for key in blob:
        if key.startswith(('ini_coeffs_', 'bnd_coeffs_')):
            out[key] *= -1
            out[key.replace('coeffs_', 'modes_', 1)] *= -1
    return out


def shifted_means(blob, shift):
    out = Blob({k: v.copy() for k, v in blob.items()})
    for key in out:
        if key.startswith(('ini_mean_', 'bnd_mean_')):
            out[key] += shift
    return out


def audit(checkpoint, case_path, metadata, out_path, *, device='cpu', max_points=4096,
          mean_shift=0.1, allow_legacy_conditional=False):
    if max_points < 1 or not np.isfinite(mean_shift) or mean_shift == 0:
        raise ValueError('Require positive max_points and a finite nonzero mean_shift.')
    with np.load(case_path) as source:
        blob = Blob({k: source[k] for k in source.files})
    original_z = latent_vector(blob)
    flipped = equivalent_sign_flip(blob)
    shifted = shifted_means(blob, mean_shift)
    initial, boundary = decode_initial(blob), decode_boundary(blob)
    flip_initial, flip_boundary = decode_initial(flipped), decode_boundary(flipped)
    shift_initial, shift_boundary = decode_initial(shifted), decode_boundary(shifted)
    z_shift = latent_vector(shifted)
    if not np.array_equal(z_shift, original_z):
        raise AssertionError('Mean-only perturbation unexpectedly changed latent.')
    recipe = Recipe.from_metadata(metadata)
    raw = json.loads(Path(metadata).read_text())
    units = raw.get('schema', {}).get('units', {})
    rows = []
    for var in STATE_VARS:
        for kind, original, sign, mean in [('initial', initial[var], flip_initial[var], shift_initial[var]),
                                          ('boundary', boundary[var], flip_boundary[var], shift_boundary[var])]:
            if not all(np.isfinite(x).all() for x in (original, sign, mean)):
                raise ValueError(f'Nonfinite conditioning field: {kind}/{var}')
            original_phys = recipe.denormalize(var, original.astype(np.float64))
            sign_phys = recipe.denormalize(var, sign.astype(np.float64))
            mean_phys = recipe.denormalize(var, mean.astype(np.float64))
            # Check in normalized space: source-unit scaling can amplify float32
            # BLAS roundoff, especially near zero pressure perturbation.
            if not np.allclose(original, sign, rtol=1e-6, atol=1e-6):
                raise AssertionError('Equivalent POD sign change altered decoded field.')
            rows.append({'field': kind, 'variable': var, 'units': units.get(var, 'unknown'),
                         'decoded_std': float(original_phys.std()),
                         'sign_flip_normalized_max_abs_change': float(np.max(np.abs(sign-original))),
                         'sign_flip_max_abs_change': float(np.max(np.abs(sign_phys-original_phys))),
                         'same_latent_mean_shift_rmse': float(np.sqrt(np.mean((mean_phys-original_phys)**2)))})
    model = load_model(checkpoint, allow_legacy_conditional=allow_legacy_conditional)
    if not isinstance(model, ConditionalModel):
        raise ValueError('Conditioning audit requires a ConditionalModel.')
    if tuple(model.state_names) != STATE_VARS:
        raise ValueError('Checkpoint state order differs from canonical dataset order.')
    datasets = model.checkpoint_manifest.get('data', {}).get('datasets', [])
    if datasets and not any(d.get('normalization') == raw['normalization'] for d in datasets):
        raise ValueError('Checkpoint normalization differs from audit metadata.')
    model.to(device).eval()
    dtype = next(model.parameters()).dtype
    coords = blob['interior']
    if len(coords) == 0 or not np.isfinite(coords).all():
        raise ValueError('Require finite nonempty interior coordinates.')
    indices = np.linspace(0, len(coords)-1, min(len(coords), max_points), dtype=int)
    x = torch.as_tensor(coords[indices], dtype=dtype, device=device)
    with torch.no_grad():
        predictions = []
        for z in (original_z, latent_vector(flipped)):
            if not np.isfinite(z).all():
                raise ValueError('Nonfinite latent.')
            predictions.append(model(x, torch.as_tensor(z, dtype=dtype, device=device)[None]).cpu().numpy())
    if not all(np.isfinite(p).all() for p in predictions):
        raise ValueError('Nonfinite model predictions.')
    changes = {}
    for j, var in enumerate(STATE_VARS):
        a, b = [recipe.denormalize(var, p[:, j].astype(np.float64)) for p in predictions]
        changes[var] = {'units': units.get(var, 'unknown'),
                        'prediction_change_rmse': float(np.sqrt(np.mean((a-b)**2))),
                        'prediction_change_max_abs': float(np.max(np.abs(a-b)))}
    report = {'schema_version': 1, 'case_id': Path(case_path).stem,
              'checkpoint_sha256': file_hash(checkpoint), 'case_sha256': file_hash(case_path),
              'metadata_sha256': file_hash(metadata), 'checkpoint_manifest': model.checkpoint_manifest,
              'runtime': {'device': str(device), 'torch': str(torch.__version__)},
              'sampled_point_count': len(indices), 'sampling': 'deterministic evenly spaced case rows',
              'normalized_mean_shift': mean_shift,
              'mean_shift_latent_identical': True,
              'sign_flip_latent_l2_change': float(np.linalg.norm(latent_vector(flipped)-original_z)),
              'decoded_field_checks': rows, 'equivalent_sign_prediction_changes': changes,
              'limitations': ['Decoded POD fields are the reference; no raw-field reconstruction error measured.',
                              'Mean shift is a constructed information-loss counterexample, not an observed case.',
                              'Equivalent sign sensitivity is a representation stress test, not forecast accuracy.',
                              'All-column sign flip is one equivalent convention; not a distribution of uncertainty.',
                              'No shared training-fitted basis or retrained surrogate comparison in this audit.',
                              'Legacy checkpoint provenance and source units may be unknown.']}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'case', 'metadata', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-points', type=int, default=4096)
    parser.add_argument('--mean-shift', type=float, default=0.1)
    parser.add_argument('--legacy-tanh', action='store_true')
    args = parser.parse_args()
    audit(args.checkpoint, args.case, args.metadata, args.output, device=args.device,
          max_points=args.max_points, mean_shift=args.mean_shift,
          allow_legacy_conditional=args.legacy_tanh)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()

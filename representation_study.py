"""Compare a training-fitted shared POD basis with its fixed-mean baseline.

Reference fields are decoded case-local POD fields, not original raw simulation
fields. This is an experimental reconstruction study, not a forecasting engine.
Requires wrf_pinn and the wrf-preprocessor module directory on PYTHONPATH.
"""
import argparse
import json
import logging
from pathlib import Path
import re

import numpy as np

from shared_basis import SharedBasis
from wrf_pinn.data.conditional_case import STATE_VARS, decode_initial, decode_boundary
from conditioning_audit import Blob
from denormalize import Recipe
from evaluate import file_hash

log = logging.getLogger(__name__)


def groups(path):
    match = re.fullmatch(r'(tile\d+)_(win\d+)', path.stem)
    if not match:
        raise ValueError(f'Cannot establish tile/window groups for {path.name}')
    return match.groups()


def load_fields(path):
    with np.load(path) as source:
        blob = Blob({k: source[k] for k in source.files})
    initial = decode_initial(blob)
    boundary = decode_boundary(blob)
    return blob, {'initial': initial, 'boundary': boundary}


def layout(blob):
    initial = np.asarray(blob['initial_coords'][:, :3], dtype=np.float64)
    boundary = np.asarray(blob['boundary_coords'], dtype=np.float64)
    if not np.isfinite(initial).all() or not np.isfinite(boundary).all():
        raise ValueError('Nonfinite conditioning coordinates.')
    return {'grid': np.asarray(blob['initial_grid']),
            'initial': initial-initial.min(axis=0),
            'boundary': boundary-boundary.reshape(-1, 4).min(axis=0)}


def study(cases_dir, output_dir, *, ranks=(0, 2, 4, 8)):
    cases_dir, output_dir = Path(cases_dir), Path(output_dir)
    if not ranks or any(r < 0 for r in ranks):
        raise ValueError('Require nonnegative ranks.')
    paths = {s: sorted((cases_dir/s).glob('*.npz')) for s in ['train','test']}
    if len(paths['train']) < 2 or not paths['test']:
        raise ValueError('Require at least two training cases and one held-out case.')
    ids = {s: [p.stem for p in pp] for s, pp in paths.items()}
    if set(ids['train']) & set(ids['test']):
        raise ValueError('Train/test case IDs overlap.')
    grouped = {s: [groups(p) for p in pp] for s, pp in paths.items()}
    for column, label in [(0,'tile'), (1,'window')]:
        if {g[column] for g in grouped['train']} & {g[column] for g in grouped['test']}:
            raise ValueError(f'Train/test {label} groups overlap.')
    metadata = cases_dir/'metadata.json'
    raw = json.loads(metadata.read_text())
    for split in paths:
        declared = raw.get('cases', {}).get(split)
        if declared is not None and not set(ids[split]) <= set(declared):
            raise ValueError(f'Selected {split} cases are absent from metadata split.')
    selection_path = cases_dir/'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection is not None:
        declared = {(r['split'],r['case_id']):r['conditioning_sha256'] for r in selection['records']}
        for split, pp in paths.items():
            for p in pp:
                if declared.get((split,p.stem)) != file_hash(p):
                    raise ValueError(f'Staged conditioning source hash mismatch: {p.stem}')
    if tuple(raw['normalization']['state_vars']) != STATE_VARS:
        raise ValueError('Normalization state order differs from canonical fields.')
    recipe = Recipe.from_metadata(metadata)
    units = raw.get('schema', {}).get('units', {})
    loaded = {}
    reference_layout = None
    for split, pp in paths.items():
        loaded[split] = []
        for p in pp:
            blob, fields = load_fields(p)
            current = layout(blob)
            if reference_layout is None:
                reference_layout = current
            elif any(current[k].shape != reference_layout[k].shape or
                     not np.allclose(current[k], reference_layout[k], rtol=1e-5, atol=1e-5)
                     for k in current):
                raise ValueError(f'Conditioning grid layouts differ: {p.stem}')
            loaded[split].append(fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_dir = output_dir/'bases';basis_dir.mkdir(exist_ok=True)
    records, artifacts = [], []
    for field in ['initial','boundary']:
        for var in STATE_VARS:
            train = np.stack([f[field][var] for f in loaded['train']])
            basis = SharedBasis.fit(train, max_rank=max(ranks))
            dest = basis_dir/f'{field}_{var}.npz';basis.save(dest)
            artifacts.append({'field':field, 'variable':var, 'path':str(dest),
                              'sha256':file_hash(dest), 'field_shape':list(basis.field_shape),
                              'fitted_rank':basis.modes.shape[1], 'feature_count':len(basis.mean)})
            for split in ['train','test']:
                data = np.stack([f[field][var] for f in loaded[split]])
                target = recipe.denormalize(var, data.astype(np.float64))
                mean = recipe.denormalize(var, basis.decode(basis.encode(data, rank=0)))
                for requested in ranks:
                    rank = min(requested, basis.modes.shape[1])
                    reconstructed = recipe.denormalize(var, basis.decode(basis.encode(data, rank=rank)))
                    for i, case_id in enumerate(ids[split]):
                        error = reconstructed[i]-target[i]
                        mse = float(np.mean(error**2))
                        baseline = float(np.mean((mean[i]-target[i])**2))
                        # Spatial spread at fixed height (initial) or fixed face/time (boundary).
                        target_spread = float(np.sqrt(np.mean(np.var(target[i],axis=-1))))
                        pred_spread = float(np.sqrt(np.mean(np.var(reconstructed[i],axis=-1))))
                        records.append({'case_id':case_id, 'split':split, 'field':field, 'variable':var,
                                        'units':units.get(var,'unknown'), 'requested_rank':requested,
                                        'effective_rank':rank, 'point_count':target[i].size,
                                        'rmse':float(np.sqrt(mse)), 'training_mean_rmse':float(np.sqrt(baseline)),
                                        'mse_skill_vs_training_mean':1-mse/baseline if baseline else None,
                                        'target_spatial_spread_rms':target_spread,
                                        'reconstructed_spatial_spread_rms':pred_spread,
                                        'spatial_spread_ratio':pred_spread/target_spread if target_spread else None})
            log.info('fitted/evaluated %s/%s rank=%d',field,var,basis.modes.shape[1])
    summary = []
    for field in ['initial','boundary']:
        for var in STATE_VARS:
            for rank in ranks:
                selected = [r for r in records if r['split']=='test' and r['field']==field
                            and r['variable']==var and r['requested_rank']==rank]
                count = sum(r['point_count'] for r in selected)
                mse = sum(r['rmse']**2*r['point_count'] for r in selected)/count
                baseline = sum(r['training_mean_rmse']**2*r['point_count'] for r in selected)/count
                summary.append({'field':field,'variable':var,'requested_rank':rank,
                                'effective_rank':selected[0]['effective_rank'], 'units':units.get(var,'unknown'),
                                'rmse':float(np.sqrt(mse)), 'training_mean_rmse':float(np.sqrt(baseline)),
                                'mse_skill_vs_training_mean':1-mse/baseline if baseline else None,
                                'mean_spatial_spread_ratio':float(np.mean([r['spatial_spread_ratio'] for r in selected]))
                                  if all(r['spatial_spread_ratio'] is not None for r in selected) else None})
    report = {'schema_version':1,'case_ids':ids,'basis_fit_scope':'selected_train_cases_only',
              'runtime':{'numpy':np.__version__},
              'selection_manifest':selection,
              'selection_manifest_sha256':file_hash(selection_path) if selection is not None else None,
              'reference':'stored_decoded_case_local_POD_fields',
              'coordinate_contract':'matching relative local xyz and relative boundary time, same grid/flatten ordering',
              'metadata_sha256':file_hash(metadata), 'normalization_fit_scope':raw.get('provenance',{}).get('normalization_fit_scope','unknown'),
              'sources':{s:[{'case_id':p.stem,'sha256':file_hash(p)} for p in pp] for s,pp in paths.items()},
              'bases':artifacts,'heldout_summary':summary,'by_case':records,
              'limitations':['Stored-field reconstruction does not measure original raw POD truncation error.',
                             'Existing normalization may include test cases; basis fitting alone is train-only.',
                             'Small within-simulation blocked sample; no independent-regime validation.',
                             'Rank counts case-level coefficients per field, unlike existing per-case POD coefficients.',
                             'No surrogate retraining, forecast persistence comparison, or predictive calibration.',
                             'Relative grid matching permits translated local domains; absolute geography/terrain are not encoded.']}
    (output_dir/'study.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases_dir',type=Path);parser.add_argument('output_dir',type=Path)
    parser.add_argument('--ranks',nargs='+',type=int,default=[0,2,4,8])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    study(args.cases_dir,args.output_dir,ranks=tuple(args.ranks))


if __name__=='__main__':main()

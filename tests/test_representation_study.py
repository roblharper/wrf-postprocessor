"""Blocked reconstruction must fit only training fields and reject overlap."""
import json
from pathlib import Path
import numpy as np
import pytest
from test_conditional_evaluation import dataset
from representation_study import study
from shared_basis import SharedBasis


@pytest.fixture
def blocked(dataset):
    root, _, _ = dataset
    original = root/'test/tile0001_win01.npz'
    with np.load(original) as source:
        blob = {k:source[k] for k in source.files}
    original.unlink()
    (root/'train').mkdir()
    for split, name, value in [('train','tile1000_win00',0),('train','tile1001_win01',2),('test','tile2000_win24',4)]:
        changed = {k:v.copy() for k,v in blob.items()}
        changed['initial_grid'] = np.array([1,1,1])
        for key in changed:
            if '_mean_' in key:
                changed[key][:] = value
        np.savez(root/split/(name+'.npz'), **changed)
    metadata = json.loads((root/'metadata.json').read_text())
    metadata['cases'] = {s:[p.stem for p in (root/s).glob('*.npz')] for s in ['train','test']}
    (root/'metadata.json').write_text(json.dumps(metadata))
    return root


def test_train_only_fit_mean_baseline_and_heldout_projection(blocked, tmp_path):
    result = study(blocked, tmp_path/'result', ranks=(0,1))
    u = [r for r in result['heldout_summary'] if r['field']=='initial' and r['variable']=='u']
    assert u[0]['rmse'] == pytest.approx(6)
    assert u[1]['rmse'] == pytest.approx(0)
    basis = SharedBasis.load(tmp_path/'result/bases/initial_u.npz')
    np.testing.assert_allclose(basis.mean, 1)
    test = blocked/'test/tile2000_win24.npz'
    with np.load(test) as source:
        changed = {k:source[k] for k in source.files}
    changed['ini_mean_u'][:] = 100
    np.savez(test, **changed)
    study(blocked, tmp_path/'second', ranks=(0,1))
    second = SharedBasis.load(tmp_path/'second/bases/initial_u.npz')
    np.testing.assert_array_equal(second.mean, basis.mean)
    np.testing.assert_array_equal(second.modes, basis.modes)


def test_reject_group_overlap_and_layout_mismatch(blocked, tmp_path):
    test = blocked/'test/tile2000_win24.npz'
    test.rename(blocked/'test/tile1000_win24.npz')
    with pytest.raises(ValueError, match='tile groups overlap'):
        study(blocked, tmp_path/'out')
    (blocked/'test/tile1000_win24.npz').rename(test)
    with np.load(test) as source:
        blob = {k:source[k] for k in source.files}
    blob['initial_grid'] = np.array([2,1,1])
    np.savez(test, **blob)
    with pytest.raises(ValueError, match='grid layouts differ'):
        study(blocked, tmp_path/'out')

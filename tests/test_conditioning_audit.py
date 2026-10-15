"""Representation counterexamples must preserve their stated invariants."""
import numpy as np
import pytest
import torch

from test_conditional_evaluation import dataset
from conditioning_audit import Blob, audit, equivalent_sign_flip, shifted_means
from wrf_pinn.data.conditional_case import decode_initial, decode_boundary, latent_vector


def test_sign_and_mean_counterexamples(dataset, tmp_path):
    root, checkpoint, model = dataset
    case = root / 'test/tile0001_win01.npz'
    with np.load(case) as source:
        blob = Blob({k: source[k] for k in source.files})
    for key in blob:
        if 'coeffs_' in key:
            blob[key][:] = 1
        if 'modes_' in key:
            blob[key][:] = 0.25
    flipped = equivalent_sign_flip(blob)
    shifted = shifted_means(blob, 0.1)
    np.testing.assert_array_equal(latent_vector(shifted), latent_vector(blob))
    np.testing.assert_array_equal(latent_vector(flipped), -latent_vector(blob))
    for decoder in [decode_initial, decode_boundary]:
        for var, original in decoder(blob).items():
            np.testing.assert_array_equal(decoder(flipped)[var], original)
            np.testing.assert_allclose(decoder(shifted)[var]-original, 0.1, atol=1e-7)
    np.savez(case, **blob)
    payload = torch.load(checkpoint, weights_only=False)
    for param in payload['model'].values():
        param.fill_(0.1)
    torch.save(payload, checkpoint)
    report = audit(checkpoint, case, root/'metadata.json', tmp_path/'audit.json', max_points=1)
    assert report['mean_shift_latent_identical']
    assert report['sampled_point_count'] == 1
    assert report['equivalent_sign_prediction_changes']['u']['prediction_change_rmse'] > 0
    for row in report['decoded_field_checks']:
        assert row['sign_flip_max_abs_change'] == 0
        assert row['same_latent_mean_shift_rmse'] == pytest.approx(0.2, abs=1e-6)


def test_audit_rejects_invalid_sampling(dataset, tmp_path):
    root, checkpoint, _ = dataset
    with pytest.raises(ValueError, match='positive max_points'):
        audit(checkpoint, root/'test/tile0001_win01.npz', root/'metadata.json', tmp_path/'a.json', max_points=0)

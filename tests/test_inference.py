import numpy as np
import pandas as pd
import pytest

from floodbench.inference import compare_to_baseline, holm


def test_holm_known_answer():
    # classic example: sorted p = .01,.02,.03,.04 with m=4 -> .04, .06, .06, .06 (monotone)
    adj = holm(np.array([0.04, 0.01, 0.03, 0.02]))
    np.testing.assert_allclose(adj, [0.06, 0.04, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm(np.array([0.9, 0.8])).max() == 1.0


def mse_table(model_factor, noise=0.02, n=11, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(1e6, 4e6, size=n)
    model = base * model_factor * np.exp(rng.normal(0, noise, size=n))
    rows = []
    for s, (b, m) in enumerate(zip(base, model)):
        rows += [dict(model="base", h=1, season=2010 + s, mse=b), dict(model="m", h=1, season=2010 + s, mse=m)]
    return pd.DataFrame(rows)


def test_clearly_better_model_is_detected():
    res = compare_to_baseline(mse_table(0.7), "base").iloc[0]
    assert res["rmse_ratio"] == pytest.approx(np.sqrt(0.7), rel=0.05)
    assert res["p_t"] < 1e-4 and res["wins"] == 11
    assert res["mse_diff_pct_ci_high"] < 0


def test_equal_models_are_not_flagged_and_equivalence_is_shown():
    res = compare_to_baseline(mse_table(1.0, noise=0.01), "base", margin=0.05).iloc[0]
    assert res["p_t"] > 0.05
    assert res["p_tost"] < 0.05  # inside +-5% RMSE


def test_noisy_equal_models_cannot_claim_equivalence():
    res = compare_to_baseline(mse_table(1.0, noise=0.5, seed=3), "base", margin=0.05).iloc[0]
    assert res["p_tost"] > 0.05


def test_clearly_worse_model_is_not_equivalent():
    res = compare_to_baseline(mse_table(1.5, noise=0.01), "base", margin=0.05).iloc[0]
    assert res["rmse_ratio"] > 1.2 and res["p_tost"] > 0.5

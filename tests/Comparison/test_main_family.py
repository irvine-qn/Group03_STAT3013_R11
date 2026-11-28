"""Họ chính: 3 bản chính × 3 tầm = 9 phép so với AnomAR1, Holm chung, không làm tròn."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from comparison_tables import CONTENDERS, RESULTS, SOURCES, TABLES, load_all, main_family

OWNER = {"gam": "GAM", "lgbm_raw": "LightGBM", "chronos2": "Chronos-2"}
SAME = ["n_seasons", "rmse_ratio", "wins", "mse_diff_pct", "mse_diff_pct_ci_low", "mse_diff_pct_ci_high",
        "p_t", "p_wilcoxon", "p_tost"]


@pytest.fixture(scope="module")
def family():
    return main_family(load_all())


def test_family_has_exactly_the_nine_main_comparisons(family):
    assert len(family) == 9
    assert sorted(zip(family["model"], family["h"])) == sorted((m, h) for m in CONTENDERS for h in (1, 2, 4))
    assert list(family.columns)[-2:] == ["p_t_holm_model", "p_t_holm_main"]


def test_main_adjustment_is_never_smaller_than_per_model(family):
    assert (family["p_t_holm_main"] >= family["p_t_holm_model"] - 1e-12).all()
    assert (family["p_t_holm_main"] >= family["p_t"]).all()


def test_chronos2_32_days_by_hand():
    """Tính tay từ predictions.csv: MSE từng mùa, t-test cặp, rồi Holm (p nhỏ nhất trong 9 → nhân 9)."""
    pred = pd.read_csv(RESULTS / "Chronos-2" / "tables" / "predictions.csv").query("h == 4")
    mse = (pred.assign(se=(pred["y_target"] - pred["pred"]) ** 2)
               .groupby(["model", "season"])["se"].mean().unstack("model"))
    p = stats.ttest_rel(mse["chronos2"], mse["anom_ar1"]).pvalue
    row = main_family(load_all()).set_index(["model", "h"]).loc[("chronos2", 4)]
    assert row["p_t"] == pytest.approx(p, rel=1e-9)
    assert row["p_t_holm_main"] == pytest.approx(9 * p, rel=1e-9)
    assert row["p_t_holm_main"] == pytest.approx(0.045, abs=5e-4)


@pytest.mark.parametrize("model", CONTENDERS)
def test_other_columns_match_each_model_table(family, model):
    own = pd.read_csv(RESULTS / OWNER[model] / "tables" / "vs_anom_ar1.csv").query("model == @model")
    own = own.set_index("h").sort_index()
    mine = family[family["model"] == model].set_index("h").sort_index()
    pd.testing.assert_frame_equal(mine[SAME], own[SAME], check_dtype=False, atol=1e-4)
    np.testing.assert_allclose(mine["p_t_holm_model"], own["p_t_holm"], atol=1e-4)


def test_written_table_is_not_rounded(family):
    written = pd.read_csv(TABLES / "vs_anom_ar1_main.csv")
    pd.testing.assert_frame_equal(written, family, check_dtype=False, rtol=1e-12)
    assert (written["p_t"].astype(str).str.len() > 8).all()

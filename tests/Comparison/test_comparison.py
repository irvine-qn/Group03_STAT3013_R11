"""Bảng so sánh chung phải khớp với bảng của từng mô hình và phải dùng chung một mốc AnomAR1."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comparison_tables import (CONTENDERS, MAIN, RESULTS, SOURCES, baseline_check, build_tables, holm,
                               load_all)

SRC = Path(__file__).resolve().parents[2] / "src" / "Comparison"
CORE = ["rmse_ratio", "wins", "mse_diff_pct", "mse_diff_pct_ci_low", "mse_diff_pct_ci_high",
        "p_t", "p_wilcoxon", "p_tost"]


@pytest.fixture(scope="module")
def tables():
    return build_tables(load_all(), baseline_check())


def owned(source):
    return [m for m in SOURCES[source]]


def test_modules_have_comparison_prefix():
    names = [p.name for p in SRC.glob("*.py")]
    assert names and all(n.startswith("comparison_") for n in names)


def test_every_folder_uses_the_same_anom_ar1():
    check = baseline_check()
    assert check["same_rows"].all()
    assert (check["max_abs_diff"] <= 1e-6).all()


@pytest.mark.parametrize("source", list(SOURCES))
def test_rmse_all_matches_each_model(tables, source):
    mine = tables["rmse_all"].set_index("model")
    theirs = pd.read_csv(RESULTS / source / "tables" / "rmse.csv").set_index("model")
    for model in owned(source):
        if model in theirs.index:
            np.testing.assert_allclose(mine.loc[model, ["1", "2", "4"]].astype(float),
                                       theirs.loc[model, ["1", "2", "4"]].astype(float))


@pytest.mark.parametrize("source", list(SOURCES))
def test_tests_match_each_model(tables, source):
    mine = tables["vs_anom_ar1_all"].set_index(["model", "h"])
    theirs = pd.read_csv(RESULTS / source / "tables" / "vs_anom_ar1.csv").set_index(["model", "h"])
    shared = [k for k in theirs.index if k[0] in owned(source)]
    assert shared
    pd.testing.assert_frame_equal(mine.loc[shared, CORE], theirs.loc[shared, CORE], check_dtype=False, atol=1e-4)


@pytest.mark.parametrize("name", ["rmse_by_phase", "rmse_by_size"])
@pytest.mark.parametrize("source", ["GAM", "LightGBM", "Chronos-2"])
def test_condition_tables_match_each_model(tables, source, name):
    mine = tables[name].set_index(["h", "model"])
    theirs = pd.read_csv(RESULTS / source / "tables" / f"{name}.csv").set_index(["h", "model"])
    keys = [k for k in mine.index if k in theirs.index]
    assert len(keys) == 6  # anom_ar1 và bản chính của thư mục đó, 3 tầm
    pd.testing.assert_frame_equal(mine.loc[keys], theirs.loc[keys], check_dtype=False)


def test_global_holm_covers_the_nine_main_comparisons(tables):
    t = tables["vs_anom_ar1_all"]
    main = t[t["role"] == "main"]
    assert sorted(main["model"].unique()) == sorted(CONTENDERS) and len(main) == 9
    assert t.loc[t["role"] != "main", "p_t_holm_main"].isna().all()
    # bảng làm tròn p đến 4 chữ số, nên Holm tính lại từ p đã làm tròn chỉ khớp tới ~1e-3
    np.testing.assert_allclose(main["p_t_holm_main"], holm(main["p_t"]), atol=1e-3)
    assert (main["p_t_holm_main"] >= main["p_t"] - 1e-9).all()


def test_season_table_ratio_is_one_for_baseline(tables):
    s = tables["season_rmse"]
    assert set(s["model"]) == set(MAIN) and len(s) == 4 * 3 * 11
    assert (s.loc[s["model"] == "anom_ar1", "ratio_vs_anom_ar1"] == 1).all()


def test_mismatched_baseline_is_refused(tmp_path):
    for source in SOURCES:
        dst = tmp_path / source / "tables"
        dst.mkdir(parents=True)
        shutil.copy(RESULTS / source / "tables" / "predictions.csv", dst)
    path = tmp_path / "GAM" / "tables" / "predictions.csv"
    pred = pd.read_csv(path)
    pred.loc[pred["model"] == "anom_ar1", "pred"] += 1.0
    pred.to_csv(path, index=False)
    with pytest.raises(ValueError, match="anom_ar1"):
        load_all(tmp_path)


def test_holm_known_answer():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

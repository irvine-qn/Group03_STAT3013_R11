"""Bảng Level 1 dựng lại bằng code khôi phục ở src/Data/ (từ commit db99215^).

Các test tổng hợp lấy từ tests/test_level1.py cũ (cùng commit), giữ nguyên đáp án biết trước.
Khác bản cũ: cấu hình vùng dựng từ configs/regions/mekong.yaml hiện tại cộng khóa `interim_series`
(khóa này đã bị bỏ khỏi YAML khi gỡ pipeline; RegionConfig vẫn bắt buộc có).
"""
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from level1 import build_level1
from regions import RegionConfig, repo_root
from schema import FLAG_COLUMNS, REQUIRED_COLUMNS, validate_long

ROOT = Path(__file__).resolve().parents[2]


def mekong_cfg() -> RegionConfig:
    raw = yaml.safe_load((ROOT / "configs" / "regions" / "mekong.yaml").read_text(encoding="utf-8"))
    raw.setdefault("interim_series", "data/interim/mekong/series_mean.csv")
    raw["known_missing"] = tuple(tuple(pair) for pair in raw.get("known_missing", ()))
    return RegionConfig(**raw)


def tiny_cfg(n_years=2):
    return dataclasses.replace(mekong_cfg(), first_year=2000, last_year=2000 + n_years - 1)


def synthetic_interim(cfg, drop=(), res=232.0, md5=None):
    rows = []
    for year in cfg.seasons:
        for step in range(1, cfg.n_steps + 1):
            if (year, step) in drop:
                continue
            rows.append({
                "year": year, "doy": cfg.doy_of(step), "res": res,
                "file_md5": f"{year}-{step}" if md5 is None else md5(year, step),
                cfg.target_column: 100.0 + 10.0 * step + year,
            })
    return pd.DataFrame(rows)


def test_repo_root_is_the_project_root():
    assert repo_root() == ROOT


def test_output_has_full_calendar_and_required_columns():
    cfg = tiny_cfg()
    out = build_level1(synthetic_interim(cfg), cfg)
    assert len(out) == 2 * 27
    assert list(out.columns[: len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS
    assert set(FLAG_COLUMNS) <= set(out.columns)
    assert (out["region"] == "mekong").all()
    assert not out["imputed"].any()


def test_missing_image_is_linearly_interpolated_and_flagged():
    cfg = tiny_cfg(1)
    out = build_level1(synthetic_interim(cfg, drop={(2000, 10)}), cfg)
    row = out[out["step"] == 10].iloc[0]
    # giá trị thật tuyến tính theo bước, nên nội suy phải ra đúng: 100 + 10·10 + 2000
    assert row["target_km2"] == pytest.approx(100.0 + 100.0 + 2000)
    assert bool(row["imputed"]) is True
    assert out["imputed"].sum() == 1


def test_missing_image_at_season_edge_is_an_error():
    cfg = tiny_cfg(1)
    with pytest.raises(ValueError):
        build_level1(synthetic_interim(cfg, drop={(2000, 1)}), cfg)


def test_only_later_copy_of_identical_image_is_flagged_duplicate():
    cfg = tiny_cfg(1)
    interim = synthetic_interim(cfg, md5=lambda y, s: "same" if s in (5, 6) else f"{y}-{s}")
    out = build_level1(interim, cfg).set_index("step")
    assert not out.loc[5, "duplicate"]
    assert out.loc[6, "duplicate"]
    assert out["duplicate"].sum() == 1


def test_coarse_grid_flag_follows_resolution_threshold():
    cfg = tiny_cfg(1)
    assert not build_level1(synthetic_interim(cfg, res=232.0), cfg)["grid_500m"].any()
    assert build_level1(synthetic_interim(cfg, res=500.0), cfg)["grid_500m"].all()


def test_image_outside_configured_seasons_is_an_error():
    cfg = tiny_cfg(1)
    with pytest.raises(ValueError):
        build_level1(synthetic_interim(dataclasses.replace(cfg, last_year=2001)), cfg)


def test_validate_long_rejects_nan_target_and_repeated_keys():
    cfg = tiny_cfg(1)
    good = build_level1(synthetic_interim(cfg), cfg)
    validate_long(good, cfg)
    nan = good.copy()
    nan.loc[3, "target_km2"] = np.nan
    with pytest.raises(ValueError):
        validate_long(nan, cfg)
    with pytest.raises(ValueError):
        validate_long(pd.concat([good, good.iloc[[0]]], ignore_index=True), cfg)


def test_shared_level1_table_passes_the_schema():
    """Bảng chung hiện có (data/processed/level1_series.csv) đúng schema và đủ 21 × 27 dòng."""
    cfg = mekong_cfg()
    out = pd.read_csv(ROOT / cfg.processed_level1)
    validate_long(out, cfg)
    assert sorted(out[out["imputed"]][["season", "doy"]].values.tolist()) == [[2013, 233], [2019, 257]]
    assert int(out["duplicate"].sum()) == 15


@pytest.mark.skipif(not (ROOT / "data" / "interim" / "mekong" / "series_mean.csv").exists(),
                    reason="bảng trung gian series_mean.csv không có trong repo (cần ảnh gốc để dựng)")
def test_rebuild_from_interim_matches_shared_table():
    cfg = mekong_cfg()
    rebuilt = build_level1(pd.read_csv(ROOT / cfg.interim_series), cfg)
    shared = pd.read_csv(ROOT / cfg.processed_level1)
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), shared, check_dtype=False)

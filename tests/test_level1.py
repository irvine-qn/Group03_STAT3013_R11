import dataclasses

import numpy as np
import pandas as pd
import pytest

from floodbench.data.level1 import build_level1
from floodbench.data.regions import load_region, repo_root
from floodbench.data.schema import FLAG_COLUMNS, REQUIRED_COLUMNS, validate_long


def tiny_cfg(n_years=2):
    cfg = load_region("mekong")
    return dataclasses.replace(cfg, first_year=2000, last_year=2000 + n_years - 1)


def synthetic_interim(cfg, drop=(), res=232.0, md5=None):
    rows = []
    for year in cfg.seasons:
        for step in range(1, cfg.n_steps + 1):
            if (year, step) in drop:
                continue
            rows.append(
                {
                    "year": year,
                    "doy": cfg.doy_of(step),
                    "res": res,
                    "file_md5": f"{year}-{step}" if md5 is None else md5(year, step),
                    cfg.target_column: 100.0 + 10.0 * step + year,
                }
            )
    return pd.DataFrame(rows)


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
    # truth is linear in step, so interpolation must recover it exactly
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
    fine = build_level1(synthetic_interim(cfg, res=232.0), cfg)
    coarse = build_level1(synthetic_interim(cfg, res=500.0), cfg)
    assert not fine["grid_500m"].any()
    assert coarse["grid_500m"].all()


def test_image_outside_configured_seasons_is_an_error():
    cfg = tiny_cfg(1)
    interim = synthetic_interim(dataclasses.replace(cfg, last_year=2001))
    with pytest.raises(ValueError):
        build_level1(interim, cfg)


def test_validate_long_rejects_nan_target_and_repeated_keys():
    cfg = tiny_cfg(1)
    good = build_level1(synthetic_interim(cfg), cfg)
    validate_long(good, cfg)

    nan = good.copy()
    nan.loc[3, "target_km2"] = np.nan
    with pytest.raises(ValueError):
        validate_long(nan, cfg)

    repeated = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_long(repeated, cfg)


def test_real_mekong_level1_matches_known_facts():
    path = repo_root() / "data" / "interim" / "mekong" / "series_mean.csv"
    cfg = load_region("mekong")
    out = build_level1(pd.read_csv(path), cfg)
    validate_long(out, cfg)
    assert len(out) == 21 * 27
    imputed = out[out["imputed"]][["season", "doy"]].values.tolist()
    assert sorted(imputed) == [[2013, 233], [2019, 257]]
    assert out["target_km2"].notna().all()
    assert out["duplicate"].sum() == 15

import pandas as pd
import pytest

from floodbench.validation.sensitivity import VARIANTS, make_variants, sensitivity_table


def synthetic_level1():
    peaks = {2000: 10.0, 2001: 20.0, 2002: 30.0, 2003: 40.0}
    rows = []
    for season, peak in peaks.items():
        for step in (1, 2, 3):
            rows.append(
                {
                    "season": season,
                    "step": step,
                    "target_km2": peak if step == 2 else 1.0,
                    "imputed": False,
                    "duplicate": False,
                    "grid_500m": False,
                }
            )
    df = pd.DataFrame(rows)
    df.loc[(df.season == 2001) & (df.step == 2), "duplicate"] = True  # the peak of 2001 is a copy
    df.loc[(df.season == 2003) & (df.step == 2), "imputed"] = True    # the peak of 2003 is interpolated
    df.loc[df.season == 2002, "grid_500m"] = True                      # 2002 is on the coarse grid
    return df


def test_variants_drop_exactly_what_they_name():
    variants = make_variants(synthetic_level1())
    assert set(variants) == set(VARIANTS)
    assert len(variants["default"]) == 12
    assert not variants["drop_imputed"]["imputed"].any() and len(variants["drop_imputed"]) == 11
    assert not variants["drop_suspected_copies"]["duplicate"].any()
    assert sorted(variants["only_232m_seasons"]["season"].unique()) == [2000, 2001, 2003]


def test_sensitivity_table_has_one_row_per_variant_with_known_peak_slope():
    level1 = synthetic_level1()
    gauge = pd.DataFrame({"season": [2000, 2001, 2002, 2003], "Max. WL (m)": [1.0, 2.0, 3.0, 4.0]})
    table = sensitivity_table(level1, gauge, "Max. WL (m)").set_index("variant")
    assert list(table.index) == list(VARIANTS)
    # default peaks 10,20,30,40 over years 2000..2003 -> slope 10 per year
    assert table.loc["default", "peak_slope_km2_per_year"] == pytest.approx(10.0)
    assert table.loc["default", "n_seasons"] == 4
    assert table.loc["only_232m_seasons", "n_seasons"] == 3

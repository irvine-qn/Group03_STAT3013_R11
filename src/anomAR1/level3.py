"""
Bước thực hiện: 3
- tạo bảng level 3 (mỗi mùa/dòng)
- chạy lệnh: python src/anomAR1/process/level3.py
"""

from pathlib import Path
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "dataset" / "processed"
LAND_USE = {
    1: "aquaculture",
    2: "shrimp_rice",
    5: "other",
    6: "rice_single",
    7: "rice_triple",
    8: "rice_double_dry",
    9: "rice_double_wet",
}


def build_level3(level1: pd.DataFrame, interim: pd.DataFrame) -> pd.DataFrame:
    peak_rows = level1.loc[level1.groupby("season")["target_km2"].idxmax()]

    table = peak_rows[["season", "target_km2", "doy"]].rename(
        columns={"target_km2": "peak_km2", "doy": "peak_doy"}
    )

    table = table.merge(
        level1.groupby("season", as_index=False)["target_km2"]
        .mean()
        .rename(columns={"target_km2": "mean_km2"}),
        on="season",
    )

    # lấy dòng đầu của mỗi năm (do diện tích đất ko đổi)
    land = interim.groupby("year", as_index=False).first()
    land = land[["year"] + [f"lu{c}_km2" for c in LAND_USE]]
    land.columns = ["season"] + [f"{name}_km2" for name in LAND_USE.values()]
    return table.merge(land, on="season", validate="one_to_one")


if __name__ == "__main__":
    out = build_level3(
        pd.read_csv(PROCESSED / "level1_series.csv"),
        pd.read_csv(PROCESSED / "series_mean.csv"),
    )
    out.to_csv(PROCESSED / "level3_yearly.csv", index=False)
    print(out.round(0).to_string(index=False))

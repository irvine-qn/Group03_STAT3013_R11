"""Kiểm chứng chuỗi diện tích ngập (suy ra từ ảnh) bằng mực nước thực đo.

Số liệu mực nước chỉ có theo năm, nên phép kiểm chứng ở cấp mùa lũ, không kiểm được
động lực theo bước 8 ngày. Vì cả diện tích lẫn mực nước đều giảm theo năm, tương quan thô
bị thổi phồng bởi xu hướng chung; phải báo thêm tương quan sau khi bỏ xu hướng.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GAUGE_SHEETS = {"TanChau": "WL_TanChau", "MyThuan": "WL_MyThuan"}


def season_summary(level1: pd.DataFrame) -> pd.DataFrame:
    """Đỉnh và trung bình diện tích ngập (km²) của mỗi mùa."""
    return (
        level1.groupby("season")["target_km2"]
        .agg(peak_km2="max", mean_km2="mean")
        .reset_index()
    )


def read_gauge_levels(zip_path: Path, sheet: str) -> pd.DataFrame:
    """Đọc một sheet mực nước theo năm (cột Year -> season) từ zip thủy văn Mê Kông."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        frame = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=sheet)
    frame = frame.dropna(subset=["Year"]).rename(columns={"Year": "season"})
    frame["season"] = frame["season"].astype(int)
    return frame


def _detrend(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(years, values, 1)
    return values - (slope * years + intercept)


def _fisher_ci(r: float, n: int, z_crit: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy 95% xấp xỉ cho hệ số tương quan (cần n > 3)."""
    if n <= 3 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se)))


def compare_to_gauge(area: pd.DataFrame, gauge: pd.DataFrame, area_col: str, gauge_col: str) -> dict:
    """Tương quan giữa một chỉ số diện tích theo mùa và một chỉ số mực nước theo năm.

    r_levels: tương quan thô · r_detrended: sau khi mỗi chuỗi bỏ xu hướng tuyến tính theo năm ·
    r_first_diff: tương quan của chênh lệch giữa hai năm liên tiếp.
    """
    both = area[["season", area_col]].merge(gauge[["season", gauge_col]], on="season").dropna()
    both = both.sort_values("season")
    years = both["season"].to_numpy(dtype=float)
    x, y = both[area_col].to_numpy(), both[gauge_col].to_numpy()

    r_detrended = float(stats.pearsonr(_detrend(x, years), _detrend(y, years))[0])
    consecutive = np.diff(years) == 1
    dx, dy = np.diff(x)[consecutive], np.diff(y)[consecutive]
    ci = _fisher_ci(r_detrended, len(both))
    return {
        "n": len(both),
        "first_season": int(years[0]),
        "last_season": int(years[-1]),
        "r_levels": float(stats.pearsonr(x, y)[0]),
        "rho_levels": float(stats.spearmanr(x, y)[0]),
        "r_detrended": r_detrended,
        "r_detrended_ci_low": ci[0],
        "r_detrended_ci_high": ci[1],
        "r_first_diff": float(stats.pearsonr(dx, dy)[0]) if len(dx) > 2 else float("nan"),
    }


PAIRS = [
    ("peak_km2", "Max. WL (m)"),
    ("peak_km2", "High-flow WL (m)"),
    ("mean_km2", "High-flow WL (m)"),
    ("mean_km2", "Mean WL (m)"),
]


def main() -> None:
    """Ghi results/tables/waterlevel_validation.csv: python -m floodbench.validation.waterlevel."""
    from floodbench.data.regions import load_region, repo_root

    cfg = load_region("mekong")
    area = season_summary(pd.read_csv(cfg.path("processed_level1")))
    zip_path = repo_root() / "data" / "raw" / "mekong_hydrology" / "fm996rrfyz-3.zip"
    rows = []
    for station, sheet in GAUGE_SHEETS.items():
        gauge = read_gauge_levels(zip_path, sheet)
        for area_col, gauge_col in PAIRS:
            rows.append({"station": station, "area_metric": area_col, "gauge_metric": gauge_col,
                         **compare_to_gauge(area, gauge, area_col, gauge_col)})
    out = pd.DataFrame(rows).round(3)
    dst = repo_root() / "results" / "tables" / "waterlevel_validation.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

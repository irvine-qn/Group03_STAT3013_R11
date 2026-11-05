"""Độ nhạy ở cấp dữ liệu: các kết luận mô tả có đổi khi loại ảnh đáng ngờ không?

Đây mới là kiểm tra trên thống kê mô tả (xu hướng đỉnh lũ, tương quan với mực nước).
Độ nhạy của kết quả mô hình sẽ chạy sau khi có khung đánh giá.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from floodbench.validation.waterlevel import compare_to_gauge, season_summary

VARIANTS = ("default", "drop_imputed", "drop_suspected_copies", "only_232m_seasons")


def make_variants(level1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Các phiên bản của bảng Level 1: mặc định, bỏ hàng nội suy, bỏ bản sao nghi vấn,
    hoặc chỉ giữ các mùa không có ảnh nào ở lưới 500 m."""
    coarse_seasons = level1.loc[level1["grid_500m"], "season"].unique()
    return {
        "default": level1,
        "drop_imputed": level1[~level1["imputed"]],
        "drop_suspected_copies": level1[~level1["duplicate"]],
        "only_232m_seasons": level1[~level1["season"].isin(coarse_seasons)],
    }


def sensitivity_table(level1: pd.DataFrame, gauge: pd.DataFrame, gauge_col: str) -> pd.DataFrame:
    rows = []
    for name, frame in make_variants(level1).items():
        peaks = season_summary(frame)
        slope = np.polyfit(peaks["season"], peaks["peak_km2"], 1)[0]
        vs_gauge = compare_to_gauge(peaks, gauge, "peak_km2", gauge_col)
        rows.append(
            {
                "variant": name,
                "n_seasons": len(peaks),
                "peak_slope_km2_per_year": slope,
                "n_overlap_with_gauge": vs_gauge["n"],
                "r_levels": vs_gauge["r_levels"],
                "r_detrended": vs_gauge["r_detrended"],
                "r_first_diff": vs_gauge["r_first_diff"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Ghi results/tables/data_sensitivity.csv: python -m floodbench.validation.sensitivity."""
    from floodbench.data.regions import load_region, repo_root
    from floodbench.validation.waterlevel import read_gauge_levels

    cfg = load_region("mekong")
    level1 = pd.read_csv(cfg.path("processed_level1"))
    zip_path = repo_root() / "data" / "raw" / "mekong_hydrology" / "fm996rrfyz-3.zip"
    gauge = read_gauge_levels(zip_path, "WL_TanChau")
    out = sensitivity_table(level1, gauge, "Max. WL (m)").round(3)
    dst = repo_root() / "results" / "tables" / "data_sensitivity.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

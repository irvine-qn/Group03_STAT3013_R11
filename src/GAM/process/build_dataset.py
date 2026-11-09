"""BƯỚC 1 – Dựng bảng dữ liệu đầu vào của GAM.

Chạy từ gốc repo:  python src/GAM/process/build_dataset.py

Đọc  : data/processed/mekong/level1_series.csv (bảng Level 1 chung của nhóm, chỉ đọc)
Ghi  : src/GAM/dataset/gam_rows.csv          mọi dòng dự báo của mùa 2003–2020, 3 tầm
       src/GAM/dataset/dataset_summary.csv   vài con số kiểm tra nhanh

Bảng gam_rows.csv chỉ để XEM (mở bằng Excel cũng được): mỗi dòng là một lần dự báo, đủ cột
đầu vào (doy_target, a_t, a_t1) và mục tiêu (anomaly_target). Bước 2 KHÔNG đọc file này mà tự
dựng lại dòng bên trong từng fold, để climatology chỉ dùng các mùa trước (không rò rỉ).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import DATASET, DAYS, HORIZONS, ensure_dirs, load_data
from floodbench.features import build_rows  # dựng dòng dự báo; climatology chỉ từ mùa trước


def gam_rows(Y: np.ndarray, cfg) -> pd.DataFrame:
    """Mọi dòng dự báo của các mùa có đủ 3 mùa trước (2003–2020), cho từng tầm h."""
    seasons_idx = range(3, Y.shape[0])  # mùa 2000–2002 chưa đủ mùa trước để tính climatology
    frames = []
    for h in HORIZONS:
        rows = build_rows(Y, cfg, seasons_idx, h)
        frames.append(pd.DataFrame({
            "h_steps": h,
            "h_days": DAYS[h],
            "season": rows["season"],
            "t": rows["t"],                                   # bước gốc (1..27)
            "doy_target": rows["doy_target"],                 # ngày trong năm của bước đích
            "y_t": rows["y_t"], "c_t": rows["c_t"],
            "a_t": rows["y_t"] - rows["c_t"],                 # anomaly hôm nay
            "a_t1": rows["y_t1"] - rows["c_t1"],              # anomaly 8 ngày trước
            "c_target": rows["c_target"],                     # mức bình thường ở bước đích
            "y_target": rows["y_target"],                     # giá trị thật ở bước đích
            "anomaly_target": rows["y_target"] - rows["c_target"],  # mục tiêu GAM học
        }))
    return pd.concat(frames, ignore_index=True)


def summary(table: pd.DataFrame, Y: np.ndarray) -> pd.DataFrame:
    out = []
    for h, g in table.groupby("h_steps"):
        out.append({
            "h_days": DAYS[h],
            "n_rows": len(g),
            "n_seasons": g["season"].nunique(),
            "rows_per_season": len(g) // g["season"].nunique(),
            "mean_y_target_km2": g["y_target"].mean(),
            "mean_anomaly_target_km2": g["anomaly_target"].mean(),
            "sd_anomaly_target_km2": g["anomaly_target"].std(),
            "corr_a_t_vs_anomaly_target": np.corrcoef(g["a_t"], g["anomaly_target"])[0, 1],
        })
    return pd.DataFrame(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    cfg, Y = load_data()
    table = gam_rows(Y, cfg)
    table.to_csv(DATASET / "gam_rows.csv", index=False)
    summ = summary(table, Y)
    summ.to_csv(DATASET / "dataset_summary.csv", index=False)
    print(f"Y: {Y.shape[0]} mùa x {Y.shape[1]} bước; {len(table)} dòng -> {DATASET / 'gam_rows.csv'}")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

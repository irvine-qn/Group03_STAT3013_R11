"""Bước 0 – Đọc dữ liệu: chuỗi diện tích ngập ĐBSCL thành ma trận (mùa × bước).

Nguồn: data/processed/level1_series.csv, mỗi dòng là một ảnh MODIS đã gộp thành km² ngập
(Vu et al. 2022). Mỗi mùa lũ có 27 ảnh, cách nhau 8 ngày, từ ngày 153 đến ngày 361.
Mọi thông số riêng của vùng (năm đầu, số bước, đường dẫn) đọc từ configs/regions/mekong.yaml,
không viết cứng trong code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# File này nằm ở <repo>/src/Chronos-2/, nên lùi 2 cấp là thư mục gốc repo
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(region: str = "mekong") -> dict:
    """Đọc cấu hình vùng từ configs/regions/<region>.yaml."""
    path = REPO_ROOT / "configs" / "regions" / f"{region}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def step_to_doy(config: dict, step: int) -> int:
    """Đổi số thứ tự bước (1..27) sang ngày trong năm. Bước 1 là ngày 153."""
    return config["first_doy"] + config["step_days"] * (step - 1)


def load_flood_matrix(config: dict, csv_path: Path | None = None) -> np.ndarray:
    """Trả ma trận Y, Y[i, s] = km² ngập của mùa (first_year + i) tại bước s + 1.

    Báo lỗi nếu thiếu bất kỳ ô nào, để không âm thầm dự báo trên giá trị NaN.
    """
    csv_path = csv_path or REPO_ROOT / config["processed_level1"]
    table = pd.read_csv(csv_path)

    seasons = range(config["first_year"], config["last_year"] + 1)
    steps = range(1, config["n_steps"] + 1)
    wide = table.pivot(index="season", columns="step", values="target_km2")
    wide = wide.reindex(index=seasons, columns=steps)

    is_missing = wide.isna().stack()  # một giá trị True/False cho mỗi cặp (mùa, bước)
    missing = is_missing[is_missing].index.tolist()
    if missing:
        raise ValueError(f"missing data (season, step): {missing[:5]}")
    return wide.to_numpy(dtype=float)

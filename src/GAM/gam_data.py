"""Đọc dữ liệu cho GAM: cấu hình vùng và chuỗi diện tích ngập thành ma trận (mùa × bước).

Nguồn: data/processed/level1_series.csv, mỗi dòng là một ảnh MODIS đã gộp thành km² ngập
(Vu et al. 2022). Mỗi mùa lũ có 27 ảnh, cách nhau 8 ngày, từ ngày 153 đến ngày 361.
Mọi thông số của vùng (năm đầu, số bước, đường dẫn) đọc từ configs/regions/<vùng>.yaml,
không viết cứng trong code. Cùng cách làm với src/Chronos-2/chronos2_data.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# File này nằm ở <repo>/src/GAM/, nên lùi 2 cấp là thư mục gốc repo
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Region:
    """Những thông số của vùng mà GAM cần, lấy từ file YAML."""
    name: str
    first_year: int     # mùa đầu tiên (2000)
    last_year: int      # mùa cuối cùng (2020)
    first_doy: int      # ngày trong năm của bước 1 (153)
    step_days: int      # số ngày giữa hai bước (8)
    n_steps: int        # số bước mỗi mùa (27)
    level1_path: Path   # bảng Level 1 (đường dẫn tuyệt đối)

    @property
    def seasons(self) -> range:
        return range(self.first_year, self.last_year + 1)

    def doy_of(self, step: int) -> int:
        """Đổi số thứ tự bước (1..27) sang ngày trong năm. Bước 1 là ngày 153."""
        return self.first_doy + self.step_days * (step - 1)


def load_region(region: str = "mekong") -> Region:
    """Đọc cấu hình vùng từ configs/regions/<region>.yaml."""
    path = REPO_ROOT / "configs" / "regions" / f"{region}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Region(name=cfg["region"], first_year=cfg["first_year"], last_year=cfg["last_year"],
                  first_doy=cfg["first_doy"], step_days=cfg["step_days"], n_steps=cfg["n_steps"],
                  level1_path=REPO_ROOT / cfg["processed_level1"])


def load_flood_matrix(region: Region) -> np.ndarray:
    """Ma trận Y, Y[i, s] = km² ngập của mùa (first_year + i) tại bước s + 1.

    Báo lỗi nếu thiếu bất kỳ ô nào, để không âm thầm dự báo trên giá trị NaN.
    """
    table = pd.read_csv(region.level1_path)
    wide = table.pivot(index="season", columns="step", values="target_km2")
    wide = wide.reindex(index=list(region.seasons), columns=range(1, region.n_steps + 1))
    missing = wide.isna().stack()
    if missing.any():
        raise ValueError(f"thiếu dữ liệu (mùa, bước): {missing[missing].index.tolist()[:5]}")
    return wide.to_numpy(dtype=float)

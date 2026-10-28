"""Cấu hình theo vùng: đọc `configs/regions/<region>.yaml`.

Code trong `src/` không được hard-code hằng số của một vùng; mọi thứ riêng của vùng
(đường dẫn, lịch ảnh, cột mục tiêu) nằm ở file YAML này.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RegionConfig:
    region: str
    display_name: str
    raw_dir: str
    interim_series: str
    processed_level1: str
    first_year: int
    last_year: int
    first_doy: int
    step_days: int
    n_steps: int
    target_column: str
    resolution_column: str
    coarse_grid_threshold_m: float
    known_missing: tuple[tuple[int, int], ...] = ()

    @property
    def seasons(self) -> range:
        return range(self.first_year, self.last_year + 1)

    def doy_of(self, step: int) -> int:
        """Ngày trong năm của bước `step` (đánh số từ 1)."""
        return self.first_doy + self.step_days * (step - 1)

    def step_of(self, doy: int) -> int:
        """Bước (1..n_steps) ứng với ngày `doy`; báo lỗi nếu ngày không nằm trên lịch ảnh."""
        offset = doy - self.first_doy
        step = offset // self.step_days + 1
        if offset < 0 or offset % self.step_days or step > self.n_steps:
            raise ValueError(f"doy {doy} is not on the {self.region} image calendar")
        return step

    def path(self, attr: str) -> Path:
        return repo_root() / getattr(self, attr)


def load_region(name: str, config_dir: Path | None = None) -> RegionConfig:
    config_dir = config_dir or repo_root() / "configs" / "regions"
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no region config at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["known_missing"] = tuple(tuple(pair) for pair in raw.get("known_missing", ()))
    return RegionConfig(**raw)

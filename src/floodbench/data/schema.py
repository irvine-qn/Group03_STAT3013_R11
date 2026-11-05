"""Schema chung cho mọi vùng: một dòng = một ảnh (mùa, bước)."""
from __future__ import annotations

import pandas as pd

from floodbench.data.regions import RegionConfig

REQUIRED_COLUMNS = ["region", "season", "step", "doy", "target_km2"]
# imputed: ảnh không có, giá trị nội suy · duplicate: bản sao nghi vấn của ảnh trước đó
# grid_500m: ảnh gốc ở lưới 500 m (thay vì 232 m)
FLAG_COLUMNS = ["imputed", "duplicate", "grid_500m"]


def validate_long(df: pd.DataFrame, cfg: RegionConfig) -> None:
    """Báo ValueError nếu bảng không đúng schema hoặc không đủ lịch ảnh của vùng."""
    missing = [c for c in REQUIRED_COLUMNS + FLAG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["target_km2"].isna().any():
        raise ValueError("target_km2 contains NaN")
    if df.duplicated(["season", "step"]).any():
        raise ValueError("repeated (season, step) keys")
    expected = len(cfg.seasons) * cfg.n_steps
    if len(df) != expected:
        raise ValueError(f"expected {expected} rows, got {len(df)}")

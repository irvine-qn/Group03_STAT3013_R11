"""Bảng Level 1: một chuỗi diện tích ngập toàn vùng, đủ lịch ảnh, kèm cờ chất lượng.

Đầu vào là bảng trung gian của `build_series` (một dòng mỗi ảnh có thật), đầu ra theo
schema chung (`schema.REQUIRED_COLUMNS` + `schema.FLAG_COLUMNS`): đủ mọi (mùa, bước),
ảnh thiếu được nội suy tuyến tính trong mùa và gắn cờ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from floodbench.data.regions import RegionConfig
from floodbench.data.schema import FLAG_COLUMNS, REQUIRED_COLUMNS


def build_level1(interim: pd.DataFrame, cfg: RegionConfig) -> pd.DataFrame:
    df = interim.copy()
    outside = ~df["year"].isin(cfg.seasons)
    if outside.any():
        raise ValueError(f"images outside seasons {cfg.first_year}-{cfg.last_year}: {sorted(df.loc[outside, 'year'].unique())}")
    df["season"] = df["year"].astype(int)
    df["step"] = df["doy"].map(cfg.step_of)  # ValueError nếu ngày không nằm trên lịch ảnh
    df = df.sort_values(["season", "step"]).reset_index(drop=True)

    # Ảnh y hệt từng byte với một ảnh sớm hơn: giữ bản đầu, nghi bản sau là bản sao
    df["duplicate"] = df.duplicated("file_md5", keep="first")
    df["grid_500m"] = df[cfg.resolution_column] > cfg.coarse_grid_threshold_m

    calendar = pd.MultiIndex.from_product([cfg.seasons, range(1, cfg.n_steps + 1)], names=["season", "step"])
    keep = ["season", "step", cfg.target_column, "duplicate", "grid_500m"]
    full = df[keep].set_index(["season", "step"]).reindex(calendar).reset_index()

    target = full[cfg.target_column]
    full["imputed"] = target.isna()
    for season, grp in full.groupby("season"):
        if grp["imputed"].iloc[[0, -1]].any():
            raise ValueError(f"season {season}: first or last image missing, cannot interpolate")
        known = ~grp["imputed"]
        full.loc[grp.index, cfg.target_column] = np.interp(grp["step"], grp.loc[known, "step"], grp.loc[known, cfg.target_column])

    # Hàng nội suy không có ảnh thật nên không thể là bản sao hay thuộc lưới nào: NaN -> False
    full["duplicate"] = full["duplicate"].eq(True)
    full["grid_500m"] = full["grid_500m"].eq(True)
    full["region"] = cfg.region
    full["doy"] = full["step"].map(cfg.doy_of)
    full["target_km2"] = full[cfg.target_column].astype(float)
    return full[REQUIRED_COLUMNS + FLAG_COLUMNS]


def main(region: str = "mekong") -> None:
    """Dựng và ghi bảng Level 1 của một vùng: python -m floodbench.data.level1 [region]."""
    from floodbench.data.regions import load_region
    from floodbench.data.schema import validate_long

    cfg = load_region(region)
    out = build_level1(pd.read_csv(cfg.path("interim_series")), cfg)
    validate_long(out, cfg)
    dst = cfg.path("processed_level1")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"{region}: {len(out)} rows, {out['imputed'].sum()} imputed, "
          f"{out['duplicate'].sum()} suspected copies -> {dst}")


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:2])

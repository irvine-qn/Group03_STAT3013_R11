"""
Thứ tự chạy: 1
Chạy thử lại dataset để kiểm tra
"""
import pandas as pd
from __future__ import annotations
from pathlib import Path
import re
import sys
from floodbench.data.build_series import main as build_series
from floodbench.data.level1 import build_level1
from floodbench.data.regions import load_region, repo_root
from floodbench.data.schema import validate_long

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "dataset" / "raw"
PROCESSED = HERE / "dataset" / "processed"
RESULT = HERE / "result"


def audit_raw(raw: Path, cfg) -> dict:
    """Đếm file raw và tìm các ảnh thiếu so với lịch ảnh của vùng."""
    files = [p for p in (raw / "Flooding").iterdir() if p.suffix in (".tif", ".tiff")]
    found = set()
    for p in files:
        m = re.search(r"Flood_(\d{4})_\s*(\d{3})", p.name)
        found.add((int(m.group(1)), int(m.group(2))))
    expected = {(y, cfg.doy_of(s)) for y in cfg.seasons for s in range(1, cfg.n_steps + 1)}
    
    return {
        "flood_images": len(files),
        "tif": sum(p.suffix == ".tif" for p in files),
        "tiff_envi": sum(p.suffix == ".tiff" for p in files),
        "landuse_images": len(list((raw / "Land use").glob("Landuse_*.tif"))),
        "expected_images": len(expected),
        "missing": " ".join(f"{y}-{d}" for y, d in sorted(expected - found)),
        "off_calendar": " ".join(f"{y}-{d}" for y, d in sorted(found - expected)),
        "names_with_space": " ".join(p.name for p in files if " " in p.name),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_region("mekong")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)

    audit = audit_raw(RAW, cfg)
    print(audit)

    interim_path = PROCESSED / "series_mean.csv"
    build_series(str(RAW), str(interim_path))
    interim = pd.read_csv(interim_path)
    level1 = build_level1(interim, cfg)
    validate_long(level1, cfg)
    level1.to_csv(PROCESSED / "level1_series.csv", index=False)

    # Đối chiếu với bản đang dùng của nhóm (chỉ đọc)
    team_interim = pd.read_csv(repo_root() / "data" / "interim" / "mekong" / "series_mean.csv")
    team_level1 = pd.read_csv(cfg.path("processed_level1"))
    keys_i, keys_l = ["year", "doy"], ["season", "step"]
    a = interim.merge(team_interim, on=keys_i, suffixes=("", "_team"), how="outer", indicator=True)
    b = level1.merge(team_level1, on=keys_l, suffixes=("", "_team"), how="outer", indicator=True)
    check = {
        **audit,
        "level1_rows": len(level1),
        "imputed": int(level1["imputed"].sum()),
        "duplicate": int(level1["duplicate"].sum()),
        "grid_500m": int(level1["grid_500m"].sum()),
        "interim_unmatched": int((a["_merge"] != "both").sum()),
        "interim_md5_differs": int((a["file_md5"] != a["file_md5_team"]).sum()),
        "interim_water_max_abs_diff_km2": float((a["water_km2"] - a["water_km2_team"]).abs().max()),
        "level1_unmatched": int((b["_merge"] != "both").sum()),
        "level1_target_max_abs_diff_km2": float((b["target_km2"] - b["target_km2_team"]).abs().max()),
        "level1_flags_differ": int(sum((b[c] != b[f"{c}_team"]).sum() for c in ("imputed", "duplicate", "grid_500m"))),
    }
    pd.Series(check, name="value").rename_axis("check").to_csv(RESULT / "dataset_check.csv")
    print(pd.Series(check).to_string())


if __name__ == "__main__":
    main()

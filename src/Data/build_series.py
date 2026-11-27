"""Dựng lại chuỗi diện tích ngập ĐBSCL từ ảnh gốc Mendeley, quy lưới bằng mean.

Chạy: python -m floodbench.data.build_series <thư mục kpftzmsyyz-2> <output.csv>

- Lưới chung: lưới 500 m của Landuse_2000 (EPSG:32648).
- Mặt nạ ĐBSCL: hợp các pixel != 0 của 21 ảnh sử dụng đất.
- Ảnh lũ (232 m hoặc 500 m) -> tỷ lệ nước / hỗn hợp trên mỗi ô 500 m bằng
  Resampling.average trên lớp chỉ thị 0/1 (thay cho nearest của script cũ).
- Ghi thêm md5 từng file để phát hiện ảnh chép lại (trong năm và giữa các năm).
"""
import csv
import glob
import hashlib
import os
import re
import sys

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

LU_CLASSES = (1, 2, 5, 6, 7, 8, 9)
CELL_KM2 = 0.25


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_ref(src_arr, src_profile, ref, resampling):
    out = np.zeros(ref["shape"], dtype="float32")
    reproject(
        src_arr, out,
        src_transform=src_profile.transform, src_crs=src_profile.crs,
        dst_transform=ref["transform"], dst_crs=ref["crs"],
        resampling=resampling,
    )
    return out


def main(root, dst):
    lu_files = sorted(glob.glob(os.path.join(root, "Land use", "Landuse_*.tif")))
    with rasterio.open(lu_files[0]) as d:
        ref = {"shape": d.shape, "transform": d.transform, "crs": d.crs}

    landuse = {}
    for f in lu_files:
        year = int(re.search(r"(\d{4})", os.path.basename(f)).group(1))
        with rasterio.open(f) as d:
            landuse[year] = to_ref(d.read(1), d, ref, Resampling.nearest).astype("uint8")
    mask = np.any(np.stack([a != 0 for a in landuse.values()]), axis=0)
    print(f"mặt nạ ĐBSCL = {mask.sum() * CELL_KM2:.2f} km²")

    flood_files = sorted(
        glob.glob(os.path.join(root, "Flooding", "Flood_*.tif"))
        + glob.glob(os.path.join(root, "Flooding", "Flood_*.tiff"))
    )
    rows = []
    for f in flood_files:
        m = re.search(r"Flood_(\d{4})_\s*(\d{3})", os.path.basename(f))
        year, doy = int(m.group(1)), int(m.group(2))
        with rasterio.open(f) as d:
            a = d.read(1)
            res = d.transform.a
            water = to_ref((a == 1).astype("float32"), d, ref, Resampling.average)
            mixed = to_ref((a == 2).astype("float32"), d, ref, Resampling.average)
        lu = landuse[year]
        row = {
            "year": year, "doy": doy, "res": round(res, 4), "file_md5": md5(f),
            "vmd_km2": mask.sum() * CELL_KM2,
            "water_km2": round(float(water[mask].sum()) * CELL_KM2, 2),
            "mixed_km2": round(float(mixed[mask].sum()) * CELL_KM2, 2),
        }
        row["dry_km2"] = round(row["vmd_km2"] - row["water_km2"] - row["mixed_km2"], 2)
        for c in LU_CLASSES:
            sel = mask & (lu == c)
            row[f"lu{c}_km2"] = sel.sum() * CELL_KM2
            row[f"lu{c}_water"] = round(float(water[sel].sum()) * CELL_KM2, 2)
        rows.append(row)
        print(year, doy, row["water_km2"])

    rows.sort(key=lambda r: (r["year"], r["doy"]))
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} dòng -> {dst}")


if __name__ == "__main__":
    # Console Windows (cp1258/cp1252) không in được tiếng Việt có dấu
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2])

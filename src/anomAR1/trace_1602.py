"""
Bước 6
- truy nguyên chênh lệch AnomAR1 32 ngày: mean/nearest x bước gốc đầu tiên (t = 3 hoặc 1).
- Chạy: python src/anomAR1/process/trace_1602.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from floodbench.data.regions import repo_root

RESULT = Path(__file__).resolve().parents[1] / "result"
HORIZONS = (1, 2, 4)
OLD = {
    "persistence": [1299, 2003, 3326],
    "clim5": [1767, 1787, 1844],
    "anom_ar1": [1089, 1312, 1602],
}


def load_wide(path: Path) -> np.ndarray:
    """Ma trận mùa x bước; ảnh thiếu được nội suy tuyến tính trong mùa."""
    table = pd.read_csv(path).pivot(index="year", columns="doy", values="water_km2")
    return table.reindex(columns=range(153, 362, 8)).interpolate(axis=1).to_numpy()


def rmse_table(Y: np.ndarray, first_t: int) -> dict[str, list[float]]:
    """Huấn luyện như bản hiện tại (mùa 2003 trở đi); chỉ đổi tập cặp dự báo được đánh giá."""
    clim = lambda i: Y[max(0, i - 5) : i].mean(axis=0)
    out = {name: [] for name in OLD}
    for h in HORIZONS:
        errors = {name: [] for name in OLD}
        for test in range(10, 21):
            pairs = [
                (i, o) for i in range(3, test) for o in range(2, 27 - h)
            ]  # o = chỉ số gốc 0-based
            now = np.array([Y[i, o] - clim(i)[o] for i, o in pairs])
            prev = np.array([Y[i, o - 1] - clim(i)[o - 1] for i, o in pairs])
            phi = (now * prev).sum() / (prev**2).sum()
            c = clim(test)
            origins = np.arange(first_t - 1, 27 - h)
            target = Y[test, origins + h]
            errors["persistence"].append(target - Y[test, origins])
            errors["clim5"].append(target - c[origins + h])
            errors["anom_ar1"].append(
                target - c[origins + h] - phi**h * (Y[test, origins] - c[origins])
            )
        for name, err in errors.items():
            out[name].append(float(np.sqrt(np.mean(np.concatenate(err) ** 2))))
    return out


def main() -> None:
    interim = repo_root() / "data" / "interim" / "mekong"
    series = {
        "mean": interim / "series_mean.csv",
        "nearest": interim / "legacy" / "series_nearest_v1.csv",
    }
    rows = [
        {
            "series": "old_report",
            "first_t": "",
            "model": name,
            **dict(zip(HORIZONS, values)),
        }
        for name, values in OLD.items()
    ]
    for label, path in series.items():
        Y = load_wide(path)
        for first_t in (3, 1):
            for name, values in rmse_table(Y, first_t).items():
                rows.append(
                    {
                        "series": label,
                        "first_t": first_t,
                        "model": name,
                        **dict(zip(HORIZONS, values)),
                    }
                )
    out = pd.DataFrame(rows).round(1)
    out.to_csv(RESULT / "trace_1602.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

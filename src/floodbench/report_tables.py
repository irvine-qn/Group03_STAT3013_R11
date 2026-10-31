"""Sinh các bảng tóm tắt từ results/tables/probe_predictions.csv."""
from __future__ import annotations

import pandas as pd

from floodbench.data.regions import repo_root
from floodbench.evaluation import pooled_rmse, season_mse
from floodbench.inference import compare_to_baseline

BASE = "anom_ar1"
PHASES = [(0, 257, "rising"), (257, 298, "peak"), (298, 999, "falling")]


def main() -> None:
    d = repo_root() / "results" / "tables"
    pred = pd.read_csv(d / "probe_predictions.csv")
    main_pred = pred[~pred["model"].str.endswith("_w7")]
    w7 = pred[pred["model"].str.endswith("_w7")]

    rmse = pooled_rmse(main_pred).pivot(index="model", columns="h", values="rmse").round(0)
    rmse.to_csv(d / "probe_rmse.csv")
    print("RMSE km2 (cột = tầm h):\n", rmse.to_string(), "\n")

    tests = compare_to_baseline(season_mse(main_pred), BASE).round(4)
    tests.to_csv(d / "probe_vs_anom_ar1.csv", index=False)
    print("So với anom_ar1 (cấp mùa):\n", tests.drop(columns=["n_seasons"]).to_string(index=False), "\n")

    h2 = main_pred[main_pred["h"] == 2].copy()
    h2["phase"] = pd.cut(h2["doy_target"], [a for a, _, _ in PHASES] + [999], right=False,
                         labels=[n for _, _, n in PHASES])
    phase = pooled_rmse(h2, by=("model", "phase")).pivot(index="model", columns="phase", values="rmse").round(0)
    phase.to_csv(d / "probe_by_phase_h2.csv")
    print("Theo giai đoạn lũ, h=2:\n", phase.to_string(), "\n")

    peaks = main_pred[main_pred["model"] == "clim_all"]  # chỉ để lấy y theo mùa
    peak = peaks.groupby("season")["y_target"].max()
    size = pd.qcut(peak, 3, labels=["small", "medium", "large"])
    h2["size"] = h2["season"].map(size)
    mag = pooled_rmse(h2, by=("model", "size")).pivot(index="model", columns="size", values="rmse").round(0)
    mag.to_csv(d / "probe_by_magnitude_h2.csv")
    print("Theo cỡ lũ của mùa test, h=2:\n", mag.to_string(), "\n")

    both = pd.concat([w7, main_pred[main_pred["model"].isin(["gam", "lgbm_raw", "anom_ar1"])]])
    win = pooled_rmse(both).pivot(index="model", columns="h", values="rmse").round(0)
    win.to_csv(d / "probe_window7.csv")
    print("Cửa sổ huấn luyện cố định 7 mùa (_w7) vs toàn bộ:\n", win.to_string())


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()

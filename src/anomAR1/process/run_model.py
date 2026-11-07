"""
- tạo bảng hồi quy, huấn luyện, có sử dụng file dataset level1_series.csv
- Lệnh chạy: python src/anomAR1/process/run_model.py
- hàm floodbench.models.baselines.anom_ar1 sinh ra dự báo
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from floodbench.cv import rolling_origin
from floodbench.data.regions import load_region, repo_root
from floodbench.features import build_rows, to_wide
from floodbench.models.baselines import anom_ar1

HERE = Path(__file__).resolve().parents[1]
LEVEL1 = HERE / "dataset" / "processed" / "level1_series.csv"
PROCESSED = HERE / "dataset" / "processed"
RESULT = HERE / "result"
horizons = (1, 2, 4)
WINDOW = 7


# Hồi quy của phi: x = anomaly bước trước, y = anomaly hiện tại.
def regression_xy(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = (rows["y_t1"] - rows["c_t1"]).to_numpy()
    y = (rows["y_t"] - rows["c_t"]).to_numpy()
    return x, y


# Hồi quy qua gốc: phi = sum(x*y) / sum(x^2).
def estimate_phi(rows: pd.DataFrame) -> float:
    x, y = regression_xy(rows)
    return float((x * y).sum() / (x**2).sum())


def regression_table(Y: np.ndarray, cfg) -> pd.DataFrame:
    idx = range(3, len(cfg.seasons))
    frames = []
    for h in horizons:
        rows = build_rows(Y, cfg, idx, h)
        x, y = regression_xy(rows)
        frames.append(
            rows[
                [
                    "season",
                    "t",
                    "y_t",
                    "y_t1",
                    "c_t",
                    "c_t1",
                    "c_target",
                    "doy_target",
                    "y_target",
                ]
            ].assign(h=h, x_anomaly_prev=x, y_anomaly_now=y)
        )
    return pd.concat(frames, ignore_index=True)


def run(Y: np.ndarray, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = lambda seasons: [s - cfg.first_year for s in seasons]
    preds, phis = [], []
    for window, name in ((None, "anom_ar1"), (WINDOW, "anom_ar1_w7")):
        for fold in rolling_origin(cfg, window=window):
            for h in horizons:
                train = build_rows(Y, cfg, idx(fold.train_seasons), h)
                test = build_rows(Y, cfg, idx([fold.test_season]), h)
                pred = anom_ar1(train, test, h)
                phi = estimate_phi(train)
                x, y = regression_xy(train)
                ols = sm.OLS(y, x).fit()
                phis.append(
                    {
                        "model": name,
                        "test_season": fold.test_season,
                        "h": h,
                        "first_train": fold.train_seasons[0],
                        "last_train": fold.train_seasons[-1],
                        "n_train_seasons": len(fold.train_seasons),
                        "n_train_rows": len(train),
                        "phi": phi,
                        "phi_statsmodels": float(ols.params[0]),
                        "phi_se": float(ols.bse[0]),
                        "phi_pow_h": phi**h,
                    }
                )
                preds.append(
                    test[["season", "t", "doy_target", "y_target"]].assign(
                        model=name, h=h, pred=pred
                    )
                )
    return pd.concat(preds, ignore_index=True), pd.DataFrame(phis)


def rmse_by(pred: pd.DataFrame, by: list[str]) -> pd.Series:
    return (
        ((pred["y_target"] - pred["pred"]) ** 2)
        .groupby([pred[c] for c in by])
        .mean()
        .pow(0.5)
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_region("mekong")
    Y = to_wide(pd.read_csv(LEVEL1), cfg)
    RESULT.mkdir(parents=True, exist_ok=True)

    regression_table(Y, cfg).to_csv(PROCESSED / "regression_rows.csv", index=False)
    pred, phi = run(Y, cfg)
    pred.to_csv(RESULT / "predictions.csv", index=False)
    phi.to_csv(RESULT / "phi_by_fold.csv", index=False)

    rmse = rmse_by(pred, ["model", "h"]).rename("rmse_km2").reset_index()
    team = pd.read_csv(repo_root() / "results" / "tables" / "probe_predictions.csv")
    team = team[team["model"].isin(pred["model"].unique())]

    m = pred.merge(
        team,
        on=["model", "h", "season", "t"],
        suffixes=("", "_team"),
        how="outer",
        indicator=True,
    )

    diff = (
        m.assign(abs_diff=(m["pred"] - m["pred_team"]).abs())
        .groupby(["model", "h"])["abs_diff"]
        .max()
    )

    rmse = rmse.merge(
        diff.rename("max_abs_diff_vs_team_km2").reset_index(), on=["model", "h"]
    )
    rmse.to_csv(RESULT / "rmse.csv", index=False)

    print("rows", len(pred), "unmatched vs team", int((m["_merge"] != "both").sum()))
    print(rmse.round(6).to_string(index=False))
    print(
        "max |phi - statsmodels|",
        float((phi["phi"] - phi["phi_statsmodels"]).abs().max()),
    )
    print(
        phi[phi["model"] == "anom_ar1"]
        .pivot(index="test_season", columns="h", values="phi")
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()

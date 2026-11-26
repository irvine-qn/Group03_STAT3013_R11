"""
Bước: 4
- dựa vào file predictions.csv để phân tích
- lệnh: python src/anomAR1/process/analysis.py
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from floodbench.cv import rolling_origin
from floodbench.data.regions import load_region
from floodbench.features import build_rows, to_wide

root = Path(__file__).resolve().parents[1]
RESULT = root / "result"
FIGURES = RESULT / "figures"
horizons = (1, 2, 4)
days = {1: 8, 2: 16, 4: 32}
edges = [0, 257, 298, 999]
phases = ["rising", "peak", "falling"]
sizes = ["small", "medium", "large"]


def rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(((frame["y_target"] - frame["pred"]) ** 2).mean()))


def direct_phi(train: pd.DataFrame) -> float:
    # hồi quy qua gốc của anomaly đích theo anomaly hiện tại, riêng cho một tầm
    x = (train["y_t"] - train["c_t"]).to_numpy()
    y = (train["y_target"] - train["c_target"]).to_numpy()
    return float((x * y).sum() / (x**2).sum())


def compare_direct_phi(Y, cfg, pred: pd.DataFrame) -> pd.DataFrame:
    # phi ước lượng riêng từng tầm so với phi^h. Không thay định nghĩa chính
    idx = lambda seasons: [s - cfg.first_year for s in seasons]
    rows = []
    for fold in rolling_origin(cfg):
        for h in horizons:
            train = build_rows(Y, cfg, idx(fold.train_seasons), h)
            test = build_rows(Y, cfg, idx([fold.test_season]), h)
            phi = direct_phi(train)
            rows.append(
                test.assign(
                    h=h, pred=test["c_target"] + phi * (test["y_t"] - test["c_t"])
                )
            )
    direct = pd.concat(rows)
    return pd.DataFrame(
        [
            {
                "h": h,
                "days": days[h],
                "rmse_phi_pow_h": rmse(pred[pred.h == h]),
                "rmse_direct_phi": rmse(direct[direct.h == h]),
            }
            for h in horizons
        ]
    )


def condition_tables(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    size = pd.qcut(pred.groupby("season")["y_target"].max(), 3, labels=sizes)
    framed = pred.assign(
        phase=pd.cut(pred["doy_target"], edges, right=False, labels=phases),
        size=pred["season"].map(size),
    )

    phase = (
        framed.groupby(["h", "phase"], observed=True)
        .apply(rmse, include_groups=False)
        .unstack()
    )

    magnitude = (
        framed.groupby(["h", "size"], observed=True)
        .apply(rmse, include_groups=False)
        .unstack()
    )
    return phase, magnitude


def residual_tests(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, season), grp in pred.groupby(["h", "season"]):
        err = (
            grp.sort_values("t")["y_target"] - grp.sort_values("t")["pred"]
        ).to_numpy()
        r = acf(err, nlags=5, fft=False)
        rows.append(
            {
                "h": h,
                "season": season,
                "bias_km2": err.mean(),
                **{f"acf_{k}": r[k] for k in range(1, 6)},
                "ljung_box_p": float(
                    acorr_ljungbox(err, lags=[10])["lb_pvalue"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_diagnostics(pred, tests, phi) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, h in enumerate(horizons):
        p = pred[pred.h == h]
        sm.qqplot(p["y_target"] - p["pred"], line="s", ax=axes[0, col], markersize=3)
        axes[0, col].set_title(f"Q-Q plot of errors, {days[h]} days")
        mean_acf = tests[tests.h == h][[f"acf_{k}" for k in range(1, 6)]].mean()
        axes[1, col].bar(range(1, 6), mean_acf)
        axes[1, col].axhline(0, color="black", linewidth=0.8)
        axes[1, col].set(
            title=f"Mean within-season ACF, {days[h]} days",
            xlabel="Lag (steps)",
            ylim=(-0.5, 1),
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_diagnostics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in horizons:
        p = phi[(phi.model == "anom_ar1") & (phi.h == h)]
        ax.plot(
            p["test_season"], p["phi"], marker="o", label=f"Rows of {days[h]}-day horizon"
        )
    ax.set(xlabel="Test season", ylabel="phi", title="Estimated phi by fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phi_by_fold.png", dpi=150)
    plt.close(fig)


def plot_seasons(pred, seasons) -> None:
    fig, axes = plt.subplots(len(seasons), 3, figsize=(13, 6), sharey="row")
    for row, season in enumerate(seasons):
        for col, h in enumerate(horizons):
            p = pred[(pred.season == season) & (pred.h == h)].sort_values("doy_target")
            axes[row, col].plot(
                p["doy_target"], p["y_target"], color="black", label="Observed"
            )
            axes[row, col].plot(
                p["doy_target"], p["pred"], color="tab:blue", label="Forecast"
            )
            axes[row, col].set(
                title=f"{season}, {days[h]} days", xlabel="Day of year", ylabel="km²"
            )
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "typical_seasons.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cfg = load_region("mekong")
    Y = to_wide(pd.read_csv(root / "dataset" / "processed" / "level1_series.csv"), cfg)
    all_pred = pd.read_csv(RESULT / "predictions.csv")
    pred = all_pred[all_pred["model"] == "anom_ar1"]
    phi = pd.read_csv(RESULT / "phi_by_fold.csv")

    direct = compare_direct_phi(Y, cfg, pred)
    phase, magnitude = condition_tables(pred)
    tests = residual_tests(pred)
    summary = tests.groupby("h").agg(
        mean_bias_km2=("bias_km2", "mean"),
        mean_acf_1=("acf_1", "mean"),
        seasons_lb_p_below_05=("ljung_box_p", lambda p: int((p < 0.05).sum())),
        n_seasons=("season", "size"),
    )
    direct.to_csv(RESULT / "phi_direct_sensitivity.csv", index=False)
    phase.to_csv(RESULT / "rmse_by_phase.csv")
    magnitude.to_csv(RESULT / "rmse_by_magnitude.csv")
    tests.to_csv(RESULT / "residual_tests.csv", index=False)
    summary.to_csv(RESULT / "residual_summary.csv")

    plot_diagnostics(pred, tests, phi)
    peaks = pred.groupby("season")["y_target"].max()
    plot_seasons(pred, [int(peaks.idxmax()), int(peaks.idxmin())])

    for title, table in (
        ("phi^h vs direct", direct),
        ("phase", phase),
        ("magnitude", magnitude),
        ("residuals", summary),
    ):
        print(title, "\n", table.round(3).to_string(), "\n")
    print(
        "largest/smallest test season by observed peak:",
        int(peaks.idxmax()),
        int(peaks.idxmin()),
    )


if __name__ == "__main__":
    main()

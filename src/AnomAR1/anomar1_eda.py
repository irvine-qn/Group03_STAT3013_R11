"""EDA – Thống kê mô tả 21 mùa lũ và kiểm định xu hướng đỉnh lũ (độc lập với dự báo).

Chạy:  python src/AnomAR1/anomar1_eda.py
Đọc:  data/processed/level1_series.csv
Ghi:  results/AnomAR1/tables/eda_descriptive.csv, eda_inference.csv
      results/AnomAR1/figures/seasons_overlay.png, heatmap_season_step.png,
                              distribution_boxplots.png, peak_trend.png
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from anomar1_data import FIGURES, TABLES, load_config, load_level1

PHASE_EDGES = [0, 257, 298, 999]
PHASES = ["rising", "peak", "falling"]
SPLIT_YEAR = 2010  # mùa test đầu tiên: so đỉnh lũ trước và từ năm này


def season_peaks(level1: pd.DataFrame) -> pd.DataFrame:
    """Một dòng mỗi mùa: đỉnh diện tích ngập, ngày đạt đỉnh, diện tích trung bình."""
    peak_rows = level1.loc[level1.groupby("season")["target_km2"].idxmax()]
    table = peak_rows[["season", "target_km2", "doy"]].rename(
        columns={"target_km2": "peak_km2", "doy": "peak_doy"})
    means = level1.groupby("season", as_index=False)["target_km2"].mean()
    return table.merge(means.rename(columns={"target_km2": "mean_km2"}), on="season").reset_index(drop=True)


def describe(level1: pd.DataFrame) -> pd.DataFrame:
    """Thống kê mô tả diện tích ngập: toàn bộ và theo giai đoạn lũ."""
    phase = pd.cut(level1["doy"], PHASE_EDGES, right=False, labels=PHASES).astype(str)
    groups = [("all", level1["target_km2"])] + [(p, level1.loc[phase == p, "target_km2"]) for p in PHASES]
    return pd.DataFrame([
        {"group": name, "n": len(v), "mean": v.mean(), "std": v.std(), "min": v.min(),
         "median": v.median(), "max": v.max(), "skewness": stats.skew(v)}
        for name, v in groups
    ])


def inference(peaks: pd.DataFrame) -> pd.DataFrame:
    """Kiểm định trên 21 mùa: xu hướng đỉnh lũ, đổi mức trước/sau 2010, ngày đạt đỉnh theo cỡ lũ."""
    year, peak = peaks["season"], peaks["peak_km2"]
    ols = sm.OLS(peak, sm.add_constant(year)).fit(cov_type="HC3")
    low, high = ols.conf_int().loc["season"]
    tau = stats.kendalltau(year, peak)
    before, after = peak[year < SPLIT_YEAR], peak[year >= SPLIT_YEAR]
    mw = stats.mannwhitneyu(before, after, alternative="two-sided")
    lev = stats.levene(before, after)
    # Nhóm chia theo đỉnh nên không kiểm định lại đỉnh (sẽ là vòng tròn); kiểm định ngày đạt đỉnh
    size = pd.qcut(peak, 3, labels=["small", "medium", "large"])
    kw = stats.kruskal(*[peaks.loc[size == s, "peak_doy"] for s in size.cat.categories])
    return pd.DataFrame([
        {"test": "Mann-Kendall (peak ~ year)", "statistic": tau.statistic, "p": tau.pvalue, "note": "tau"},
        {"test": "OLS HC3 (peak ~ year)", "statistic": ols.params["season"], "p": ols.pvalues["season"],
         "note": f"km2/year, CI95 [{low:.0f}; {high:.0f}], R2={ols.rsquared:.3f}"},
        {"test": "Mann-Whitney (peak before vs from 2010)", "statistic": mw.statistic, "p": mw.pvalue,
         "note": f"median {before.median():.0f} vs {after.median():.0f}, n={len(before)}/{len(after)}"},
        {"test": "Levene (peak before vs from 2010)", "statistic": lev.statistic, "p": lev.pvalue,
         "note": f"std {before.std():.0f} vs {after.std():.0f}"},
        {"test": "Kruskal-Wallis (peak_doy by flood size)", "statistic": kw.statistic, "p": kw.pvalue,
         "note": "3 groups by peak tercile"},
    ])


def plot_all(level1: pd.DataFrame, peaks: pd.DataFrame) -> None:
    wide = level1.pivot(index="season", columns="doy", values="target_km2")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for season, row in wide.iterrows():
        ax.plot(row.index, row.values, color=plt.cm.viridis((season - 2000) / 20), linewidth=1)
    ax.figure.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(2000, 2020)),
                       ax=ax, label="Season")
    ax.set(xlabel="Day of year", ylabel="Flooded area (km²)", title="21 flood seasons")
    fig.tight_layout()
    fig.savefig(FIGURES / "seasons_overlay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(wide.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(0, 27, 4), wide.columns[::4])
    ax.set_yticks(range(len(wide)), wide.index)
    ax.set(xlabel="Day of year", ylabel="Season", title="Flooded area by season and step")
    fig.colorbar(image, ax=ax, label="km²")
    fig.tight_layout()
    fig.savefig(FIGURES / "heatmap_season_step.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 2.4, 1]})
    axes[0].hist(level1["target_km2"], bins=30, color="tab:blue")
    axes[0].set(xlabel="km²", ylabel="Number of images", title="Distribution of flooded area")
    axes[1].boxplot([g["target_km2"] for _, g in level1.groupby("season")], tick_labels=wide.index)
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set(ylabel="km²", title="By season")
    phase = pd.cut(level1["doy"], PHASE_EDGES, right=False, labels=PHASES)
    axes[2].boxplot([level1.loc[phase == p, "target_km2"] for p in PHASES], tick_labels=PHASES)
    axes[2].set(ylabel="km²", title="By flood phase")
    fig.tight_layout()
    fig.savefig(FIGURES / "distribution_boxplots.png", dpi=150)
    plt.close(fig)

    slope, intercept = np.polyfit(peaks["season"], peaks["peak_km2"], 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(peaks["season"], peaks["peak_km2"], marker="o", label="Season peak")
    ax.plot(peaks["season"], slope * peaks["season"] + intercept, linestyle="--", color="black",
            label=f"OLS: {slope:.0f} km²/year")
    ax.set(xlabel="Season", ylabel="km²", title="Peak flooded area by year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "peak_trend.png", dpi=150)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    level1 = load_level1(load_config("mekong"))
    peaks = season_peaks(level1)
    desc, tests = describe(level1), inference(peaks)
    desc.to_csv(TABLES / "eda_descriptive.csv", index=False)
    tests.to_csv(TABLES / "eda_inference.csv", index=False)
    plot_all(level1, peaks)
    print(desc.round(2).to_string(index=False), "\n")
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

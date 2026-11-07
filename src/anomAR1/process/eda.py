"""
Bước: 5
- thống kê mô tả và suy diễn.
- Lệnh: python src/anomAR1/process/eda.py
Note: nếu chạy khôgn theo thứ tự các bước thì phải đảm bảo file này chạy sau file level3.py nhen :v
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "dataset" / "processed"
RESULT = ROOT / "result"
FIGURES = RESULT / "figures"
PHASE_EDGES = [0, 257, 298, 999]
PHASES = ["rising", "peak", "falling"]
SPLIT_YEAR = 2010


def describe(level1: pd.DataFrame) -> pd.DataFrame:
    # ngập toàn bộ và theo giai đoạn lũ
    framed = level1.assign(
        phase=pd.cut(level1["doy"], PHASE_EDGES, right=False, labels=PHASES).astype(str)
    )
    groups = [("all", level1["target_km2"])] + [
        (p, framed.loc[framed.phase == p, "target_km2"]) for p in PHASES
    ]
    return pd.DataFrame(
        [
            {
                "group": name,
                "n": len(v),
                "mean": v.mean(),
                "std": v.std(),
                "min": v.min(),
                "median": v.median(),
                "max": v.max(),
                "skewness": stats.skew(v),
            }
            for name, v in groups
        ]
    )


def inference(level3: pd.DataFrame) -> pd.DataFrame:
    # kiểm định trên 21 mùa
    year, peak = level3["season"], level3["peak_km2"]
    ols = sm.OLS(peak, sm.add_constant(year)).fit(cov_type="HC3")
    low, high = ols.conf_int().loc["season"]
    tau = stats.kendalltau(year, peak)
    before, after = peak[year < SPLIT_YEAR], peak[year >= SPLIT_YEAR]
    mw = stats.mannwhitneyu(before, after, alternative="two-sided")
    lev = stats.levene(before, after)
    
    # Nhóm chia theo đỉnh nên không kiểm định lại đỉnh (sẽ là vòng tròn); kiểm định ngày đạt đỉnh
    size = pd.qcut(peak, 3, labels=["small", "medium", "large"])
    kw = stats.kruskal(
        *[level3.loc[size == s, "peak_doy"] for s in size.cat.categories]
    )
    return pd.DataFrame(
        [
            {
                "test": "Mann-Kendall (peak ~ year)",
                "statistic": tau.statistic,
                "p": tau.pvalue,
                "note": "tau",
            },
            {
                "test": "OLS HC3 (peak ~ year)",
                "statistic": ols.params["season"],
                "p": ols.pvalues["season"],
                "note": f"km2/year, CI95 [{low:.0f}; {high:.0f}], R2={ols.rsquared:.3f}",
            },
            {
                "test": "Mann-Whitney (peak before vs from 2010)",
                "statistic": mw.statistic,
                "p": mw.pvalue,
                "note": f"median {before.median():.0f} vs {after.median():.0f}, n={len(before)}/{len(after)}",
            },
            {
                "test": "Levene (peak before vs from 2010)",
                "statistic": lev.statistic,
                "p": lev.pvalue,
                "note": f"std {before.std():.0f} vs {after.std():.0f}",
            },
            {
                "test": "Kruskal-Wallis (peak_doy by flood size)",
                "statistic": kw.statistic,
                "p": kw.pvalue,
                "note": "3 groups by peak tercile",
            },
        ]
    )


def plot_all(level1: pd.DataFrame, level3: pd.DataFrame) -> None:
    wide = level1.pivot(index="season", columns="doy", values="target_km2")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for season, row in wide.iterrows():
        ax.plot(
            row.index,
            row.values,
            color=plt.cm.viridis((season - 2000) / 20),
            linewidth=1,
        )
    ax.figure.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(2000, 2020)),
        ax=ax,
        label="Mùa",
    )
    ax.set(xlabel="Ngày trong năm", ylabel="Diện tích ngập (km²)", title="21 mùa lũ")
    fig.tight_layout()
    fig.savefig(FIGURES / "seasons_overlay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(wide.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(0, 27, 4), wide.columns[::4])
    ax.set_yticks(range(21), wide.index)
    ax.set(
        xlabel="Ngày trong năm", ylabel="Mùa", title="Diện tích ngập theo mùa và bước"
    )
    fig.colorbar(image, ax=ax, label="km²")
    fig.tight_layout()
    fig.savefig(FIGURES / "heatmap_season_step.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 2.4, 1]}
    )
    axes[0].hist(level1["target_km2"], bins=30, color="tab:blue")
    axes[0].set(xlabel="km²", ylabel="Số ảnh", title="Phân phối diện tích ngập")
    axes[1].boxplot(
        [g["target_km2"] for _, g in level1.groupby("season")], tick_labels=wide.index
    )
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set(ylabel="km²", title="Theo mùa")
    phase = pd.cut(level1["doy"], PHASE_EDGES, right=False, labels=PHASES)
    axes[2].boxplot(
        [level1.loc[phase == p, "target_km2"] for p in PHASES], tick_labels=PHASES
    )
    axes[2].set(ylabel="km²", title="Theo giai đoạn lũ")
    fig.tight_layout()
    fig.savefig(FIGURES / "distribution_boxplots.png", dpi=150)
    plt.close(fig)

    slope, intercept = np.polyfit(level3["season"], level3["peak_km2"], 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(level3["season"], level3["peak_km2"], marker="o", label="Đỉnh mùa")
    ax.plot(
        level3["season"],
        slope * level3["season"] + intercept,
        linestyle="--",
        color="black",
        label=f"OLS: {slope:.0f} km²/năm",
    )
    ax.set(xlabel="Mùa", ylabel="km²", title="Đỉnh diện tích ngập theo năm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "peak_trend.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    level1 = pd.read_csv(PROCESSED / "level1_series.csv")
    level3 = pd.read_csv(PROCESSED / "level3_yearly.csv")
    desc, tests = describe(level1), inference(level3)
    desc.to_csv(RESULT / "eda_descriptive.csv", index=False)
    tests.to_csv(RESULT / "eda_inference.csv", index=False)
    plot_all(level1, level3)
    print(desc.round(2).to_string(index=False), "\n")
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

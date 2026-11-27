"""Bước 2 – Hình so sánh chung của 4 bản chính (chỉ đọc bảng của bước 1 và predictions.csv).

Chạy (sau comparison_tables.py):  python src/Comparison/comparison_figures.py
Ghi:  results/Comparison/figures/
        rmse_by_horizon.png    RMSE gộp theo tầm, 4 bản chính và 2 mốc đơn giản
        season_ratio.png       tỉ lệ RMSE so với AnomAR1 của từng mùa test, mỗi tầm
        conditions.png         RMSE theo giai đoạn lũ và cỡ lũ, mỗi tầm
        example_seasons.png    một mùa lũ lớn và một mùa lũ nhỏ: quan sát và dự báo 16 ngày của 4 bản chính

Màu và nét theo thiết kế video "flood" (giống app Streamlit): trắng, xanh trời, đen.
Mỗi mô hình có cả màu lẫn kiểu nét riêng để vẫn phân biệt được khi in đen trắng.
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comparison_tables import (BASELINE, CONTENDERS, FIGURES, HORIZON_DAYS, HORIZONS, MAIN, PHASES, SIZES,
                               TABLES, load_all)

INK, MUTED, RULE, PAPER = "#111111", "#6b7280", "#d9dde3", "#ffffff"
SKY_DEEP, SKY_DARK, GREY = "#3fa9e8", "#1d5f8a", "#9aa1ab"
STYLE = {  # màu, kiểu nét, ký hiệu
    "anom_ar1": (GREY, "-", "o"),
    "gam": (SKY_DEEP, "-", "s"),
    "lgbm_raw": (SKY_DARK, "--", "^"),
    "chronos2": (MUTED, ":", "D"),
    "clim5": (RULE, "-.", "x"),
    "persistence": (RULE, (0, (1, 1)), "+"),
}
LABEL = {"anom_ar1": "AnomAR1", "gam": "GAM", "lgbm_raw": "LightGBM thô", "chronos2": "Chronos-2",
         "clim5": "Clim5", "persistence": "Persistence"}
PHASE_VI = {"rising": "Nước lên", "peak": "Đỉnh lũ", "falling": "Nước rút"}
SIZE_VI = {"small": "Lũ nhỏ", "medium": "Lũ vừa", "large": "Lũ lớn"}

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": INK, "axes.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": RULE, "grid.linewidth": 0.8, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "font.size": 10,
})


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print("  ->", name)


def fig_rmse_by_horizon(rmse: pd.DataFrame) -> None:
    table = rmse.set_index("model")
    days = [HORIZON_DAYS[h] for h in HORIZONS]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model in MAIN + ["clim5", "persistence"]:
        color, ls, marker = STYLE[model]
        width = 2.6 if model in MAIN else 1.4
        ax.plot(days, table.loc[model, [str(h) for h in HORIZONS]], color=color, ls=ls, marker=marker,
                lw=width, label=LABEL[model])
    ax.set(xticks=days, xlabel="Tầm dự báo (ngày)", ylabel="RMSE gộp (km²)",
           title="RMSE gộp, 11 mùa test 2010–2020")
    ax.legend(ncols=2, fontsize=9)
    save(fig, "rmse_by_horizon.png")


def fig_season_ratio(seasons: pd.DataFrame, tests: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(13, 4.2), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        ticks = []
        for k, model in enumerate(CONTENDERS):
            color, _, marker = STYLE[model]
            part = seasons[(seasons["h"] == h) & (seasons["model"] == model)]
            x = np.full(len(part), k) + np.linspace(-0.18, 0.18, len(part))
            ax.scatter(x, part["ratio_vs_anom_ar1"], color=color, marker=marker, s=30, zorder=3)
            row = tests[(tests["model"] == model) & (tests["h"] == h)].iloc[0]
            ax.hlines(row["rmse_ratio"], k - 0.3, k + 0.3, color=INK, lw=2.5, zorder=4)
            ticks.append(f"{LABEL[model]}\nthắng {int(row['wins'])}/11\np = {row['p_t_holm_main']:.3f}")
        ax.axhline(1, color=INK, lw=1, ls="--")
        ax.set(xticks=range(len(CONTENDERS)), title=f"Tầm {HORIZON_DAYS[h]} ngày")
        ax.set_xticklabels(ticks, fontsize=8.5)
    axes[0].set_ylabel("RMSE mô hình / RMSE AnomAR1\n(< 1: tốt hơn AnomAR1)")
    fig.suptitle("Mỗi điểm là một mùa test; vạch đen = tỉ lệ RMSE gộp. Dưới tên: số mùa thắng AnomAR1, p sau Holm toàn cục (9 phép so)",
                 fontsize=10, color=MUTED)
    save(fig, "season_ratio.png")


def fig_conditions(phase: pd.DataFrame, size: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(HORIZONS), figsize=(14, 7))
    width = 0.8 / len(MAIN)
    for col, h in enumerate(HORIZONS):
        for row, (table, groups, names) in enumerate([(phase, PHASES, PHASE_VI), (size, SIZES, SIZE_VI)]):
            ax = axes[row, col]
            sub = table[table["h"] == h].set_index("model")
            for k, model in enumerate(MAIN):
                color, _, _ = STYLE[model]
                hatch = {"lgbm_raw": "//", "chronos2": ".."}.get(model)
                ax.bar(np.arange(3) + (k - (len(MAIN) - 1) / 2) * width, sub.loc[model, groups], width,
                       color=color if model != "chronos2" else PAPER, edgecolor=INK if model == "chronos2" else PAPER,
                       hatch=hatch, label=LABEL[model])
            ax.set(xticks=range(3), title=f"Tầm {HORIZON_DAYS[h]} ngày")
            ax.set_xticklabels([names[g] for g in groups])
            if col == 0:
                ax.set_ylabel("RMSE (km²)")
    axes[0, 0].legend(fontsize=8, ncols=2)
    fig.suptitle("Hàng trên: theo giai đoạn lũ (ngày đích). Hàng dưới: theo cỡ lũ của mùa test (tam phân vị đỉnh). "
                 "Chỉ để mô tả, không kiểm định.", fontsize=10, color=MUTED)
    save(fig, "conditions.png")


def fig_example_seasons(pred: pd.DataFrame, seasons: pd.DataFrame, h: int = 2) -> None:
    peaks = seasons.drop_duplicates("season").set_index("season")["peak_km2"]
    chosen = [int(peaks.idxmax()), int(peaks.idxmin())]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, season in zip(axes, chosen):
        part = pred[(pred["season"] == season) & (pred["h"] == h)]
        obs = part.drop_duplicates("doy_target").sort_values("doy_target")
        ax.plot(obs["doy_target"], obs["y_target"], color=INK, lw=3.2, label="Quan sát")
        for model in MAIN:
            color, ls, marker = STYLE[model]
            m = part[part["model"] == model].sort_values("doy_target")
            ax.plot(m["doy_target"], m["pred"], color=color, ls=ls, marker=marker, ms=3.5, lw=1.8,
                    label=LABEL[model])
        kind = "đỉnh cao nhất" if season == chosen[0] else "đỉnh thấp nhất"
        ax.set(title=f"Mùa {season} ({kind} trong 11 mùa test)", xlabel="Ngày được dự báo (ngày trong năm)")
    axes[0].set_ylabel("Diện tích ngập (km²)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Dự báo trước {HORIZON_DAYS[h]} ngày", fontsize=10, color=MUTED)
    save(fig, "example_seasons.png")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rmse = pd.read_csv(TABLES / "rmse_all.csv")
    tests = pd.read_csv(TABLES / "vs_anom_ar1_all.csv")
    phase = pd.read_csv(TABLES / "rmse_by_phase.csv")
    size = pd.read_csv(TABLES / "rmse_by_size.csv")
    seasons = pd.read_csv(TABLES / "season_rmse.csv")
    pred = load_all()
    pred = pred[pred["model"].isin(MAIN)]
    fig_rmse_by_horizon(rmse)
    fig_season_ratio(seasons, tests)
    fig_conditions(phase, size)
    fig_example_seasons(pred, seasons)


if __name__ == "__main__":
    main()

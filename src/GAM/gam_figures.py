"""BƯỚC 4 – Vẽ toàn bộ hình cho báo cáo GAM (đọc bảng của bước 2 và 3).

Chạy từ gốc repo:  python src/GAM/gam_figures.py
Ghi vào results/GAM/figures/*.png.
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from gam_common import (BASELINE, DAYS, FIGURES, HORIZONS, PHASE_NAMES, PHASE_LABEL, SIZE_NAMES, SIZE_LABEL,
                        TABLES, load_data, season_index)
from gam_model import GAM_GRID, contributions, fit_gam
from gam_protocol import N_VALIDATION_SEASONS, build_rows, climatology, rolling_origin

# Màu cố định cho cả bộ hình: GAM luôn xanh, AnomAR1 luôn cam, giá trị thật luôn đen
C_GAM, C_AR1, C_OBS, C_CLIM, C_OTHER = "#2563eb", "#ea580c", "#111111", "#6b7280", "#a3a3a3"
NAMES = {"gam": "GAM", BASELINE: "AnomAR1", "gam_w7": "GAM (7 seasons)", "anom_ar1_w7": "AnomAR1 (7 seasons)",
         "clim5": "5-season climatology",
         "persistence": "Persistence", "gam_linear_doy": "GAM, linear day",
         "ar1_direct_intercept": "Only β0 + β1·a_t"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


# ---- Hình 1: dữ liệu ----------------------------------------------------------------------------
def fig_data(Y, cfg):
    doys = [cfg.doy_of(s) for s in range(1, cfg.n_steps + 1)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, season in enumerate(cfg.seasons):
        a.plot(doys, Y[i] / 1000, color=cmap(i / (len(cfg.seasons) - 1)), lw=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cfg.first_year, cfg.last_year))
    fig.colorbar(sm, ax=a, label="Season", format="%d")
    a.set(title="a) 21 flood seasons: one line per season", xlabel="Day of year",
          ylabel="Flooded area (thousand km²)")
    years = np.arange(cfg.first_year, cfg.last_year + 1)
    peaks = Y.max(axis=1) / 1000
    slope, icpt = np.polyfit(years, peaks, 1)
    b.plot(years, peaks, "o-", color=C_OBS)
    b.plot(years, icpt + slope * years, "--", color=C_AR1, label=f"Trend {slope * 1000:+.0f} km²/year")
    b.set(title="b) Seasonal flood peak is declining", xlabel="Season", ylabel="Peak (thousand km²)",
          xticks=years[::2])
    b.tick_params(axis="x", rotation=45)
    b.legend()
    save(fig, "fig01_data.png")


# ---- Hình 2: giao thức rolling-origin -----------------------------------------------------------
def fig_protocol(cfg):
    years = np.arange(cfg.first_year, cfg.last_year + 1)
    folds = list(rolling_origin(cfg))
    colors = {"clim": "#e5e7eb", "fit": "#93c5fd", "val": "#fcd34d", "test": "#f87171", "none": "white"}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for r, fold in enumerate(folds):
        val = fold.train_seasons[-N_VALIDATION_SEASONS:]
        for c, y in enumerate(years):
            kind = ("test" if y == fold.test_season else "val" if y in val else
                    "fit" if y in fold.train_seasons else "clim" if y < fold.test_season else "none")
            ax.add_patch(plt.Rectangle((c, r), 0.95, 0.9, color=colors[kind], ec="#d1d5db"))
    ax.set(xlim=(0, len(years)), ylim=(len(folds), 0), xticks=np.arange(len(years)) + 0.47,
           yticks=np.arange(len(folds)) + 0.45, xlabel="Season", ylabel="Fold (test season)",
           title="Yearly rolling origin: each row is one fit – forecast round")
    ax.set_xticklabels(years, rotation=60)
    ax.set_yticklabels([f.test_season for f in folds])
    ax.grid(False)
    ax.legend(handles=[Patch(color=colors["clim"], label="Used only for climatology"),
                       Patch(color=colors["fit"], label="Training during config selection"),
                       Patch(color=colors["val"], label="3 validation seasons (config selection)"),
                       Patch(color=colors["test"], label="Test season (never used for selection)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    save(fig, "fig02_protocol.png")


# ---- Hình 3: một dự báo được tạo ra như thế nào -------------------------------------------------
def fig_one_forecast(Y, cfg, test_season=2016, h=2, origin_step=11):
    configs = pd.read_csv(TABLES / "gam_chosen_configs.csv")
    row = configs.query("model == 'gam' and test_season == @test_season and h == @h").iloc[0]
    fold = next(f for f in rolling_origin(cfg) if f.test_season == test_season)
    train = build_rows(Y, cfg, season_index(cfg, fold.train_seasons), h)
    test = build_rows(Y, cfg, season_index(cfg, [test_season]), h)
    model = fit_gam(train, (int(row.n_splines), float(row.lam)))
    one = test[test["t"] == origin_step]
    parts = contributions(model, one).iloc[0]

    i = test_season - cfg.first_year
    doys = np.array([cfg.doy_of(s) for s in range(1, cfg.n_steps + 1)])
    clim = climatology(Y, i)
    t0, t1 = origin_step - 1, origin_step - 1 + h  # chỉ số cột (0-based) của gốc và đích

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for k in range(1, 6):
        a.plot(doys, Y[i - k] / 1000, color=C_OTHER, lw=0.7, alpha=0.6)
    a.plot(doys, clim / 1000, "--", color=C_CLIM, lw=2, label="Normal level c = mean of 5 prior seasons (light grey)")
    a.plot(doys[:t0 + 1], Y[i, :t0 + 1] / 1000, color=C_OBS, lw=2, label=f"Season {test_season}: known up to today")
    a.plot(doys[t0:], Y[i, t0:] / 1000, color=C_OBS, lw=1, ls=":", label="This season: future (unknown)")
    a.annotate("", xy=(doys[t0], Y[i, t0] / 1000), xytext=(doys[t0], clim[t0] / 1000),
               arrowprops=dict(arrowstyle="<->", color=C_AR1, lw=1.5))
    a.text(doys[t0] + 3, (Y[i, t0] + clim[t0]) / 2000 + 0.6, f"a_t = {Y[i, t0] - clim[t0]:+,.0f} km²",
           color=C_AR1, ha="left", va="center", bbox=dict(fc="white", ec="none", alpha=0.85))
    a.scatter([doys[t1]], [parts.pred / 1000], color=C_GAM, s=60, zorder=5, label=f"GAM forecast {DAYS[h]} days ahead")
    a.scatter([doys[t1]], [Y[i, t1] / 1000], color=C_OBS, marker="x", s=60, zorder=5, label="Actual value")
    a.axvline(doys[t0], color=C_CLIM, lw=0.8)
    a.text(doys[t0] + 1, a.get_ylim()[1] * 0.97, "today", color=C_CLIM, va="top")
    a.set(title="a) Data GAM sees when forecasting", xlabel="Day of year",
          ylabel="Flooded area (thousand km²)")
    a.legend(fontsize=8, loc="upper left")

    labels = ["c\n(normal)", "β0\n(intercept)", f"s(day\n{int(one.doy_target.iloc[0])})",
              "β1·a_t", "β2·a_t−1", "Forecast"]
    values = [parts.c_target, parts.beta0, parts.s_doy, parts.beta1_a_t, parts.beta2_a_t1]
    start = np.concatenate([[0], np.cumsum(values)[:-1]])
    box = dict(fc="white", ec="none", alpha=0.85, pad=1)
    for k, (v, s0) in enumerate(zip(values, start)):
        col = C_CLIM if k == 0 else ("#16a34a" if v >= 0 else "#dc2626")
        b.bar(k, v, bottom=s0, color=col, width=0.6)
        # số ghi ngay giữa đoạn của cột (cột nhỏ thì ghi phía trên) để không đè lên nhau
        y_txt = s0 + v / 2 if abs(v) > 400 else max(s0, s0 + v) + 120
        b.text(k, y_txt, f"{v:+,.0f}" if k else f"{v:,.0f}", ha="center", va="center", fontsize=9, bbox=box)
    b.bar(5, parts.pred, color=C_GAM, width=0.6)
    b.text(5, parts.pred / 2, f"{parts.pred:,.0f}", ha="center", va="center", fontsize=9, bbox=box)
    b.axhline(Y[i, t1], color=C_OBS, ls=":", lw=1)
    b.text(5.35, Y[i, t1] - 250, f"actual = {Y[i, t1]:,.0f}", va="top", ha="center", fontsize=8, bbox=box)
    b.set(xticks=range(6), ylim=(0, parts.c_target * 1.1), ylabel="km²",
          title=f"b) Forecast = sum of parts\n(fold {test_season}, config n_splines={int(row.n_splines)}, lam={row.lam:g})")
    b.set_xticklabels(labels, fontsize=8)
    save(fig, "fig03_one_forecast.png")


# ---- Hình 4: chọn cấu hình ----------------------------------------------------------------------
def fig_configs():
    cf = pd.read_csv(TABLES / "gam_chosen_configs.csv").query("model == 'gam'")
    cols = [f"val_mse_{n}_{lam}" for n, lam in GAM_GRID]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        g = cf[cf["h"] == h].set_index("test_season")[cols]
        rel = 100 * (g.div(g.min(axis=1), axis=0) - 1)  # % tệ hơn cấu hình tốt nhất của fold
        im = ax.imshow(rel.to_numpy(), cmap="Blues", vmin=0, vmax=15, aspect="auto")
        for r in range(rel.shape[0]):
            for c in range(rel.shape[1]):
                v = rel.iat[r, c]
                ax.text(c, r, "★" if v == 0 else f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="black" if v < 9 else "white")
        ax.set(xticks=range(len(cols)), yticks=range(rel.shape[0]), title=f"Horizon {DAYS[h]} days")
        ax.set_xticklabels([f"({n};{lam:g})" for n, lam in GAM_GRID], rotation=45, fontsize=8)
        ax.set_yticklabels(rel.index)
        ax.grid(False)
    axes[0].set_ylabel("Fold (test season)")
    fig.colorbar(im, ax=axes, label="Validation MSE worse than best config (%)", shrink=0.8)
    fig.suptitle("Config (n_splines; lam) chosen ★ in each fold: differences between configs are usually only a few %",
                 y=1.03)
    fig.savefig(FIGURES / "fig04_config_selection.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig04_config_selection.png")


# ---- Hình 5: hệ số trễ so với φ^h của AnomAR1 ----------------------------------------------------
def fig_coefficients(Y, cfg):
    co = pd.read_csv(TABLES / "gam_coefficients.csv").query("model == 'gam'")
    phis = []
    for fold in rolling_origin(cfg):  # φ của AnomAR1 ở từng fold (cùng công thức với baselines.anom_ar1)
        tr = build_rows(Y, cfg, season_index(cfg, fold.train_seasons), 1)
        x, y = (tr.y_t1 - tr.c_t1).to_numpy(), (tr.y_t - tr.c_t).to_numpy()
        phis.append((fold.test_season, (x * y).sum() / (x ** 2).sum()))
    phis = pd.DataFrame(phis, columns=["test_season", "phi"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, h in zip(axes, HORIZONS):
        g = co[co["h"] == h]
        ax.plot(g["test_season"], g["beta1_a_t"], "o-", color=C_GAM, label="GAM: β1 (a_t)")
        ax.plot(g["test_season"], g["beta2_a_t1"], "s-", color=C_GAM, alpha=0.45, label="GAM: β2 (a_t−1)")
        ax.plot(phis["test_season"], phis["phi"] ** h, "d--", color=C_AR1, label=f"AnomAR1: φ^{h}")
        ax.axhline(0, color="black", lw=0.6)
        ax.set(title=f"Horizon {DAYS[h]} days", xlabel="Fold (test season)", ylim=(-0.35, 1.05))
    axes[0].set_ylabel("Coefficient")
    axes[0].legend(fontsize=8)
    save(fig, "fig05_coefficients.png")


# ---- Hình 6: hàm trơn theo ngày = phần sửa lệch climatology --------------------------------------
def fig_seasonal_curve():
    cur = pd.read_csv(TABLES / "gam_seasonal_curve.csv").query("model == 'gam'")
    bias = pd.read_csv(TABLES / "climatology_bias.csv")
    mids = bias["doy_bin"].str.extract(r"\((\d+), (\d+)\]").astype(float).mean(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        for _, g in cur[cur["h"] == h].groupby("test_season"):
            ax.plot(g["doy"], g["beta0_plus_s"], color=C_GAM, lw=0.9, alpha=0.5)
        ax.plot([], [], color=C_GAM, label="β0 + s(day), one line per fold")
        ax.bar(mids, bias["mean_anomaly_km2"], width=24, color=C_AR1, alpha=0.25,
               label="Mean data anomaly\n(how far climatology is off)")
        ax.axhline(0, color="black", lw=0.7)
        ax.set(title=f"Horizon {DAYS[h]} days", xlabel="Target day of year")
    axes[0].set_ylabel("Forecast anomaly when a_t = a_t−1 = 0 (km²)")
    axes[0].legend(fontsize=8, loc="lower left")
    save(fig, "fig06_seasonal_curve.png")


# ---- Hình 7: RMSE tổng thể ----------------------------------------------------------------------
def fig_rmse():
    r = pd.read_csv(TABLES / "rmse.csv").set_index("model")
    models = ["persistence", "clim5", BASELINE, "gam"]
    colors = [C_OTHER, "#d4d4d4", C_AR1, C_GAM]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.2
    for k, (m, col) in enumerate(zip(models, colors)):
        vals = [r.loc[m, str(h)] for h in HORIZONS]  # cột của rmse.csv là tầm h: "1", "2", "4"
        xs = np.arange(3) + (k - 1.5) * width
        ax.bar(xs, vals, width, color=col, label=NAMES[m], ec="white")
        if m in ("gam", BASELINE):
            for x, v in zip(xs, vals):
                ax.text(x, v + 40, f"{v:,.0f}", ha="center", fontsize=7, rotation=90)
    ax.set(xticks=range(3), ylabel="RMSE (km²), lower is better", ylim=(0, 3700),
           title="Pooled RMSE over 11 test seasons: GAM nearly matches AnomAR1 at all three horizons")
    ax.set_xticklabels([f"{DAYS[h]} days" for h in HORIZONS])
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    save(fig, "fig07_rmse.png")


# ---- Hình 8: từng mùa test ----------------------------------------------------------------------
def fig_seasons_ratio():
    s = pd.read_csv(TABLES / "gam_season_rmse.csv")
    t = pd.read_csv(TABLES / "vs_anom_ar1.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        g = s[s["h"] == h]
        cols = [C_GAM if b else C_AR1 for b in g["gam_better"]]
        ax.scatter(g["season"], g["ratio"], c=cols, s=45, zorder=3)
        ax.axhline(1, color="black", lw=0.8)
        row = t[t["h"] == h].iloc[0]
        ax.set(title=f"Horizon {DAYS[h]} days: wins {int(row.wins)}/11, TOST p = {row.p_tost:.3f}",
               xlabel="Test season", ylim=(0.7, 1.6))
        ax.text(0.98, 0.03, f"Mean MSE diff {row.mse_diff_pct:+.1f}%\n95% CI [{row.mse_diff_pct_ci_low:+.0f}; "
                f"{row.mse_diff_pct_ci_high:+.0f}]%", transform=ax.transAxes, va="bottom", ha="right",
                fontsize=8, bbox=dict(fc="white", ec="#d1d5db"))
    axes[0].set_ylabel("GAM RMSE / AnomAR1 RMSE\n(< 1: GAM better)")
    axes[0].legend(handles=[Patch(color=C_GAM, label="GAM better"), Patch(color=C_AR1, label="AnomAR1 better")],
                   fontsize=8, loc="upper right")
    save(fig, "fig08_season_ratio.png")


# ---- Hình 9: theo điều kiện ----------------------------------------------------------------------
def fig_conditions():
    ph = pd.read_csv(TABLES / "rmse_by_phase.csv")
    mg = pd.read_csv(TABLES / "rmse_by_size.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for col, h in enumerate(HORIZONS):
        for row, (tab, keys, names, models) in enumerate([
                (ph, PHASE_NAMES, PHASE_LABEL, ["gam", BASELINE]),
                (mg, SIZE_NAMES, SIZE_LABEL, ["gam", BASELINE, "gam_w7", "anom_ar1_w7"])]):
            ax = axes[row, col]
            g = tab[tab["h"] == h].set_index("model")
            w = 0.8 / len(models)
            for k, m in enumerate(models):
                color = C_GAM if m.startswith("gam") else C_AR1
                ax.bar(np.arange(3) + (k - (len(models) - 1) / 2) * w, g.loc[m, keys], w, color=color,
                       alpha=1 if not m.endswith("_w7") else 0.45, label=NAMES[m], ec="white")
            ax.set(xticks=range(3), title=f"Horizon {DAYS[h]} days")
            ax.set_xticklabels([names[k] for k in keys])
            if col == 0:
                ax.set_ylabel("RMSE (km²)")
                ax.legend(fontsize=8)
    fig.suptitle("Top row: by flood phase (target day). Bottom row: by test-season flood size (peak tertiles)")
    save(fig, "fig09_conditions.png")


# ---- Hình 10: phần dư -----------------------------------------------------------------------------
def fig_residuals():
    p = pd.read_csv(TABLES / "predictions.csv").query("model == 'gam'")
    summ = pd.read_csv(TABLES / "gam_residual_summary.csv").set_index("h")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for col, h in enumerate(HORIZONS):
        g = p[p["h"] == h]
        err = (g["y_target"] - g["pred"]).to_numpy()
        (osm, osr), (slope, icpt, _) = stats.probplot(err, dist="norm")
        ax = axes[0, col]
        ax.scatter(osm, osr, s=8, color=C_GAM)
        ax.plot(osm, icpt + slope * osm, color="black", lw=0.8)
        ax.set(title=f"Q–Q, {DAYS[h]} days (Shapiro p = {summ.loc[h, 'shapiro_p']:.2f})",
               xlabel="Theoretical normal quantiles", ylabel="Residual (km²)")

        ax = axes[1, col]
        lags = [1, 2, 3]
        ax.bar(np.array(lags) - 0.18, [summ.loc[h, f"mean_acf_{k}"] for k in lags], 0.36, color=C_GAM,
               label="Observed (mean of 11 seasons)")
        ax.bar(np.array(lags) + 0.18, [summ.loc[h, f"theory_acf_{k}"] for k in lags], 0.36, color=C_CLIM,
               label="Theoretical if anomaly is AR(1)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set(title=f"Within-season residual autocorrelation, {DAYS[h]} days", xlabel="Lag (steps)",
               xticks=lags, ylim=(-0.4, 1))
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[2, col]
        ax.scatter(g["doy_target"], err, s=8, color=C_GAM, alpha=0.6)
        binned = pd.Series(err).groupby(pd.cut(g["doy_target"].to_numpy(), 6)).mean()
        ax.plot([iv.mid for iv in binned.index], binned.to_numpy(), "o-", color=C_AR1, label="Binned mean")
        ax.axhline(0, color="black", lw=0.6)
        ax.set(title=f"Residuals by target day, {DAYS[h]} days", xlabel="Target day", ylabel="Actual − forecast (km²)")
        if col == 0:
            ax.legend(fontsize=8)
    save(fig, "fig10_residuals.png")


# ---- Hình 11: hai mùa tiêu biểu ------------------------------------------------------------------
def fig_two_seasons(h=2):
    p = pd.read_csv(TABLES / "predictions.csv")
    s = pd.read_csv(TABLES / "gam_season_rmse.csv").query("h == @h")
    best, worst = s.loc[s["ratio"].idxmin()], s.loc[s["ratio"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, row, tag in zip(axes, [best, worst], ["GAM clearly best", "GAM clearly worst"]):
        g = p[(p["season"] == row.season) & (p["h"] == h)]
        gam, ar = g[g["model"] == "gam"].sort_values("doy_target"), g[g["model"] == BASELINE].sort_values("doy_target")
        clim = g[g["model"] == "clim5"].sort_values("doy_target")  # dự báo clim5 = mức bình thường c_target
        ax.plot(gam["doy_target"], gam["y_target"] / 1000, color=C_OBS, lw=2, label="Actual")
        ax.plot(clim["doy_target"], clim["pred"] / 1000, "--", color=C_CLIM, label="Normal level c")
        ax.plot(gam["doy_target"], gam["pred"] / 1000, "o-", ms=3, color=C_GAM, label=f"GAM (RMSE {row.rmse_gam:,.0f})")
        ax.plot(ar["doy_target"], ar["pred"] / 1000, "s-", ms=3, color=C_AR1, label=f"AnomAR1 (RMSE {row.rmse_anom_ar1:,.0f})")
        ax.set(title=f"Season {int(row.season)} – {tag}\n(RMSE ratio {row.ratio:.2f}, GAM bias {row.bias_gam:+,.0f} km²)",
               xlabel="Target day", ylabel="Flooded area (thousand km²)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{DAYS[h]}-day-ahead forecast, full season", y=1.02)
    fig.savefig(FIGURES / "fig11_two_seasons.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> fig11_two_seasons.png")


# ---- Hình 12: bóc tách ---------------------------------------------------------------------------
def fig_ablation():
    a = pd.read_csv(TABLES / "gam_ablation_rmse.csv")
    models = ["gam", BASELINE, "gam_linear_doy", "ar1_direct_intercept"]
    colors = [C_GAM, C_AR1, "#93c5fd", "#1e3a8a"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, (m, c) in enumerate(zip(models, colors)):
        g = a[a["model"] == m].set_index("h")
        xs = np.arange(3) + (k - 1.5) * 0.2
        ax.bar(xs, [g.loc[h, "rmse"] for h in HORIZONS], 0.2, color=c, label=NAMES[m], ec="white")
        for x, h in zip(xs, HORIZONS):
            w = g.loc[h, "wins_vs_gam"]
            ax.text(x, g.loc[h, "rmse"] + 15, f"{g.loc[h, 'rmse']:,.0f}" + ("" if np.isnan(w) else f"\n{int(w)}/11"),
                    ha="center", fontsize=7)
    ax.set(xticks=range(3), ylim=(900, 1850), ylabel="RMSE (km²)",
           title="SUPPLEMENTARY (after seeing test): removing GAM parts step by step. x/11 = seasons better than full GAM")
    ax.set_xticklabels([f"{DAYS[h]} days" for h in HORIZONS])
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    save(fig, "fig12_ablation.png")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    cfg, Y = load_data()
    fig_data(Y, cfg)
    fig_protocol(cfg)
    fig_one_forecast(Y, cfg)
    fig_configs()
    fig_coefficients(Y, cfg)
    fig_seasonal_curve()
    fig_rmse()
    fig_seasons_ratio()
    fig_conditions()
    fig_residuals()
    fig_two_seasons()
    fig_ablation()


if __name__ == "__main__":
    main()

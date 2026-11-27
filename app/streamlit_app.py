"""Interactive report: the group's 4 models compared with the AnomAR1 baseline.

Run from the repo root:  streamlit run app/streamlit_app.py

Reads only the standard tables of each model (results/<model>/tables/{predictions,rmse,vs_anom_ar1}.csv),
the combined tests in results/Comparison/ and data/processed/level1_series.csv. No model is retrained.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRS = ["AnomAR1", "GAM", "LightGBM", "Chronos-2"]  # folders in results/
BASE = "anom_ar1"
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
MAIN_MODELS = [BASE, "gam", "lgbm_raw", "chronos2"]  # each member's main model
LABELS = {
    "anom_ar1": "AnomAR1", "anom_ar1_w7": "AnomAR1 (last 7 seasons)", "clim5": "Seasonal average",
    "persistence": "No change",
    "gam": "GAM", "gam_w7": "GAM (last 7 seasons)",
    "lgbm_raw": "LightGBM", "lgbm_anomaly": "LightGBM (anomaly)", "lgbm_noclim": "LightGBM (no seasonal average)",
    "chronos2": "Chronos-2",
}
PHASE_EDGES, PHASES = [0, 257, 298, 999], ["Water rising", "Peak", "Water receding"]
SIZES = ["Small flood", "Medium flood", "Large flood"]

st.set_page_config(page_title="Mekong Delta flood forecasts", layout="wide")

# --- Look and feel taken from the Remotion "flood" video (src/projects/flood/theme.ts). ---
PAPER, WASH, INK, MUTED, RULE = "#ffffff", "#eaf6fd", "#111111", "#6b7280", "#d9dde3"
SKY_LIGHT, SKY_MID, SKY_DEEP, BASE_GREY = "#cfeafb", "#8fd0f5", "#3fa9e8", "#b4bac3"
FONT_BODY, FONT_DISPLAY, FONT_MONO = "Inter", "Space Grotesk", "JetBrains Mono"


@alt.theme.register("flood", enable=True)
def _flood_theme() -> alt.theme.ThemeConfig:
    """Axes and grid as in the video: light grey rules, grey mono labels, only white, black and sky blue.
    The observed line is black, so the colour range starts with deep sky blue and keeps black for last."""
    axis = {
        "domainColor": INK, "domainWidth": 1.5, "grid": True, "gridColor": RULE, "gridWidth": 1,
        "tickColor": RULE, "labelColor": MUTED, "labelFont": FONT_MONO, "labelFontSize": 12,
        "titleColor": MUTED, "titleFont": FONT_MONO, "titleFontSize": 12, "titleFontWeight": 400,
    }
    return {
        "config": {
            "background": PAPER,
            "font": FONT_BODY,
            "view": {"stroke": None},
            "range": {"category": [SKY_DEEP, SKY_MID, MUTED, BASE_GREY, INK, SKY_LIGHT]},
            "axis": axis,
            "axisX": {"grid": False},
            "legend": {"labelColor": INK, "labelFont": FONT_MONO, "labelFontSize": 12,
                       "titleColor": MUTED, "titleFont": FONT_MONO, "titleFontWeight": 400},
            "title": {"color": INK, "font": FONT_DISPLAY, "fontWeight": 700},
            "line": {"strokeWidth": 2.5},
            "bar": {"cornerRadius": 0},
            "point": {"color": INK, "strokeWidth": 2},
            "rule": {"color": INK, "strokeWidth": 2},
        }
    }


st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@500;700&display=swap&subset=vietnamese');
h1, h2, h3 {{ font-family: '{FONT_DISPLAY}', sans-serif !important; letter-spacing: -0.03em; color: {INK}; }}
h1 {{ font-weight: 700 !important; }}
[data-testid="stCaptionContainer"] {{ font-family: '{FONT_MONO}', monospace; color: {MUTED}; }}
[data-testid="stMetric"] {{ border: 3px solid {INK}; background: {PAPER}; padding: 14px 18px; }}
[data-testid="stMetricLabel"] {{ font-family: '{FONT_MONO}', monospace; color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-family: '{FONT_MONO}', monospace; font-weight: 700; letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums; color: {INK}; }}
.stTabs [data-baseweb="tab"] p {{ font-family: '{FONT_MONO}', monospace; text-transform: uppercase;
    letter-spacing: 0.08em; font-size: 0.85rem; }}
.stTabs [aria-selected="true"] p {{ color: {INK}; font-weight: 700; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {INK}; height: 3px; }}
</style>""", unsafe_allow_html=True)


def label(model: str) -> str:
    return LABELS.get(model, model)


def _read(model_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / model_dir / "tables" / f"{name}.csv").assign(source=model_dir)


@st.cache_data
def predictions() -> pd.DataFrame:
    """Forecasts of every model; AnomAR1 and the simple baselines appear in several folders, so keep one copy."""
    p = pd.concat([_read(d, "predictions") for d in MODEL_DIRS], ignore_index=True)
    return p.drop_duplicates(["model", "h", "season", "t"]).drop(columns="source")


@st.cache_data
def rmse() -> pd.DataFrame:
    r = pd.concat([_read(d, "rmse") for d in MODEL_DIRS], ignore_index=True)
    return r.drop_duplicates("model").drop(columns="source").set_index("model")


@st.cache_data
def tests() -> pd.DataFrame:
    """Tests from each folder, plus the p-value adjusted over all 9 comparisons from results/Comparison if present."""
    own = pd.concat([_read(d, "vs_anom_ar1") for d in MODEL_DIRS], ignore_index=True)
    path = REPO_ROOT / "results" / "Comparison" / "tables" / "vs_anom_ar1_all.csv"
    if path.exists():
        glob = pd.read_csv(path)[["model", "h", "p_t_holm_main"]]
        own = own.merge(glob, on=["model", "h"], how="left")
    return own


@st.cache_data
def series() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "level1_series.csv")




def condition_rmse(pred: pd.DataFrame, by: str) -> pd.DataFrame:
    """Error by flood stage (from the forecast day) or by flood size (11 test seasons split into 3 by peak)."""
    if by == "phase":
        key = pd.cut(pred["doy_target"], PHASE_EDGES, right=False, labels=PHASES)
    else:
        key = pred["season"].map(pd.qcut(pred.groupby("season")["y_target"].max(), 3, labels=SIZES))
    sq = (pred["y_target"] - pred["pred"]) ** 2
    out = sq.groupby([pred["model"], key], observed=True).mean().pow(0.5).rename("rmse").reset_index()
    return out.rename(columns={out.columns[1]: by})


def bar(df: pd.DataFrame, var: str, height: int = 380):
    """Error bars per model, coloured by `var` (horizon, flood stage or flood size)."""
    data = df.assign(model=df["model"].map(label))
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("model:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("rmse:Q", title="Typical error, RMSE (km²)"),
        color=alt.Color(f"{var}:N", title=None, sort=None),
        xOffset=alt.XOffset(f"{var}:N", sort=None),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip(f"{var}:N", title="Group"),
                 alt.Tooltip("rmse:Q", title="RMSE (km²)", format=",.0f")],
    ).properties(height=height)


def days(h) -> str:
    return f"{HORIZON_DAYS[int(h)]} days ahead"


def rmse_long(table: pd.DataFrame, models) -> pd.DataFrame:
    t = table.loc[[m for m in models if m in table.index]]
    t.columns = [days(h) for h in t.columns]
    return t.reset_index().melt("model", var_name="h", value_name="rmse")


TEST_COLUMNS = {
    "model": "Model",
    "source": "Results folder",
    "n_seasons": "Test seasons",
    "rmse_ratio": "Error vs AnomAR1 (ratio)",
    "wins": "Seasons where it beat AnomAR1",
    "mse_diff_pct": "Change in squared error (%)",
    "mse_diff_pct_ci_low": "95% range: low (%)",
    "mse_diff_pct_ci_high": "95% range: high (%)",
    "p_t": "p (t-test)",
    "p_wilcoxon": "p (Wilcoxon)",
    "p_tost": "p (same within ±5%)",
    "p_t_holm": "p adjusted (within its folder)",
    "p_t_holm_main": "p adjusted (all 9 main comparisons)",
}

st.title("Forecasting flooded area in the Mekong Delta")
st.caption("Can three newer models beat a simple baseline (AnomAR1)? Each model was tested on 11 flood seasons "
           "(2010–2020), always trained only on earlier years, and asked to forecast 8, 16 and 32 days ahead.")

R, T, P = rmse(), tests(), predictions()
tab_sum, tab_data, tab_rmse, tab_test, tab_cond, tab_pred = st.tabs(
    ["Summary", "Data", "Errors", "Significance", "When models differ", "Season by season"])

with tab_sum:
    st.markdown("**Short answer:** no model is clearly better than AnomAR1. LightGBM has slightly smaller errors "
                "at 8 and 16 days, but with only 11 seasons the gain could be chance. Chronos-2 is clearly worse "
                "at 32 days.")
    cols = st.columns(3)
    for col, model in zip(cols, ["gam", "lgbm_raw", "chronos2"]):
        rows = T[T["model"] == model].set_index("h")
        p_holm = rows.loc[2, "p_t_holm_main"] if "p_t_holm_main" in rows else rows.loc[2, "p_t_holm"]
        col.metric(f"{label(model)} error ÷ AnomAR1 error, 16 days", f"{rows.loc[2, 'rmse_ratio']:.3f}",
                   help=f"Beat AnomAR1 in {int(rows.loc[2, 'wins'])} of 11 seasons. "
                        f"Adjusted p-value (all 9 comparisons) = {p_holm:.3f}; "
                        f"p-value for 'about the same, within ±5%' = {rows.loc[2, 'p_tost']:.3f}.")
    st.markdown("A ratio **below 1** means smaller errors than AnomAR1; above 1 means larger. "
                "The table shows each model's typical error (RMSE, in km² of flooded area).")
    main = R.loc[MAIN_MODELS].rename(index=label, columns=days)
    st.dataframe(main.style.format("{:,.0f}"), use_container_width=True)

with tab_data:
    s = series()
    st.subheader("Flooded area through each season")
    seasons = st.multiselect("Seasons to show", sorted(s.season.unique()), default=[2005, 2010, 2015, 2020])
    sub = s[s.season.isin(seasons)]
    st.altair_chart(alt.Chart(sub).mark_line().encode(
        x=alt.X("doy:Q", title="Day of year"), y=alt.Y("target_km2:Q", title="Flooded area (km²)"),
        color=alt.Color("season:N", title="Season"),
        tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("doy:Q", title="Day of year"),
                 alt.Tooltip("target_km2:Q", title="Flooded area (km²)", format=",.0f"),
                 alt.Tooltip("imputed:N", title="Filled in")]).properties(height=360),
        use_container_width=True, theme=None)
    st.caption("One value every 8 days from satellite images. Two missing images (2013 day 233, 2019 day 257) "
               "were filled in and are marked, not removed.")

with tab_rmse:
    st.markdown("Typical forecast error (RMSE) over the 11 test seasons. **Lower is better.** "
                "Errors grow the further ahead a model forecasts.")
    show_all = st.checkbox("Also show variants and the simplest baselines "
                           "(seasonal average, and 'no change' = keep today's value)", value=False)
    models = list(R.index) if show_all else MAIN_MODELS
    st.altair_chart(bar(rmse_long(R, models), "h"), use_container_width=True, theme=None)
    st.dataframe(R.loc[models].rename(index=label, columns=days).style.format("{:,.0f}"),
                 use_container_width=True)

with tab_test:
    st.subheader("Is the difference from AnomAR1 real, or could it be chance?")
    st.markdown("Each of the 11 test seasons counts as one comparison. Forecasts within a season are strongly "
                "linked, so treating every forecast as separate would overstate the evidence.")
    horizon = st.radio("Forecast horizon", list(HORIZON_DAYS), horizontal=True, format_func=days)
    tt = T[T["h"] == horizon].assign(model=lambda d: d["model"].map(label))
    ci = tt[tt["model"].isin([label(m) for m in MAIN_MODELS])]
    base = alt.Chart(ci)
    st.altair_chart(
        base.mark_rule().encode(y=alt.Y("model:N", sort=None, title=None),
                                x=alt.X("mse_diff_pct_ci_low:Q",
                                        title="Change in squared error vs AnomAR1 (%), with 95% range"),
                                x2="mse_diff_pct_ci_high:Q")
        + base.mark_point(filled=True, size=70).encode(y=alt.Y("model:N", sort=None), x="mse_diff_pct:Q")
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"),
        use_container_width=True, theme=None)
    st.markdown(
        "- **How to read the chart:** the dot is the average change and the line is the 95% range. "
        "Left of the dashed line means smaller errors than AnomAR1. If the line crosses the dashed line, "
        "we cannot tell the two apart.\n"
        "- **Adjusted p-value:** we make 9 comparisons (3 models × 3 horizons), so a few could look significant "
        "by luck. The adjusted p-value (Holm method) corrects for this; below 0.05 counts as a real difference.\n"
        "- **'Same within ±5%':** a small p-value here is evidence that the model and AnomAR1 are about equally "
        "good. No significant difference is *not* proof that they are equal.")
    st.dataframe(tt.drop(columns="h").rename(columns=TEST_COLUMNS), hide_index=True, use_container_width=True)

with tab_cond:
    st.markdown("Where in the season does each model do better or worse? These splits only describe the results; "
                "they were not tested for significance.")
    horizon_c = st.radio("Forecast horizon", list(HORIZON_DAYS), index=1, horizontal=True, key="cond_h",
                         format_func=days)
    main_pred = P[(P["h"] == horizon_c) & P["model"].isin(MAIN_MODELS)]
    main_pred = main_pred.assign(model=pd.Categorical(main_pred["model"], MAIN_MODELS)).sort_values("model")
    st.subheader("By stage of the flood")
    st.altair_chart(bar(condition_rmse(main_pred, "phase").astype({"model": str}), "phase"),
                    use_container_width=True, theme=None)
    st.subheader("By size of that year's flood")
    st.altair_chart(bar(condition_rmse(main_pred, "size").astype({"model": str}), "size"),
                    use_container_width=True, theme=None)
    st.caption("Stage is taken from the forecast day: water rising before day 257, peak on days 257–297, "
               "water receding from day 298. Flood size splits the 11 test seasons into three equal groups "
               "by their largest flooded area.")

with tab_pred:
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Test season", sorted(P.season.unique()))
    horizon_p = c2.selectbox("Forecast horizon", list(HORIZON_DAYS), format_func=days, index=1)
    models = c3.multiselect("Models", sorted(P.model.unique()), default=MAIN_MODELS, format_func=label)
    sub = P[(P.season == season) & (P.h == horizon_p) & P.model.isin(models)]
    obs = P[(P.season == season) & (P.h == horizon_p)].drop_duplicates("doy_target")[["doy_target", "y_target"]]
    lines = alt.Chart(sub.assign(model=sub.model.map(label))).mark_line().encode(
        x=alt.X("doy_target:Q", title="Day being forecast (day of year)"),
        y=alt.Y("pred:Q", title="Flooded area (km²)"), color=alt.Color("model:N", title="Model"))
    truth = alt.Chart(obs).mark_line(color="black", strokeWidth=3).encode(x="doy_target:Q", y="y_target:Q")
    st.altair_chart((truth + lines).properties(height=380), use_container_width=True, theme=None)
    st.caption("The thick black line is what actually happened. Each coloured point is a forecast "
               "made 8, 16 or 32 days earlier for that day.")

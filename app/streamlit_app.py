"""Báo cáo tương tác: so sánh 4 mô hình của nhóm với mốc AnomAR1. Chạy: streamlit run app/streamlit_app.py

Chỉ đọc 3 bảng chuẩn của mỗi mô hình (results/<mô hình>/tables/{predictions,rmse,vs_anom_ar1}.csv)
và data/processed/level1_series.csv; không tính lại mô hình.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRS = ["AnomAR1", "GAM", "LightGBM", "Chronos-2"]  # thư mục trong results/
BASE = "anom_ar1"
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
MAIN_MODELS = [BASE, "gam", "lgbm_raw", "chronos2"]  # bản chính của mỗi người
LABELS = {
    "anom_ar1": "AnomAR1", "anom_ar1_w7": "AnomAR1 (7 mùa)", "clim5": "Clim5", "persistence": "Persistence",
    "gam": "GAM", "gam_w7": "GAM (7 mùa)",
    "lgbm_raw": "LightGBM thô", "lgbm_anomaly": "LightGBM anomaly", "lgbm_noclim": "LightGBM không climatology",
    "chronos2": "Chronos-2",
}
PHASE_EDGES, PHASES = [0, 257, 298, 999], ["rising", "peak", "falling"]
SIZES = ["small", "medium", "large"]

st.set_page_config(page_title="Dự báo diện tích ngập ĐBSCL", layout="wide")

# --- Giao diện: lấy từ video Remotion "flood" (src/projects/flood/theme.ts). Chỉ đổi màu, font, nét. ---
PAPER, WASH, INK, MUTED, RULE = "#ffffff", "#eaf6fd", "#111111", "#6b7280", "#d9dde3"
SKY_LIGHT, SKY_MID, SKY_DEEP, BASE_GREY = "#cfeafb", "#8fd0f5", "#3fa9e8", "#b4bac3"
FONT_BODY, FONT_DISPLAY, FONT_MONO = "Inter", "Space Grotesk", "JetBrains Mono"


@alt.theme.register("flood", enable=True)
def _flood_theme() -> alt.theme.ThemeConfig:
    """Trục và lưới như SeasonAxes/RmseByHorizon: kẻ xám nhạt, nhãn mono xám, chỉ trắng/đen/xanh trời.
    Đường quan sát vẽ màu đen nên dãy màu bắt đầu bằng xanh trời đậm, đen để sau cùng."""
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
    """Dự báo của mọi mô hình; AnomAR1 và các mốc đơn giản xuất hiện ở nhiều thư mục nên chỉ giữ 1 bản."""
    p = pd.concat([_read(d, "predictions") for d in MODEL_DIRS], ignore_index=True)
    return p.drop_duplicates(["model", "h", "season", "t"]).drop(columns="source")


@st.cache_data
def rmse() -> pd.DataFrame:
    r = pd.concat([_read(d, "rmse") for d in MODEL_DIRS], ignore_index=True)
    return r.drop_duplicates("model").drop(columns="source").set_index("model")


@st.cache_data
def tests() -> pd.DataFrame:
    return pd.concat([_read(d, "vs_anom_ar1") for d in MODEL_DIRS], ignore_index=True)


@st.cache_data
def series() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "data" / "processed" / "level1_series.csv")


def condition_rmse(pred: pd.DataFrame, by: str) -> pd.DataFrame:
    """RMSE theo giai đoạn lũ (ngày đích) hoặc cỡ lũ (tam phân vị đỉnh của mùa test)."""
    if by == "phase":
        key = pd.cut(pred["doy_target"], PHASE_EDGES, right=False, labels=PHASES)
    else:
        key = pred["season"].map(pd.qcut(pred.groupby("season")["y_target"].max(), 3, labels=SIZES))
    sq = (pred["y_target"] - pred["pred"]) ** 2
    out = sq.groupby([pred["model"], key], observed=True).mean().pow(0.5).rename("rmse").reset_index()
    return out.rename(columns={out.columns[1]: by})


def bar(df: pd.DataFrame, var: str, height: int = 380):
    """Cột RMSE theo mô hình, tách màu theo `var` (tầm, giai đoạn hoặc cỡ lũ)."""
    data = df.assign(model=df["model"].map(label))
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("model:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("rmse:Q", title="RMSE (km²)"),
        color=alt.Color(f"{var}:N", title=None, sort=None),
        xOffset=alt.XOffset(f"{var}:N", sort=None),
        tooltip=["model", var, alt.Tooltip("rmse:Q", format=",.0f")],
    ).properties(height=height)


def rmse_long(table: pd.DataFrame, models) -> pd.DataFrame:
    t = table.loc[[m for m in models if m in table.index]]
    t.columns = [f"{HORIZON_DAYS[int(h)]} ngày" for h in t.columns]
    return t.reset_index().melt("model", var_name="h", value_name="rmse")


st.title("Dự báo diện tích ngập ĐBSCL: 4 mô hình so với AnomAR1")
st.caption("11 mùa test (2010–2020), rolling origin theo năm, tầm 8/16/32 ngày. "
           "Mỗi mô hình là một thư mục trong results/; số đọc từ bảng chuẩn của từng thư mục.")

R, T, P = rmse(), tests(), predictions()
tab_sum, tab_data, tab_rmse, tab_test, tab_cond, tab_pred = st.tabs(
    ["Tóm tắt", "Dữ liệu", "RMSE", "Kiểm định", "Theo điều kiện", "Dự báo từng mùa"])

with tab_sum:
    cols = st.columns(3)
    for col, model in zip(cols, ["gam", "lgbm_raw", "chronos2"]):
        rows = T[T["model"] == model].set_index("h")
        col.metric(f"{label(model)} / AnomAR1, 16 ngày", f"{rows.loc[2, 'rmse_ratio']:.3f}",
                   help=f"Thắng {int(rows.loc[2, 'wins'])}/11 mùa; p sau Holm = {rows.loc[2, 'p_t_holm']:.3f}; "
                        f"TOST ±5%: p = {rows.loc[2, 'p_tost']:.3f}.")
    st.markdown("Tỉ lệ RMSE < 1 là tốt hơn AnomAR1. Bảng dưới: RMSE gộp (km²) của bản chính mỗi mô hình.")
    main = R.loc[MAIN_MODELS].rename(index=label, columns=lambda h: f"{HORIZON_DAYS[int(h)]} ngày")
    st.dataframe(main.style.format("{:,.0f}"), use_container_width=True)

with tab_data:
    s = series()
    st.subheader("Chuỗi diện tích ngập theo mùa")
    seasons = st.multiselect("Mùa", sorted(s.season.unique()), default=[2005, 2010, 2015, 2020])
    sub = s[s.season.isin(seasons)]
    st.altair_chart(alt.Chart(sub).mark_line().encode(
        x=alt.X("doy:Q", title="Ngày trong năm"), y=alt.Y("target_km2:Q", title="Diện tích ngập (km²)"),
        color="season:N", tooltip=["season", "doy", "target_km2", "imputed", "duplicate"]).properties(height=360),
        use_container_width=True, theme=None)
    st.caption("2 ảnh nội suy (2013 ngày 233, 2019 ngày 257) được gắn cờ imputed, không bị xóa.")

with tab_rmse:
    show_all = st.checkbox("Hiện cả các biến thể và mốc đơn giản", value=False)
    models = list(R.index) if show_all else MAIN_MODELS
    st.altair_chart(bar(rmse_long(R, models), "h"), use_container_width=True, theme=None)
    st.dataframe(R.loc[models].rename(index=label, columns=lambda h: f"{HORIZON_DAYS[int(h)]} ngày")
                 .style.format("{:,.0f}"), use_container_width=True)

with tab_test:
    st.subheader("So với AnomAR1 ở cấp mùa (n = 11)")
    horizon = st.radio("Tầm dự báo", list(HORIZON_DAYS), horizontal=True, format_func=lambda h: f"{HORIZON_DAYS[h]} ngày")
    tt = T[T["h"] == horizon].assign(model=lambda d: d["model"].map(label))
    st.dataframe(tt.drop(columns="h").rename(columns={"source": "thư mục"}), hide_index=True,
                 use_container_width=True)
    st.markdown("CI chứa 0 thì chưa phân biệt được. p TOST nhỏ là có bằng chứng tương đương trong ±5% RMSE. "
                "\"Không bác bỏ\" không có nghĩa \"tương đương\". Holm hiệu chỉnh trong bảng của từng thư mục.")
    ci = tt[tt["model"].isin([label(m) for m in MAIN_MODELS])]
    base = alt.Chart(ci)
    st.altair_chart(
        base.mark_rule().encode(y=alt.Y("model:N", sort=None, title=None),
                                x=alt.X("mse_diff_pct_ci_low:Q", title="Chênh MSE so với AnomAR1 (%), CI 95%"),
                                x2="mse_diff_pct_ci_high:Q")
        + base.mark_point(filled=True, size=70).encode(y=alt.Y("model:N", sort=None), x="mse_diff_pct:Q")
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"),
        use_container_width=True, theme=None)

with tab_cond:
    horizon_c = st.radio("Tầm", list(HORIZON_DAYS), index=1, horizontal=True, key="cond_h",
                         format_func=lambda h: f"{HORIZON_DAYS[h]} ngày")
    main_pred = P[(P["h"] == horizon_c) & P["model"].isin(MAIN_MODELS)]
    main_pred = main_pred.assign(model=pd.Categorical(main_pred["model"], MAIN_MODELS)).sort_values("model")
    st.subheader("Giai đoạn lũ")
    st.altair_chart(bar(condition_rmse(main_pred, "phase").astype({"model": str}), "phase"), use_container_width=True, theme=None)
    st.subheader("Cỡ lũ của mùa test")
    st.altair_chart(bar(condition_rmse(main_pred, "size").astype({"model": str}), "size"), use_container_width=True, theme=None)
    st.caption("Giai đoạn theo ngày đích: nước lên < 257, đỉnh 257–297, nước rút ≥ 298. "
               "Cỡ lũ: chia 11 mùa test thành 3 nhóm theo đỉnh diện tích ngập.")

with tab_pred:
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Mùa test", sorted(P.season.unique()))
    horizon_p = c2.selectbox("Tầm", list(HORIZON_DAYS), format_func=lambda h: f"{HORIZON_DAYS[h]} ngày", index=1)
    models = c3.multiselect("Mô hình", sorted(P.model.unique()), default=MAIN_MODELS, format_func=label)
    sub = P[(P.season == season) & (P.h == horizon_p) & P.model.isin(models)]
    obs = P[(P.season == season) & (P.h == horizon_p)].drop_duplicates("doy_target")[["doy_target", "y_target"]]
    lines = alt.Chart(sub.assign(model=sub.model.map(label))).mark_line().encode(
        x=alt.X("doy_target:Q", title="Ngày được dự báo"), y=alt.Y("pred:Q", title="km²"), color="model:N")
    truth = alt.Chart(obs).mark_line(color="black", strokeWidth=3).encode(x="doy_target:Q", y="y_target:Q")
    st.altair_chart((truth + lines).properties(height=380), use_container_width=True, theme=None)
    st.caption("Đường đen = giá trị quan sát. Mỗi điểm là dự báo thực hiện trước đó 8/16/32 ngày.")

"""Báo cáo tương tác kết quả thăm dò. Chạy: streamlit run app/streamlit_app.py

Chỉ đọc results/tables/*.csv và data/processed/; không tính lại mô hình.
"""
import altair as alt
import pandas as pd
import streamlit as st

from floodbench.data.regions import load_region, repo_root
from floodbench.report_md import BASE, COLUMN_LABELS, HORIZON_DAYS, build_markdown, label, load, tests_table

st.set_page_config(page_title="Dự báo diện tích ngập ĐBSCL", layout="wide")


@st.cache_data
def tables():
    return load()


@st.cache_data
def predictions():
    return pd.read_csv(repo_root() / "results" / "tables" / "probe_predictions.csv")


@st.cache_data
def series():
    cfg = load_region("mekong")
    return pd.read_csv(cfg.path("processed_level1"))


def long_form(df: pd.DataFrame, var: str) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].map(label)
    return out.melt("model", var_name=var, value_name="rmse").assign(
        **{var: lambda d: d[var].map(lambda v: COLUMN_LABELS.get(str(v), str(v)))})


def bar(df: pd.DataFrame, var: str, height: int = 380):
    data = long_form(df, var)
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("model:N", sort=None, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("rmse:Q", title="RMSE (km²)"),
        color=alt.Color(f"{var}:N", title=None),
        xOffset=f"{var}:N",
        tooltip=["model", var, alt.Tooltip("rmse:Q", format=",.0f")],
    ).properties(height=height)


st.title("Dự báo diện tích ngập ĐBSCL: kết quả thăm dò")
st.caption("Bản thăm dò của một thành viên, chưa phải kết quả cuối. 11 mùa test (2010–2020), một lần chạy, một chuỗi.")

t = tables()
tab_sum, tab_data, tab_rmse, tab_test, tab_cond, tab_pred, tab_md = st.tabs(
    ["Tóm tắt", "Dữ liệu", "RMSE", "Kiểm định", "Theo điều kiện", "Dự báo từng mùa", "Báo cáo markdown"])

with tab_sum:
    tests = t["probe_vs_anom_ar1"]
    rmse = t["probe_rmse"].set_index("model")
    g = tests[(tests.model == "gam") & (tests.h == 1)].iloc[0]
    c = tests[(tests.model == "chronos2") & (tests.h == 4)].iloc[0]
    l1 = tests[(tests.model == "lgbm_raw") & (tests.h == 1)].iloc[0]
    a, b, d = st.columns(3)
    a.metric("GAM vs AnomAR1, 8 ngày", f"{g.rmse_ratio:.3f}", help=f"TOST ±5%: p = {g.p_tost:.4f}. Tương đương.")
    b.metric("LightGBM thô vs AnomAR1, 8 ngày", f"{l1.rmse_ratio:.3f}", help="CI của chênh MSE chứa 0, chưa kết luận được.")
    d.metric("Chronos-2 vs AnomAR1, 32 ngày", f"{c.rmse_ratio:.3f}", help=f"p sau Holm = {c.p_t_holm:.3f}.")
    st.markdown("""
- **Chưa mô hình nào thắng AnomAR1 có ý nghĩa** sau hiệu chỉnh Holm.
- **Nhóm thống kê (ARX, Ridge, GAM) tương đương AnomAR1 ở 8 ngày** trong biên ±5% RMSE (p TOST chưa hiệu chỉnh đa kiểm định).
- **Random Forest thô kém LightGBM thô**; dự báo anomaly giúp RF bám sát AnomAR1.
- **LightGBM** nhỉnh về số học ở 8–16 ngày nhưng khoảng tin cậy rộng.
- **Điểm đáng bàn là kết quả theo điều kiện**: giai đoạn lũ, và việc cỡ lũ lẫn với lượng dữ liệu học.
- Hướng nên theo: **so sánh theo điều kiện**, không phải chứng minh ML thắng.
""")
    st.warning("Chưa tái lập được AnomAR1 = 1.602 km² ở 32 ngày của lần thăm dò cũ "
               f"(bản này: {rmse.loc[BASE, '4']:,.0f}). Xem docs/decisions.md.")

with tab_data:
    s = series()
    st.subheader("Chuỗi diện tích ngập theo mùa")
    seasons = st.multiselect("Mùa", sorted(s.season.unique()), default=[2005, 2010, 2015, 2020])
    sub = s[s.season.isin(seasons)]
    st.altair_chart(alt.Chart(sub).mark_line().encode(
        x=alt.X("doy:Q", title="Ngày trong năm"), y=alt.Y("target_km2:Q", title="Diện tích ngập (km²)"),
        color="season:N", tooltip=["season", "doy", "target_km2", "imputed", "duplicate"]).properties(height=360),
        use_container_width=True)
    st.caption("2 ảnh nội suy (2013 ngày 233, 2019 ngày 257) và 15 ảnh nghi là bản sao được gắn cờ, không bị xóa.")
    st.subheader("Đối chiếu mực nước")
    st.dataframe(t["waterlevel_validation"], hide_index=True, use_container_width=True)
    st.caption("r_detrended là tương quan sau khi bỏ xu hướng tuyến tính; đây mới là con số nên trích.")
    st.subheader("Độ nhạy ở cấp dữ liệu")
    st.dataframe(t["data_sensitivity"], hide_index=True, use_container_width=True)

with tab_rmse:
    st.subheader("RMSE gộp, baseline cùng bảng")
    st.altair_chart(bar(t["probe_rmse"], "h"), use_container_width=True)
    show = t["probe_rmse"].copy()
    show["model"] = show["model"].map(label)
    st.dataframe(show.rename(columns=COLUMN_LABELS), hide_index=True, use_container_width=True)

with tab_test:
    st.subheader("So với AnomAR1 ở cấp mùa (n = 11)")
    horizon = st.radio("Tầm dự báo", list(HORIZON_DAYS.values()), horizontal=True, format_func=lambda d: f"{d} ngày")
    tt = tests_table(t["probe_vs_anom_ar1"])
    st.dataframe(tt[tt["Tầm"] == f"{horizon} ngày"], hide_index=True, use_container_width=True)
    st.markdown("Tỉ lệ < 1 là tốt hơn AnomAR1. CI chứa 0 thì chưa phân biệt được. "
                "p TOST nhỏ là có bằng chứng tương đương trong ±5% RMSE. "
                "\"Không bác bỏ\" không có nghĩa \"tương đương\".")
    h = next(k for k, v in HORIZON_DAYS.items() if v == horizon)
    ci = t["probe_vs_anom_ar1"]
    ci = ci[(ci.h == h) & ci.model.isin(["gam", "lgbm_raw", "lgbm_anomaly", "lgbm_noclim", "chronos2"])].assign(
        name=lambda d: d.model.map(label))
    base = alt.Chart(ci)
    st.altair_chart(
        base.mark_rule().encode(y=alt.Y("name:N", sort=None, title=None),
                                x=alt.X("mse_diff_pct_ci_low:Q", title="Chênh MSE so với AnomAR1 (%), CI 95%"),
                                x2="mse_diff_pct_ci_high:Q")
        + base.mark_point(filled=True, size=70).encode(y=alt.Y("name:N", sort=None), x="mse_diff_pct:Q")
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"),
        use_container_width=True)

with tab_cond:
    st.subheader("Giai đoạn lũ (tầm 16 ngày)")
    st.altair_chart(bar(t["probe_by_phase_h2"], "phase"), use_container_width=True)
    st.subheader("Cỡ lũ của mùa test (tầm 16 ngày)")
    st.altair_chart(bar(t["probe_by_magnitude_h2"], "size"), use_container_width=True)
    st.subheader("Cố định 7 mùa huấn luyện: cỡ lũ và lượng dữ liệu bị lẫn nhau")
    st.altair_chart(bar(t["probe_window7_by_magnitude_h2"], "size"), use_container_width=True)
    st.caption("Các mùa lũ nhỏ đều là những năm cuối nên mô hình có nhiều mùa học hơn. "
               "Với 7 mùa cố định, ưu thế của LightGBM ở lũ nhỏ biến mất.")
    st.subheader("Toàn bộ dữ liệu so với 7 mùa gần nhất")
    st.altair_chart(bar(t["probe_window7"], "h"), use_container_width=True)

with tab_pred:
    p = predictions()
    p = p[p.h.isin(HORIZON_DAYS)]
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Mùa test", sorted(p.season.unique()))
    horizon = c2.selectbox("Tầm", list(HORIZON_DAYS), format_func=lambda h: f"{HORIZON_DAYS[h]} ngày", index=1)
    models = c3.multiselect("Mô hình", sorted(p.model.unique()), default=[BASE, "gam", "lgbm_raw"], format_func=label)
    sub = p[(p.season == season) & (p.h == horizon) & p.model.isin(models)]
    obs = p[(p.season == season) & (p.h == horizon)].drop_duplicates("doy_target")[["doy_target", "y_target"]]
    lines = alt.Chart(sub.assign(model=sub.model.map(label))).mark_line().encode(
        x=alt.X("doy_target:Q", title="Ngày được dự báo"), y=alt.Y("pred:Q", title="km²"), color="model:N")
    truth = alt.Chart(obs).mark_line(color="black", strokeWidth=3).encode(x="doy_target:Q", y="y_target:Q")
    st.altair_chart((truth + lines).properties(height=380), use_container_width=True)
    st.caption("Đường đen = giá trị quan sát. Mỗi điểm là dự báo thực hiện trước đó 8/16/32 ngày.")

with tab_md:
    md = build_markdown(t)
    st.download_button("Tải báo cáo .md", md, file_name="probe_report.md", mime="text/markdown")
    st.markdown(md)

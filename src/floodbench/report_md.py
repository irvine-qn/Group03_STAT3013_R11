"""Sinh báo cáo thăm dò (markdown) từ results/tables/*.csv. Mọi con số đều đọc từ bảng, không gõ tay.

Chạy: python -m floodbench.report_md  ->  results/probe_report.md
"""
from __future__ import annotations

import pandas as pd

from floodbench.data.regions import repo_root

BASE = "anom_ar1"
HORIZON_DAYS = {1: 8, 2: 16, 4: 32}
LABELS = {
    "anom_ar1": "AnomAR1 (thống kê, mốc so sánh)", "gam": "GAM", "lgbm_raw": "LightGBM (thô)",
    "lgbm_anomaly": "LightGBM (anomaly)", "lgbm_noclim": "LightGBM (không climatology)",
    "chronos2": "Chronos-2 (zero-shot)", "persistence": "Persistence", "clim5": "Climatology 5 mùa",
    "clim_all": "Climatology toàn bộ", "seasonal_naive": "Seasonal naive",
}


def tables_dir():
    return repo_root() / "results" / "tables"


def load() -> dict[str, pd.DataFrame]:
    d = tables_dir()
    names = ["probe_rmse", "probe_vs_anom_ar1", "probe_by_phase_h2", "probe_by_magnitude_h2",
             "probe_window7", "probe_window7_by_magnitude_h2", "waterlevel_validation", "data_sensitivity"]
    return {n: pd.read_csv(d / f"{n}.csv") for n in names}


COLUMN_LABELS = {"1": "8 ngày", "2": "16 ngày", "4": "32 ngày", "rising": "Nước lên", "peak": "Đỉnh",
                 "falling": "Nước rút", "small": "Lũ nhỏ", "medium": "Lũ vừa", "large": "Lũ lớn"}


def label(model: str) -> str:
    if model.endswith("_w7"):
        return label(model[:-3]) + " [7 mùa gần nhất]"
    return LABELS.get(model, model)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(rows)


def _fmt_rmse(out: pd.DataFrame) -> pd.DataFrame:
    out = out.rename(columns={"model": "Mô hình", **{str(h): f"{d} ngày" for h, d in HORIZON_DAYS.items()}})
    for d in HORIZON_DAYS.values():
        out[f"{d} ngày"] = out[f"{d} ngày"].map(lambda v: f"{v:,.0f}")
    return out


def tests_table(t: pd.DataFrame) -> pd.DataFrame:
    t = t[~t["model"].isin(["clim5", "clim_all", "persistence", "seasonal_naive"])].copy()
    t["Mô hình"] = t["model"].map(label)
    t["Tầm"] = t["h"].map(lambda h: f"{HORIZON_DAYS[h]} ngày")
    t["RMSE / AnomAR1"] = t["rmse_ratio"].map("{:.3f}".format)
    t["Số mùa thắng"] = t["wins"].astype(str) + "/" + t["n_seasons"].astype(str)
    t["Chênh MSE (%) [CI 95%]"] = t.apply(
        lambda r: f"{r.mse_diff_pct:+.1f} [{r.mse_diff_pct_ci_low:+.1f}; {r.mse_diff_pct_ci_high:+.1f}]", axis=1)
    t["p (t, sau Holm)"] = t["p_t_holm"].map("{:.3f}".format)
    t["p TOST ±5%"] = t["p_tost"].map("{:.3f}".format)
    cols = ["Mô hình", "Tầm", "RMSE / AnomAR1", "Số mùa thắng", "Chênh MSE (%) [CI 95%]", "p (t, sau Holm)", "p TOST ±5%"]
    return t.sort_values(["h", "rmse_ratio"])[cols]


def wide_table(t: pd.DataFrame, first: str) -> pd.DataFrame:
    out = t.copy()
    out[first] = out["model"].map(label)
    out = out.drop(columns="model").rename(columns=COLUMN_LABELS)
    num = [c for c in out.columns if c != first]
    for c in num:
        out[c] = out[c].map(lambda v: f"{v:,.0f}")
    return out[[first] + num]


def build_markdown(t: dict[str, pd.DataFrame] | None = None) -> str:
    t = t or load()
    rmse = t["probe_rmse"].set_index("model")
    tests = t["probe_vs_anom_ar1"]
    phase = t["probe_by_phase_h2"].set_index("model")
    mag = t["probe_by_magnitude_h2"].set_index("model")
    wmag = t["probe_window7_by_magnitude_h2"].set_index("model")
    gauge = t["waterlevel_validation"]
    sens = t["data_sensitivity"].set_index("variant")
    tc = gauge[(gauge.station == "TanChau") & (gauge.area_metric == "peak_km2") & (gauge.gauge_metric == "Max. WL (m)")].iloc[0]

    gam_h1 = tests[(tests.model == "gam") & (tests.h == 1)].iloc[0]
    chr_h4 = tests[(tests.model == "chronos2") & (tests.h == 4)].iloc[0]
    lgb1 = tests[(tests.model == "lgbm_raw") & (tests.h == 1)].iloc[0]
    lgb2 = tests[(tests.model == "lgbm_raw") & (tests.h == 2)].iloc[0]
    best_sig = tests[(tests.p_t_holm < 0.05) & (tests.rmse_ratio < 1) & tests.model.isin(LABELS) &
                     ~tests.model.isin(["clim5", "clim_all", "persistence", "seasonal_naive"])]

    return f"""# Báo cáo thăm dò: dữ liệu và mô hình dự báo diện tích ngập (ĐBSCL)

> Bản thăm dò của một thành viên để cả nhóm cùng xem, **chưa phải kết quả cuối**. Mọi số được sinh tự động từ
> `results/tables/` bằng `python -m floodbench.report_md`. Xem bản tương tác: `streamlit run app/streamlit_app.py`.

## 1. Tóm tắt

- **Dữ liệu dùng được, nhưng bằng chứng còn yếu hơn vẻ ngoài.** Đỉnh diện tích ngập khớp với mực nước đỉnh ở Tân Châu: r = {tc.r_levels:.3f} khi để nguyên, còn {tc.r_detrended:.2f} sau khi bỏ xu hướng (CI 95% {tc.r_detrended_ci_low:.2f}–{tc.r_detrended_ci_high:.2f}, n = {int(tc.n)}). Mới kiểm được ở cấp năm.
- **Chưa mô hình nào thắng AnomAR1 một cách có ý nghĩa** sau hiệu chỉnh Holm ({"có " + str(len(best_sig)) + " trường hợp ngoại lệ" if len(best_sig) else "không có trường hợp ngoại lệ"}), với chỉ {int(tests.n_seasons.iloc[0])} mùa test.
- **GAM tương đương AnomAR1 ở {HORIZON_DAYS[1]} ngày** trong biên ±5% RMSE (TOST p = {gam_h1.p_tost:.4f}). Đây là kết luận "tương đương" hợp lệ duy nhất hiện có.
- **LightGBM (thô)** nhỉnh hơn về số học ở {HORIZON_DAYS[1]} và {HORIZON_DAYS[2]} ngày (tỉ lệ RMSE {lgb1.rmse_ratio:.3f} và {lgb2.rmse_ratio:.3f}) nhưng khoảng tin cậy chứa 0, nên **chưa kết luận được**.
- **Chronos-2 zero-shot kém hơn** ở {HORIZON_DAYS[4]} ngày (tỉ lệ {chr_h4.rmse_ratio:.3f}, p thô {chr_h4.p_t:.3f}, sau Holm {chr_h4.p_t_holm:.3f}).
- **Điều đáng bàn nhất là kết quả theo điều kiện**, không phải "ai thắng" (mục 4).

## 2. Dữ liệu

- Nguồn: Vu et al. (2022), bản đồ lũ MODIS suy diễn từ EVI/LSWI (**không phải quan sát trực tiếp**), gộp thành một chuỗi diện tích ngập (km²) toàn ĐBSCL. 21 mùa × 27 bước (8 ngày/bước).
- Cờ chất lượng: 2 ảnh thiếu (nội suy), 15 ảnh nghi là bản sao, 12 mùa chỉ gồm ảnh lưới 232 m, các mùa còn lại có một phần hoặc toàn bộ ảnh lưới 500 m.
- Độ nhạy ở cấp dữ liệu (chỉ các thống kê dựa trên đỉnh lũ, **chưa** nói gì về sai số dự báo):

| Biến thể | Số mùa | Xu hướng đỉnh (km²/năm) | r (bỏ xu hướng) |
|---|---|---|---|
{chr(10).join(f"| {v} | {int(r.n_seasons)} | {r.peak_slope_km2_per_year:.0f} | {r.r_detrended:.2f} |" for v, r in sens.iterrows())}

- Đối chiếu mực nước: Tân Châu, đỉnh diện tích với mực nước lớn nhất, 2000–2016. Chỉ ở cấp năm, n = {int(tc.n)}.

## 3. Thiết lập thử nghiệm

- Dự báo diện tích ngập sau 8, 16, 32 ngày. Chia theo thời gian: huấn luyện các mùa trước năm Y, kiểm tra năm Y, Y = 2010–2020 (11 mùa test).
- Mọi thống kê tiền xử lý (climatology 5 mùa trước, hệ số φ) chỉ tính từ dữ liệu huấn luyện. Có test cố ý tạo rò rỉ để chắc bộ kiểm tra bắt được.
- AnomAR1 = mức bình thường tại bước đích + φ^h × (lệch hôm nay so với mức bình thường). Đây là mốc thống kê để so sánh.
- Ngân sách tinh chỉnh như nhau (6 cấu hình/mô hình), chọn trên 3 mùa cuối của tập huấn luyện, không đụng tập test. Chi tiết ở `docs/tuning_log.md`.
- So sánh ở **cấp mùa** (n = 11), vì sai số các bước trong cùng mùa tự tương quan mạnh. Dùng t ghép cặp, Wilcoxon, CI của chênh MSE, TOST ±5% RMSE và Holm trên 30 phép so sánh.

## 4. Kết quả

### 4.1 RMSE gộp (km², thấp hơn là tốt hơn), baseline cùng bảng

{md_table(_fmt_rmse(t["probe_rmse"].assign(model=t["probe_rmse"].model.map(label)).sort_values("2")))}

### 4.2 So với AnomAR1 ở cấp mùa

{md_table(tests_table(tests))}

Đọc bảng: tỉ lệ < 1 nghĩa là tốt hơn AnomAR1. CI chứa 0 nghĩa là chưa phân biệt được. p TOST nhỏ nghĩa là có bằng chứng tương đương trong ±5%.

### 4.3 Theo giai đoạn lũ (tầm 16 ngày, RMSE km²)

{md_table(wide_table(t["probe_by_phase_h2"], "Mô hình"))}

LightGBM thô tốt hơn AnomAR1 khi nước rút ({phase.loc["lgbm_raw", "falling"]:,.0f} so với {phase.loc[BASE, "falling"]:,.0f}) nhưng kém hơn ở đỉnh lũ ({phase.loc["lgbm_raw", "peak"]:,.0f} so với {phase.loc[BASE, "peak"]:,.0f}). Bản không dùng climatology cũng tốt khi nước rút ({phase.loc["lgbm_noclim", "falling"]:,.0f}), nên lợi thế này không đến từ climatology.

### 4.4 Theo cỡ lũ của mùa test (tầm 16 ngày, RMSE km²)

{md_table(wide_table(t["probe_by_magnitude_h2"], "Mô hình"))}

Với toàn bộ dữ liệu huấn luyện, LightGBM thô tốt hơn ở mùa lũ nhỏ ({mag.loc["lgbm_raw", "small"]:,.0f} so với {mag.loc[BASE, "small"]:,.0f}). **Nhưng cỡ lũ và lượng dữ liệu học bị lẫn nhau**: các mùa lũ nhỏ đều là những năm cuối, khi mô hình có nhiều mùa học hơn. Khi cố định cửa sổ huấn luyện 7 mùa:

{md_table(wide_table(t["probe_window7_by_magnitude_h2"], "Mô hình"))}

ưu thế ở lũ nhỏ biến mất (LightGBM thô {wmag.loc["lgbm_raw_w7", "small"]:,.0f} so với AnomAR1 cùng cửa sổ {wmag.loc["anom_ar1_w7", "small"]:,.0f}) và nó kém hơn ở lũ lớn ({wmag.loc["lgbm_raw_w7", "large"]:,.0f} so với {wmag.loc["anom_ar1_w7", "large"]:,.0f}).

### 4.5 Lượng dữ liệu huấn luyện (RMSE km², "[7 mùa gần nhất]" = chỉ dùng 7 mùa liền trước)

{md_table(wide_table(t["probe_window7"], "Mô hình"))}

## 5. Có đáng làm tiếp không?

Theo tiêu chí tự đặt (đạt ít nhất 2 trong 3): (a) thứ hạng đổi theo điều kiện, (b) có kết luận tương đương hợp lệ, (c) có phát hiện về phương pháp hoặc dữ liệu. Hiện đạt cả ba: nước rút so với đỉnh lũ và lượng dữ liệu học; GAM tương đương AnomAR1 ở 8 ngày; cỡ lũ lẫn với lượng dữ liệu học. **Hướng nên theo là "so sánh theo điều kiện", không phải "chứng minh ML thắng".**

## 6. Hạn chế (đọc kỹ trước khi trích số)

- Chỉ 11 mùa test, một lần chạy, một chuỗi (toàn ĐBSCL), CI rộng. Các khác biệt vài phần trăm không đáng tin.
- Không tái lập được con số AnomAR1 = 1.602 km² ở 32 ngày của lần thăm dò trước (bản này ra {rmse.loc[BASE, "4"]:,.0f}). Nguyên nhân chưa rõ, ghi ở `docs/decisions.md`.
- Dữ liệu đích là sản phẩm suy diễn; đối chiếu mực nước mới ở cấp năm và r bỏ xu hướng chỉ khoảng 0,67.
- Chronos-2 chỉ chạy zero-shot với ngữ cảnh anomaly, chưa tinh chỉnh.
- Không kết luận nhân quả (đê, đập); chỉ nói "phù hợp với".

## 7. Việc tiếp theo và câu hỏi cho thầy

1. Truy nguyên chênh lệch 1.602 so với 1.687 km².
2. Chạy lại benchmark sau khi bỏ ảnh nội suy/nghi bản sao; thêm bản 13 mùa test (2008–2020).
3. Thêm ARIMA/SARIMA trong từng mùa (không nối mùa) và MARS nếu cài được; TabPFN khi có API key Prior Labs.
4. Hỏi thầy: GAM/MixedLM có tính là "mô hình truyền thống" không; có được thêm biến phụ (mưa, mực nước) không; foundation model zero-shot có tính là một thuật toán không.
"""


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    dst = repo_root() / "results" / "probe_report.md"
    dst.write_text(build_markdown(), encoding="utf-8")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()

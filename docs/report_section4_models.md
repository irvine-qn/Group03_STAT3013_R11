# Mục 4: Các mô hình – Formula, Assumptions, Hypothesis test

Thứ tự: GAM → Chronos-2 → LightGBM → AnomAR1. Mọi công thức là đúng những gì code tính (trích dẫn kiểu IEEE, danh mục ở cuối); mỗi ý có ghi file/hàm
tương ứng. Số liệu lấy từ `results/`.

## Ký hiệu và công thức chung

**Dữ liệu.** $y_{s,t}$ là diện tích ngập (km²) của mùa $s$ tại bước $t$, mỗi bước 8 ngày, $t = 1,\dots,27$, suy ra từ ảnh MODIS tổ hợp 8 ngày theo hướng tiếp cận của [18].

**Mức bình thường (climatology)** – `climatology`: trung bình cùng bước của tối đa 5 mùa liền trước (cần ít nhất 3 mùa):

$$
c_{s,t} = \frac{1}{K_s}\sum_{k=1}^{K_s} y_{s-k,\,t}, \qquad K_s = \min(5,\ s - s_0),\quad K_s \ge 3
$$

**Độ lệch (anomaly):**

$$
a_{s,t} = y_{s,t} - c_{s,t}
$$

**Tầm và gốc dự báo** – `forecast_origins`: $h \in \{1, 2, 4\}$ (8, 16, 32 ngày); gốc $t = 3, \dots, 27 - h$.
Ngày trong năm của bước đích:

$$
\text{doy}_{t+h} = \text{first\_doy} + 8\,(t + h - 1)
$$

**Đánh giá theo rolling origin** [16]: 11 mùa kiểm tra 2010–2020; mùa kiểm tra $s^{\star}$ chỉ học từ các mùa $s < s^{\star}$ (`training_seasons`).

**Chọn cấu hình** (GAM, LightGBM – `pick_config`): với mỗi $s^{\star}$ và mỗi $h$, học từng cấu hình $\theta$ trên tập huấn luyện
trừ 3 mùa cuối $\mathcal{V}$, rồi chọn

$$
\hat\theta = \arg\min_{\theta}\ \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} \big(y_i - \hat y_i(\theta)\big)^2
$$

và học lại $\hat\theta$ trên toàn bộ tập huấn luyện.

**Thước đo sai số** [17] (`pooled_rmse`, `r2_all`), với $N_h$ dự báo ở tầm $h$:

$$
\text{RMSE}_h = \sqrt{\frac{1}{N_h}\sum_{i=1}^{N_h}(y_i - \hat y_i)^2},
\qquad
R^2_h = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}
$$

### Khung kiểm định so với AnomAR1 (dùng chung cho cả 4 mô hình)

Code: `versus_baseline` (GAM), `compare_with_anom_ar1` (Chronos-2, LightGBM, AnomAR1, Comparison).
Cách so sánh độ chính xác dự báo của hai mô hình trên cùng tập kiểm tra theo tinh thần Diebold–Mariano [15]; đơn vị quan sát là **mùa**. Với mô hình $M$ và mốc $B$ = AnomAR1, mỗi mùa kiểm tra $s = 1,\dots,11$ có

$$
\text{MSE}_{M,s} = \frac{1}{n_s}\sum_{t}\big(y_{s,t+h} - \hat y^{M}_{s,t+h}\big)^2
$$

**(1) Chỉ số mô tả**

$$
\text{rmse\_ratio} = \sqrt{\frac{\overline{\text{MSE}}_M}{\overline{\text{MSE}}_B}},
\qquad
\text{wins} = \#\{s : \text{MSE}_{M,s} < \text{MSE}_{B,s}\}
$$

**(2) t-test cặp** (`scipy.stats.ttest_rel`), $d_s = \text{MSE}_{M,s} - \text{MSE}_{B,s}$:

$$
H_0: \mathbb{E}[d_s] = 0 \quad\text{vs}\quad H_1: \mathbb{E}[d_s] \ne 0,
\qquad
T = \frac{\bar d}{s_d/\sqrt{11}} \sim t_{10}
$$

**(3) Wilcoxon signed-rank** [12] (`scipy.stats.wilcoxon`): $H_0$: phân phối của $d_s$ đối xứng quanh 0 (không cần giả định chuẩn).

**(4) Khoảng tin cậy 95% của chênh MSE tương đối** (`_relative_mse_diff`):

$$
r_s = \frac{\text{MSE}_{M,s} - \text{MSE}_{B,s}}{\text{MSE}_{B,s}},
\qquad
100\cdot\Big(\bar r \pm t_{0.975,\,10}\ \frac{s_r}{\sqrt{11}}\Big)
$$

**(5) TOST – kiểm định tương đương, biên ±5% RMSE** [13] (`_tost_p`):

$$
g_s = \tfrac12 \ln\frac{\text{MSE}_{M,s}}{\text{MSE}_{B,s}} \;(=\ln \text{ tỉ lệ RMSE}),
\qquad \delta = \ln(1.05)
$$

$$
H_0: \lvert \mathbb{E}[g_s] \rvert \ge \delta \quad\text{vs}\quad H_1: \lvert \mathbb{E}[g_s] \rvert < \delta
$$

$$
p_{\text{TOST}} = \max\Big( P\big(t_{10} > \tfrac{\bar g + \delta}{\text{SE}}\big),\ P\big(t_{10} < \tfrac{\bar g - \delta}{\text{SE}}\big) \Big),
\qquad \text{SE} = \frac{s_g}{\sqrt{11}}
$$

**(6) Hiệu chỉnh Holm–Bonferroni** [14] (`holm`): sắp $p_{(1)} \le \dots \le p_{(m)}$,

$$
\tilde p_{(k)} = \min\Big(1,\ \max_{j \le k}\,(m - j + 1)\,p_{(j)}\Big)
$$

Họ chính: 3 mô hình × 3 tầm, $m = 9$.

---

## 4.2. GAM

Code: `src/GAM/gam_model.py`, `gam_protocol.py`, `gam_run_model.py`, `gam_analysis.py`.

### 4.2.1. Formula

GAM [1] học độ lệch tại bước đích (`anomaly_target`) từ ma trận $[\text{doy}_{t+h},\ a_{s,t},\ a_{s,t-1}]$ (`design_matrix`):

$$
\hat a_{s,t+h} = \beta_0 + f(\text{doy}_{t+h}) + \beta_1\, a_{s,t} + \beta_2\, a_{s,t-1}
$$

$$
\hat y_{s,t+h} = c_{s,t+h} + \hat a_{s,t+h}
$$

Hàm trơn $f$ là P-spline (B-spline có phạt) [2] với $K$ = `n_splines` hàm cơ sở:

$$
f(x) = \sum_{j=1}^{K} \theta_j B_j(x)
$$

Các tham số được ước lượng bằng bình phương tối thiểu có phạt độ cong [2], [3], $\lambda$ = `lam`:

$$
\min_{\beta,\theta}\ \sum_i \big(a_{i,t+h} - \hat a_{i,t+h}\big)^2 + \lambda\, \boldsymbol\theta^{\top} \mathbf{P}\, \boldsymbol\theta
$$

trong đó $\mathbf P = \mathbf D^{\top}\mathbf D$ là ma trận phạt mặc định của `pygam` cho `s(...)` (sai phân bậc hai của các hệ số
spline, xấp xỉ $\int f''(x)^2\,dx$). Hai số hạng `l(1)`, `l(2)` là tuyến tính, không bị phạt độ cong.

- Cài đặt (`fit_gam`): `LinearGAM(s(0, n_splines, lam) + l(1) + l(2))`, mỗi tầm $h$ một mô hình.
- Lưới `GAM_GRID` $(K, \lambda)$: (6; 0,1), (6; 1), (8; 0,6), (8; 5), (12; 1), (12; 10). Cấu hình (6; 1) được chọn 16/33 lần.
- Quan hệ với mốc: AnomAR1 là trường hợp riêng khi $\beta_0 = 0,\ f \equiv 0,\ \beta_2 = 0,\ \beta_1 = \varphi^h$.
- Biến thể `gam_w7`: chỉ học 7 mùa gần nhất.

**Ví dụ** (`contributions`, mùa 2016, gốc ngày 233, $h = 2$):

$$
\hat y = \underbrace{5993}_{c_{t+h}} \underbrace{-\,83}_{\beta_0} \underbrace{-\,583}_{f(249)} \underbrace{-\,1311}_{\beta_1 a_t} \underbrace{+\,68}_{\beta_2 a_{t-1}} = 4084 \text{ km}^2 \quad(\text{thật: } 3803)
$$

Hệ số học được (`gam_coefficients.csv`): $\beta_1$ trung bình 0,89 / 0,74 / 0,62 (8 / 16 / 32 ngày), sát $\varphi^h$ của AnomAR1;
$\beta_2 \approx$ −0,05 / −0,05 / −0,10; bậc tự do hiệu dụng của $f \approx$ 3,7–4,2; $R^2$ giả 0,74 / 0,53 / 0,26.

### 4.2.2. Assumptions

| Giả định | Kiểm tra trong code | Kết quả |
|---|---|---|
| Sai số có phân phối chuẩn, $\varepsilon \sim \mathcal N(0,\sigma^2)$ (giả định của `LinearGAM`) | Shapiro–Wilk [11] trên sai số dự báo (`residual_tables`) | $p$ = 0,08 / 0,43 / 0,44 → không bác bỏ chuẩn |
| Phần dư không còn cấu trúc ngoài tự tương quan do dự báo đa bước | ACF phần dư (`acf`) so với ACF lý thuyết của AR(1) (`theoretical_acf`, $\varphi$ gộp từ `pooled_phi`); Ljung–Box độ trễ 10 [10] (`ljung_box_p`) | ACF quan sát thấp hơn lý thuyết → không bỏ sót cấu trúc |
| Mức bình thường bị lệch theo mùa vụ, $f$ sửa được | `climatology_bias`; phần dư trung bình theo ngày đích | Lệch thật −880 đến −1.030 km² sau ngày ~230; phần dư GAM quanh 0; thiên lệch +34 / +67 / +105 km² |
| Tính cộng, $\beta_1,\beta_2$ hằng số trong mùa | Bóc tách `ablation_table` (phân tích phụ) | Bản chỉ $\beta_0 + \beta_1 a_t$: RMSE 1.080 / 1.337 / 1.635 |
| Không rò rỉ tương lai | `tests/GAM/test_gam_protocol.py` | đạt |

Công thức các kiểm tra trong `gam_analysis.py` (ACF và thống kê Ljung–Box [10], [23]):

$$
\hat\rho_k = \frac{\sum_{t=1}^{n-k}(e_t - \bar e)(e_{t+k} - \bar e)}{\sum_{t=1}^{n}(e_t - \bar e)^2},
\qquad
Q_{LB} = n(n+2)\sum_{k=1}^{10}\frac{\hat\rho_k^2}{n-k} \sim \chi^2_{10}
$$

ACF lý thuyết của sai số dự báo $h$ bước khi độ lệch là AR(1) (`theoretical_acf`):

$$
\rho_k^{\text{theory}} = \frac{\sum_{j=0}^{h-1-k} \varphi^{2j+k}}{\sum_{j=0}^{h-1} \varphi^{2j}}\quad (k < h),\qquad 0 \ (k \ge h)
$$

### 4.2.3. Hypothesis test

Theo khung chung, so `gam` với `anom_ar1` (`results/Comparison/tables/vs_anom_ar1_main.csv`):

| Tầm | Tỉ lệ RMSE | Mùa thắng | $p_t$ | $p_{\text{Wilcoxon}}$ | $p_{\text{TOST}}$ | $p_{\text{Holm}}$ (9 phép) |
|---|--:|--:|--:|--:|--:|--:|
| 8 ngày | 1,006 | 5/11 | 0,508 | 0,765 | **0,002** | 1 |
| 16 ngày | 0,997 | 7/11 | 0,914 | 0,638 | 0,053 | 1 |
| 32 ngày | 1,001 | 7/11 | 0,990 | 0,700 | 0,253 | 1 |

- Không bác bỏ $H_0: \mathbb E[d_s] = 0$ ở tầm nào: GAM không khác AnomAR1 có ý nghĩa.
- Ở 8 ngày TOST bác bỏ $H_0: \lvert\mathbb E[g_s]\rvert \ge \ln 1.05$ ($p$ = 0,002): GAM **tương đương** AnomAR1 trong biên ±5% RMSE.
- Ở 16 và 32 ngày chưa đủ bằng chứng để nói khác hay tương đương.

---

## 4.3. Chronos-2

Code: `src/Chronos-2/chronos2_model.py`, `chronos2_protocol.py`, `chronos2_forecast.py`, `chronos2_evaluate.py`;
phụ: `chronos2_finetune_check.py`.

### 4.3.1. Formula

Chronos-2 (`MODEL_ID = "amazon/chronos-2"`) [5] là mô hình nền huấn luyện sẵn, phiên bản kế tiếp của Chronos [4], dùng **zero-shot** (không học thêm).
Ngữ cảnh là chuỗi độ lệch của chính mùa đang dự báo, từ bước 1 đến gốc $t$ (`build_anomaly_context`):

$$
\mathbf{x}_{s,t} = \big(a_{s,1},\ a_{s,2},\ \dots,\ a_{s,t}\big)
$$

Mô hình trả về phân vị 0,5 (trung vị, `MEDIAN = 0.5`) cho 4 bước tới
(`predict_quantiles(..., prediction_length=4, quantile_levels=[0.5])`):

$$
\hat a_{s,t+k} = Q_{0.5}\big(a_{s,t+k} \mid \mathbf{x}_{s,t}\big), \qquad k = 1, 2, 3, 4
$$

$$
\hat y_{s,t+h} = c_{s,t+h} + \hat a_{s,t+h}, \qquad h \in \{1, 2, 4\}
$$

- Mỗi gốc gọi mô hình một lần, xin 4 bước; tầm $h$ lấy bước thứ $h$.
- Không có siêu tham số, không chọn cấu hình.
- Cùng thông tin đầu vào với AnomAR1 (chu kỳ mùa nằm trong $c$, phần còn lại là độ lệch của mùa hiện tại).
  Thư mục tự tính lại AnomAR1 (`estimate_phi`, `anom_ar1_forecast`) để đối chiếu.

### 4.3.2. Assumptions

| Giả định | Kiểm tra trong code | Kết quả |
|---|---|---|
| Mẫu hình học sẵn chuyển giao được sang chuỗi độ lệch ĐBSCL mà không cần học thêm | So với AnomAR1; thử fine-tune (`chronos2_finetune_check.py`, thăm dò) | Fine-tune toàn bộ (50 bước, lr $10^{-6}$): 32 ngày 1.758 so với 1.788 zero-shot, không đổi kết luận |
| Ngữ cảnh trong mùa (3–26 bước) là đủ | RMSE theo gốc $t$ (`rmse_by_origin.csv`) và theo phần mùa (`add_season_part`, `LATE_ORIGIN = 21`) | 32 ngày: $t \le 20$: 1.703 so với 1.687; $t \ge 21$: **2.236** so với 1.689 → yếu ở cuối mùa |
| Trung vị là dự báo điểm phù hợp | (lựa chọn thiết kế) | 15 dự báo âm, nhiều nhất trong các mô hình |
| Tính tất định, tái lập được | Chạy lại CPU/GPU | CPU khớp tới $10^{-12}$ km²; GPU lệch tối đa 0,002 km² |
| Không rò rỉ tương lai | `test_chronos2_protocol.py`, `test_chronos2_model.py` (thay tương lai bằng $10^9$) | đạt |

### 4.3.3. Hypothesis test

Theo khung chung, so `chronos2` với `anom_ar1`:

| Tầm | Tỉ lệ RMSE | Mùa thắng | Chênh MSE % (CI 95%) | $p_t$ | $p_{\text{Wilcoxon}}$ | $p_{\text{TOST}}$ | $p_{\text{Holm}}$ (9 phép) |
|---|--:|--:|--:|--:|--:|--:|--:|
| 8 ngày | 1,024 | 4/11 | +7,7 (−1,0; +16,5) | 0,303 | 0,175 | 0,216 | 1 |
| 16 ngày | 1,027 | 2/11 | +7,2 (−2,4; +16,8) | 0,346 | 0,175 | 0,211 | 1 |
| 32 ngày | 1,060 | 2/11 | +14,3 (**+4,0; +24,5**) | 0,005 | 0,010 | 0,748 | **0,045** |

- 32 ngày: bác bỏ $H_0$ cả trước và sau Holm ($0.0050 \times 9 = 0.045 < 0.05$); Wilcoxon đồng thuận; CI nằm hẳn trên 0
  → Chronos-2 **kém AnomAR1 có ý nghĩa**. Đây là phép so duy nhất trong 9 phép có ý nghĩa.
- 8 và 16 ngày: không bác bỏ $H_0$; TOST cũng không chứng minh được tương đương.

---

## 4.4. LightGBM

Code: `src/LightGBM/lightgbm_model.py`, `lightgbm_protocol.py`, `lightgbm_forecast.py`, `lightgbm_evaluate.py`, `lightgbm_diag.py`.

### 4.4.1. Formula

Gradient boosting [6] trên cây quyết định, cài đặt bằng LightGBM [7] (`lgb.LGBMRegressor`, mất mát bình phương):

$$
F_0 = \bar z, \qquad
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\, T_m(\mathbf{x}), \quad m = 1,\dots,M
$$

Mỗi cây $T_m$ (tối đa `num_leaves` lá) khớp phần dư của bước trước:

$$
r_i^{(m)} = z_i - F_{m-1}(\mathbf{x}_i), \qquad
T_m = \arg\min_{T} \sum_{i \in \mathcal{S}_m} \big(r_i^{(m)} - T(\mathbf{x}_i)\big)^2
$$

với $M$ = `n_estimators`, $\eta$ = `learning_rate`, $\mathcal S_m$ là mẫu con 80% (`subsample=0.8`, `subsample_freq=1`),
mỗi lá tối thiểu 10 mẫu (`min_child_samples=10`), `random_state = 0`.

Ba biến thể khai báo trước (`DESIGN`, `make_model`), mỗi tầm $h$ một mô hình:

| Biến thể | Đặc trưng $\mathbf{x}$ | Mục tiêu $z$ | Dự báo |
|---|---|---|---|
| `lgbm_raw` (**bản chính**) | $y_t,\ y_{t-1},\ y_{t-2},\ c_t,\ c_{t+h},\ \text{doy}_{t+h}$ | $y_{t+h}$ | $\hat y = F_M(\mathbf{x})$ |
| `lgbm_anomaly` | $a_t,\ a_{t-1},\ \text{doy}_{t+h}$ | $a_{t+h}$ | $\hat y = c_{t+h} + F_M(\mathbf{x})$ |
| `lgbm_noclim` | $y_t,\ y_{t-1},\ y_{t-2},\ \text{doy}_{t+h}$ | $y_{t+h}$ | $\hat y = F_M(\mathbf{x})$ |

- Lưới `LGBM_GRID` (num_leaves, $M$, $\eta$): (4, 200, 0,05), (4, 100, 0,05), (8, 200, 0,05), (4, 400, 0,03), (2, 300, 0,05), (8, 100, 0,1).
- Cấu hình chọn cho `lgbm_raw` (33 lần): (2, 300, 0,05) 12 lần, (4, 400, 0,03) 8, (4, 200, 0,05) 7, (4, 100, 0,05) 6.
- Đối chứng phụ: Random Forest [9] cùng đặc trưng (`RF_GRID`, `fixed_config_table`, $h = 2$).

### 4.4.2. Assumptions

LightGBM không giả định phân phối sai số hay dạng hàm. Các giả định thực chất và cách code kiểm tra:

| Giả định | Kiểm tra trong code | Kết quả |
|---|---|---|
| Kết quả không phụ thuộc may rủi của seed | 10 seed (`run(seeds=range(10))`, `seed_summary`) | Độ lệch chuẩn RMSE 0,4–1,7%; `lgbm_raw` < AnomAR1 ở 10/10 seed tại 8 và 16 ngày, 7/10 tại 32 ngày |
| Quan hệ học được hợp lý | SHAP [8] (`importance_of`), gain, permutation importance 30 lần (`permutation_importance`) | 8 ngày dựa vào $y_t$ (+1.871 km² khi xáo trộn); 32 ngày dựa vào $c_{t+h}$ (+1.455), $y_t$ chỉ +179 |
| Mùa kiểm tra nằm trong miền giá trị đã học (cây không ngoại suy) | Dự báo từng mùa (`season_forecasts`) | Vi phạm ở 2011 (lũ to bất thường): 32 ngày 1.956 so với 1.248; không có dự báo âm |
| Lợi thế đều giữa các mùa | `season_phase_table`, `gain_concentration` ($h = 2$) | Nước lên thắng 8/11 mùa; nước rút thắng 7/11 nhưng 3 mùa chiếm ~74% mức giảm sai số |
| Không rò rỉ tương lai | `tests/LightGBM/test_lightgbm_pipeline.py` | đạt |

Permutation importance [9] của đặc trưng $j$ (`permutation_importance`, $R = 30$ lần xáo trộn):

$$
\text{PI}_j = \frac{1}{R}\sum_{r=1}^{R} \text{RMSE}\big(\mathbf{X}^{(j,r)}_{\text{perm}}\big) - \text{RMSE}(\mathbf{X})
$$

### 4.4.3. Hypothesis test

Theo khung chung, so `lgbm_raw` với `anom_ar1`:

| Tầm | Tỉ lệ RMSE | Mùa thắng | Chênh MSE % (CI 95%) | $p_t$ | $p_{\text{Wilcoxon}}$ | $p_{\text{TOST}}$ | $p_{\text{Holm}}$ (9 phép) |
|---|--:|--:|--:|--:|--:|--:|--:|
| 8 ngày | 0,965 | 5/11 | −4,9 (−18,7; +8,9) | 0,408 | 0,638 | 0,363 | 1 |
| 16 ngày | 0,934 | **9/11** | −5,3 (−27,0; +16,4) | 0,140 | 0,102 | 0,511 | 1 |
| 32 ngày | 1,001 | 5/11 | +11,8 (−23,9; +47,5) | 0,989 | 0,898 | 0,293 | 1 |

- Không bác bỏ $H_0$ ở tầm nào, kể cả trước hiệu chỉnh. CI đều chứa 0; TOST không chứng minh tương đương.
- RMSE thấp nhất ở 8 và 16 ngày (−3,5%, −6,6%), thắng 9/11 mùa ở 16 ngày: là tín hiệu, chưa phải kết luận thống kê.
- Biến thể (Holm trong thư mục LightGBM, 9 phép): `lgbm_anomaly` 1.101 / 1.364 / 1.699 km², `lgbm_noclim` 1.059 / 1.301 / 1.778 km²,
  đều không khác AnomAR1 có ý nghĩa; `lgbm_noclim` kém rõ ở 32 ngày → tầm xa cần mức bình thường.

---

## 4.5. AnomAR1

Code: `src/AnomAR1/anomar1_protocol.py`, `anomar1_forecast.py`, `anomar1_evaluate.py`, `anomar1_diagnostics.py`, `anomar1_eda.py`.

### 4.5.1. Formula

Dự báo (`anom_ar1`):

$$
\hat y_{s,t+h} = c_{s,t+h} + \varphi^{h}\, a_{s,t}
$$

xuất phát từ giả định độ lệch là AR(1) không hằng số, tức ARIMA(1,0,0) trên chuỗi đã khử mùa [23]:

$$
a_{s,t} = \varphi\, a_{s,t-1} + \varepsilon_{s,t}
\quad\Longrightarrow\quad
\mathbb{E}\big[a_{s,t+h} \mid a_{s,t}\big] = \varphi^{h} a_{s,t}
$$

$\varphi$ ước lượng bằng hồi quy qua gốc trên các cặp $(a_{t-1}, a_t)$ của mùa huấn luyện (`regression_xy`, `estimate_phi`),
ước lượng lại cho mỗi mùa kiểm tra và mỗi tầm:

$$
\hat\varphi = \frac{\sum_{i} a_{i,t-1}\, a_{i,t}}{\sum_{i} a_{i,t-1}^2},
\qquad
\text{SE}(\hat\varphi) = \frac{\hat\sigma}{\sqrt{\sum_i a_{i,t-1}^2}},
\quad
\hat\sigma^2 = \frac{\sum_i (a_{i,t} - \hat\varphi a_{i,t-1})^2}{n - 1}
$$

`phi_by_fold.csv` ghi $\hat\varphi$, `phi_se` và `phi_statsmodels` (khớp $\hat\varphi$ tới $10^{-15}$).
Không có siêu tham số. Biến thể `anom_ar1_w7` chỉ học 7 mùa gần nhất (`WINDOW = 7`).

Hai mốc ngây thơ (`persistence`, `clim5`):

$$
\hat y^{\text{Persistence}}_{s,t+h} = y_{s,t} \;(\varphi = 1),
\qquad
\hat y^{\text{Clim5}}_{s,t+h} = c_{s,t+h} \;(\varphi = 0)
$$

**Ví dụ** (mùa 2016, $h = 2$): $\hat\varphi = 0.9237$ (299 cặp, 2003–2015), $\hat\varphi^2 = 0.8533$:

$$
\hat y = 5992.5 + 0.8533 \times (-1694.3) = 4546.8 \text{ km}^2
$$

### 4.5.2. Assumptions

| Giả định | Kiểm tra trong code | Kết quả |
|---|---|---|
| Độ lệch là AR(1) dừng, $\lvert\varphi\rvert < 1$, ổn định | `phi_by_fold.csv` ($\hat\varphi$, SE theo fold) | $\hat\varphi$ = 0,853–0,898 (8 ngày), 0,884–0,936 (16), 0,896–0,933 (32); SE ≈ 0,03–0,05 |
| $\varphi^h$ đúng cho dự báo đa bước | `direct_phi`, `compare_direct_phi`: hồi quy trực tiếp $a_{t+h}$ theo $a_t$ | RMSE 1.088 / 1.356 / 1.675 so với 1.090 / 1.375 / 1.687 → gần như không đổi |
| $\mathbb E[a] = 0$ (mức bình thường không thiên lệch) | `residual_tests`, `summarize_residuals` | Vi phạm một phần: thiên lệch −84 / −130 / −302 km² (đỉnh lũ giảm dần) |
| Sai số gần chuẩn | Q–Q plot (`plot_diagnostics`) | Gần thẳng, lệch nhẹ ở hai đuôi |
| Nhiễu trắng trong mùa | ACF (`statsmodels.acf`) và Ljung–Box độ trễ 10 [10] (`acorr_ljungbox`) từng mùa | ACF bậc 1: −0,14 / +0,36 / +0,48; số mùa $p < 0.05$: 0 / 4 / 7 trên 11 → tự tương quan do dự báo đa bước chồng lấn, lý do kiểm định ở cấp mùa |
| Có xu hướng giảm nên phải dùng mức bình thường **gần** | `anomar1_eda.py`: hồi quy đỉnh theo năm, Mann–Kendall [21], Mann–Whitney [25], Levene, Kruskal–Wallis [22] | Đỉnh giảm 314 km²/năm ($p \approx 7\cdot10^{-14}$, $R^2 = 0.77$); đỉnh 2010–2020 thấp hơn trước 2010 ($p$ = 0,001); Levene $p$ = 0,89; Kruskal–Wallis $p$ = 0,77 |

Hồi quy xu hướng đỉnh lũ trong `anomar1_eda.py`. Xu hướng giảm này phù hợp với các nghiên cứu về tác động của đê bao và hạ tầng thủy lợi lên chế độ lũ ĐBSCL [19], [20] (dữ liệu của nhóm không đủ để kết luận nhân quả):

$$
\text{peak}_s = \alpha + \gamma\cdot\text{year}_s + u_s, \qquad \hat\gamma = -314 \text{ km}^2/\text{năm}\ \ (\text{CI 95\%: } -396;\ -232)
$$

### 4.5.3. Hypothesis test

AnomAR1 là mốc, nên code kiểm định nó so với các mốc ngây thơ và biến thể của chính nó, theo khung chung
(`anomar1_evaluate.py`, Holm trên 9 phép trong thư mục AnomAR1, `results/AnomAR1/tables/vs_anom_ar1.csv`):

| Mô hình so với AnomAR1 | Tỉ lệ RMSE (8 / 16 / 32 ngày) | Số mùa mô hình này thắng AnomAR1 | $p_{\text{Holm}}$ |
|---|--:|--:|--:|
| Persistence | 1,20 / 1,50 / 2,05 | 1 / 0 / 0 | **0,030 / 0,002 / 0,0002** |
| Clim5 | 1,66 / 1,34 / 1,14 | 0 / 1 / 5 | 0,115 / 0,233 / 0,505 |
| AnomAR1 7 mùa (`anom_ar1_w7`) | 0,994 / 0,985 / 0,997 | 8 / 9 / 7 | 0,665 / 0,115 / 0,766; $p_{\text{TOST}} < 0.01$ ở cả 3 tầm |

- AnomAR1 tốt hơn Persistence có ý nghĩa ở cả 3 tầm.
- Tốt hơn Clim5 về số nhưng chưa có ý nghĩa sau Holm ($n = 11$; Clim5 thắng 5/11 mùa ở 32 ngày).
- Bản 7 mùa **tương đương** AnomAR1 trong ±5%.
- Kết luận: AnomAR1 là một mốc mạnh, phù hợp với kết quả cuộc thi M4 rằng phương pháp thống kê đơn giản thường khó bị vượt [24]; trong 4.2–4.4 nó đóng vai trò $B$ của mọi kiểm định.

---

## Tài liệu tham khảo

Ghi chú: phần lớn là tạp chí xếp hạng Q1 (SCImago). [4], [5], [7] là bài gốc của mô hình/thư viện được dùng (arXiv, NeurIPS); [12], [13] là bài gốc của kiểm định, đăng trên tạp chí đã đổi tên. Nên tra lại hạng Q của từng tạp chí trên scimagojr.com trước khi nộp.

[1] T. Hastie and R. Tibshirani, "Generalized additive models," *Statistical Science*, vol. 1, no. 3, pp. 297–318, 1986.

[2] P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines and penalties," *Statistical Science*, vol. 11, no. 2, pp. 89–121, 1996.

[3] S. N. Wood, "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models," *Journal of the Royal Statistical Society: Series B*, vol. 73, no. 1, pp. 3–36, 2011.

[4] A. F. Ansari *et al.*, "Chronos: Learning the language of time series," arXiv:2403.07815, 2024.

[5] A. F. Ansari *et al.*, "Chronos-2: From univariate to universal forecasting," arXiv:2510.15821, 2025.

[6] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[7] G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017, pp. 3146–3154.

[8] S. M. Lundberg *et al.*, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.

[9] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[10] G. M. Ljung and G. E. P. Box, "On a measure of lack of fit in time series models," *Biometrika*, vol. 65, no. 2, pp. 297–303, 1978.

[11] S. S. Shapiro and M. B. Wilk, "An analysis of variance test for normality (complete samples)," *Biometrika*, vol. 52, no. 3–4, pp. 591–611, 1965.

[12] F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.

[13] D. J. Schuirmann, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," *Journal of Pharmacokinetics and Biopharmaceutics*, vol. 15, no. 6, pp. 657–680, 1987.

[14] S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979.

[15] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *Journal of Business & Economic Statistics*, vol. 13, no. 3, pp. 253–263, 1995.

[16] L. J. Tashman, "Out-of-sample tests of forecasting accuracy: An analysis and review," *International Journal of Forecasting*, vol. 16, no. 4, pp. 437–450, 2000.

[17] R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *International Journal of Forecasting*, vol. 22, no. 4, pp. 679–688, 2006.

[18] T. Sakamoto *et al.*, "Detecting temporal changes in the extent of annual flooding within the Cambodia and the Vietnamese Mekong Delta from MODIS time-series imagery," *Remote Sensing of Environment*, vol. 109, no. 3, pp. 295–313, 2007, doi: 10.1016/j.rse.2007.01.011.

[19] N. V. K. Triet *et al.*, "Has dyke development in the Vietnamese Mekong Delta shifted flood hazard downstream?," *Hydrology and Earth System Sciences*, vol. 21, pp. 3991–4010, 2017, doi: 10.5194/hess-21-3991-2017.

[20] T. D. Dang, T. A. Cochrane, M. E. Arias, P. D. T. Van, and T. T. de Vries, "Hydrological alterations from water infrastructure development in the Mekong floodplains," *Hydrological Processes*, vol. 30, no. 21, pp. 3824–3838, 2016.

[21] H. B. Mann, "Nonparametric tests against trend," *Econometrica*, vol. 13, no. 3, pp. 245–259, 1945.

[22] W. H. Kruskal and W. A. Wallis, "Use of ranks in one-criterion variance analysis," *Journal of the American Statistical Association*, vol. 47, no. 260, pp. 583–621, 1952.

[23] G. E. P. Box and D. A. Pierce, "Distribution of residual autocorrelations in autoregressive-integrated moving average time series models," *Journal of the American Statistical Association*, vol. 65, no. 332, pp. 1509–1526, 1970.

[24] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods," *International Journal of Forecasting*, vol. 36, no. 1, pp. 54–74, 2020.

[25] H. B. Mann and D. R. Whitney, "On a test of whether one of two random variables is stochastically larger than the other," *The Annals of Mathematical Statistics*, vol. 18, no. 1, pp. 50–60, 1947.

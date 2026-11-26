"""Kiểm tra chẩn đoán LightGBM bằng các trường hợp có đáp án biết trước."""
import numpy as np  # mảng số
import pandas as pd  # bảng dữ liệu
import pytest  # khung chạy test; dùng thêm pytest.approx và parametrize

from floodbench.cv import rolling_origin  # sinh các fold rolling-origin
from floodbench.data.regions import load_region  # đọc cấu hình vùng
from floodbench.features import build_rows  # dựng bảng đặc trưng
# Các hàm cần kiểm tra, nhập từ lightgbm_diag.py (đường dẫn thêm trong conftest.py)
from lightgbm_diag import (FEATURES, choose_config, fit_model, gain_concentration,
                           importance_of, run_fold, seed_summary, season_phase_table)
# Mô hình chính thức của nhóm trong benchmark, dùng để đối chiếu với mã chẩn đoán
from floodbench.models.ml import LGBM_GRID, SEED, lgbm_anomaly, lgbm_noclim, lgbm_raw

# Cấu hình vùng Mekong, đọc một lần dùng cho mọi test
CFG = load_region("mekong")
# Từ điển: tên biến thể -> mô hình benchmark tương ứng
BENCHMARK = {"raw": lgbm_raw, "anomaly": lgbm_anomaly, "noclim": lgbm_noclim}


def synthetic_Y(seed=0):
    # Bộ sinh số ngẫu nhiên có seed để dữ liệu giả lập tái lập được
    rng = np.random.default_rng(seed)
    # 27 bước thời gian của một mùa lũ (0..26)
    steps = np.arange(27)
    # Ma trận diện tích ngập giả lập: 21 mùa x 27 bước, khởi tạo bằng 0
    Y = np.zeros((21, 27))
    # Dựng từng mùa
    for i in range(21):
        # Nhiễu AR(1) của mùa này, bắt đầu từ 0
        noise = np.zeros(27)
        # Mỗi bước nhiễu = 0,8 x nhiễu bước trước + nhiễu trắng (độ lệch chuẩn 150)
        for k in range(1, 27):
            noise[k] = 0.8 * noise[k - 1] + rng.normal(0, 150)
        # Hình chuông đỉnh ở bước 14 trên nền 2000; đỉnh giảm dần theo mùa (9000 - 150 x i); cộng nhiễu
        Y[i] = 2000 + (9000 - 150 * i) * np.exp(-((steps - 14) ** 2) / 40) + noise
    return Y


def one_fold(h=2, test=2012):
    # Dữ liệu giả lập
    Y = synthetic_Y()
    # Lấy đúng một fold, với mùa test là `test` (first_test = last_test)
    fold = next(iter(rolling_origin(CFG, first_test=test, last_test=test)))
    # Đổi năm thành chỉ số hàng trong Y
    idx = lambda ss: [s - CFG.first_year for s in ss]
    # Bảng đặc trưng của các mùa huấn luyện
    train = build_rows(Y, CFG, idx(fold.train_seasons), h)
    # Trả về: ma trận Y, fold, bảng huấn luyện và bảng test
    return Y, fold, train, build_rows(Y, CFG, idx([fold.test_season]), h)


# Chạy test này 3 lần, mỗi lần với một biến thể
@pytest.mark.parametrize("variant", ["raw", "anomaly", "noclim"])
def test_diag_reproduces_the_benchmark_prediction_exactly(variant):
    """Mã chẩn đoán phải cho đúng dự báo của mô hình nhóm khi dùng cùng seed."""
    # Lấy bảng huấn luyện và bảng test của một fold
    _, _, train, test = one_fold()
    # Chọn cấu hình bằng validation (tên `idx` ở đây là chỉ số cấu hình, không phải hàm idx ở trên)
    idx, _ = choose_config(train, variant, 2, SEED)
    # Huấn luyện bằng mã chẩn đoán với cấu hình vừa chọn
    model = fit_model(train, LGBM_GRID[idx], variant, SEED)
    # Nhập predict tại chỗ (chỉ test này cần)
    from lightgbm_diag import predict
    # Dự báo của mã chẩn đoán phải trùng dự báo của mô hình benchmark (sai lệch tuyệt đối <= 1e-9)
    np.testing.assert_allclose(predict(model, test, variant), BENCHMARK[variant](train, test, 2), rtol=0, atol=1e-9)


@pytest.mark.parametrize("variant", ["raw", "anomaly", "noclim"])
def test_importance_gain_shares_sum_to_one_and_shap_is_additive(variant):
    # Lấy dữ liệu một fold
    _, _, train, test = one_fold()
    # Huấn luyện với cấu hình số 1 trong lưới
    model = fit_model(train, LGBM_GRID[1], variant, SEED)
    # Tính độ quan trọng đặc trưng
    imp = importance_of(model, test, variant)
    # Tên đặc trưng phải đúng và đúng thứ tự
    assert list(imp["feature"]) == FEATURES[variant]
    # Các phần gain cộng lại bằng 1
    assert imp["gain_share"].sum() == pytest.approx(1.0)
    # SHAP trung bình tuyệt đối không thể âm
    assert (imp["mean_abs_shap"] >= 0).all()
    from lightgbm_diag import design
    # SHAP của từng dòng (cột cuối là giá trị nền)
    contrib = model.predict(design(test, variant), pred_contrib=True)
    # Tính cộng tính: tổng SHAP + giá trị nền của mỗi dòng phải bằng đúng dự báo của dòng đó
    np.testing.assert_allclose(contrib.sum(axis=1), model.predict(design(test, variant)), atol=1e-6)


def test_run_fold_row_counts_and_seed0_matches_benchmark():
    # Dữ liệu một fold
    Y, fold, train, test = one_fold()
    # Chạy hai biến thể x hai seed (0 và 1) ở một (fold, tầm)
    cfgs, imps, preds = run_fold(Y, CFG, fold, 2, variants=("raw", "noclim"), seeds=(0, 1))
    # Bảng cấu hình: 1 dòng cho mỗi (biến thể, seed) = 4 dòng
    assert len(cfgs) == 2 * 2  # biến thể x seed
    # Bảng độ quan trọng: mỗi (biến thể, seed) có số dòng bằng số đặc trưng của biến thể đó
    assert len(imps) == 2 * (len(FEATURES["raw"]) + len(FEATURES["noclim"]))
    # Bảng dự báo: mỗi (biến thể, seed) có một dòng cho mỗi dòng của mùa test
    assert len(preds) == 2 * 2 * len(test)
    # Lấy dự báo của lgbm_raw với seed 0
    got = preds[(preds["model"] == "lgbm_raw") & (preds["seed"] == 0)]["pred"].to_numpy()
    # Phải trùng dự báo của mô hình benchmark (seed 0 là seed mặc định)
    np.testing.assert_allclose(got, lgbm_raw(train, test, 2), atol=1e-9)
    # Chỉ số cấu hình được chọn phải nằm trong lưới
    assert set(cfgs["cfg_index"]) <= set(range(len(LGBM_GRID)))


def test_seed_summary_known_values():
    # Danh sách các bảng dự báo, mỗi seed một bảng
    rows = []
    # Seed 0 sai ±100, seed 1 sai ±200; với hai dòng sai ±err thì RMSE đúng bằng err
    for seed, err in ((0, 100.0), (1, 200.0)):
        rows.append(pd.DataFrame({"model": "lgbm_raw", "h": 1, "seed": seed, "season": 2010,
                                  "y_target": [1000.0, 1000.0], "pred": [1000.0 + err, 1000.0 - err]}))
    # Tổng hợp theo seed; RMSE của AnomAR1 giả định là 150
    out = seed_summary(pd.concat(rows), baseline_rmse={("lgbm_raw", 1): 150.0}, seed0=0)
    # Chỉ có một nhóm (mô hình, tầm) nên lấy dòng đầu
    r = out.iloc[0]
    # 2 seed, RMSE trung bình = (100 + 200) / 2 = 150
    assert r["n_seeds"] == 2 and r["rmse_mean"] == pytest.approx(150.0)
    # Độ lệch chuẩn mẫu (ddof=1) của [100, 200]
    assert r["rmse_sd"] == pytest.approx(np.std([100.0, 200.0], ddof=1))
    # Độ lệch chuẩn tính theo phần trăm của trung bình
    assert r["rmse_sd_pct"] == pytest.approx(100 * np.std([100.0, 200.0], ddof=1) / 150.0)
    # Nhỏ nhất, lớn nhất và RMSE của seed 0
    assert r["rmse_min"] == 100.0 and r["rmse_max"] == 200.0 and r["rmse_seed0"] == 100.0
    # Chỉ 1 trong 2 seed (RMSE 100) tốt hơn mốc 150 nên tỉ lệ là 0,5
    assert r["frac_seeds_better_than_base"] == pytest.approx(0.5)  # chỉ seed 0 (100 < 150)
    # Trung bình 150 chia mốc 150 = 1
    assert r["mean_over_base"] == pytest.approx(1.0)


def _phase_df():
    # hai mùa ở pha nước rút (doy 300): mùa A mô hình tốt hơn (10 so với 20), mùa B kém hơn (30 so với 20)
    # Tạo khối dữ liệu 4 dòng cho một (mô hình, mùa), mọi dòng đều sai đúng `err`
    def block(model, season, err):
        return pd.DataFrame({"model": model, "h": 2, "season": season, "doy_target": 300,
                             "y_target": [1000.0] * 4, "pred": [1000.0 + err] * 4})
    # Ghép 4 khối: (lgbm_raw, anom_ar1) x (2010, 2011)
    return pd.concat([block("lgbm_raw", 2010, 10.0), block("anom_ar1", 2010, 20.0),
                      block("lgbm_raw", 2011, 30.0), block("anom_ar1", 2011, 20.0)])


def test_season_phase_table_known_values():
    # Bảng theo (giai đoạn, mùa) của lgbm_raw so với anom_ar1 ở tầm 2
    t = season_phase_table(_phase_df(), "lgbm_raw", "anom_ar1", h=2)
    # Ngày 300 thuộc giai đoạn "falling"; đặt mùa làm chỉ mục để tra cứu
    f = t[t["phase"] == "falling"].set_index("season")
    # Mùa 2010: RMSE 10 / 20 = 0,5 và mô hình tốt hơn
    assert f.loc[2010, "ratio"] == pytest.approx(0.5) and bool(f.loc[2010, "model_better"])
    # Mùa 2011: RMSE 30 / 20 = 1,5 và mô hình kém hơn
    assert f.loc[2011, "ratio"] == pytest.approx(1.5) and not bool(f.loc[2011, "model_better"])
    # Mức giảm sai số bình phương = (MSE đối chứng - MSE mô hình) x số dòng = (400 - 100) x 4
    assert f.loc[2010, "sse_gain"] == pytest.approx((400 - 100) * 4)
    # Mùa 2011 âm vì mô hình kém hơn: (400 - 900) x 4
    assert f.loc[2011, "sse_gain"] == pytest.approx((400 - 900) * 4)


def test_gain_concentration_known_values():
    # Bốn mùa với mức giảm sai số 60, 30, 20 và -10 (mùa 4 mô hình kém hơn)
    t = pd.DataFrame({"phase": "falling", "season": [1, 2, 3, 4], "sse_gain": [60.0, 30.0, 20.0, -10.0]})
    # Xét 2 mùa đóng góp nhiều nhất
    out = gain_concentration(t, "falling", k=2)
    # Tổng = 60 + 30 + 20 - 10 = 100
    assert out["total_gain"] == pytest.approx(100.0)
    # Hai mùa hàng đầu: 60 + 30 = 90, chiếm 90% tổng
    assert out["top_k_gain"] == pytest.approx(90.0) and out["top_k_share"] == pytest.approx(0.9)
    # 3 trong 4 mùa mô hình thắng (sse_gain > 0)
    assert out["seasons_better"] == 3 and out["seasons"] == 4


# ---- permutation importance và thí nghiệm cấu hình cố định -----------------------------------

def _synthetic_rows(n=300, seed=0):
    """y_target = 2 * y_t (không nhiễu); các cột còn lại là nhiễu độc lập, không liên quan."""
    # Bộ sinh số ngẫu nhiên có seed
    rng = np.random.default_rng(seed)
    # n dòng chia thành n/30 mùa, mỗi mùa 30 dòng (season, t); các đặc trưng là số ngẫu nhiên độc lập
    d = pd.DataFrame({"season": np.repeat(np.arange(n // 30), 30), "t": np.tile(np.arange(30), n // 30),
                      "y_t": rng.uniform(1000, 9000, n), "y_t1": rng.uniform(1000, 9000, n),
                      "y_t2": rng.uniform(1000, 9000, n), "c_t": rng.uniform(1000, 9000, n),
                      "c_t1": rng.uniform(1000, 9000, n), "c_target": rng.uniform(1000, 9000, n),
                      "doy_target": rng.integers(150, 360, n)})
    # Nhãn là hàm chính xác của y_t, nên y_t là đặc trưng duy nhất có thông tin
    d["y_target"] = 2.0 * d["y_t"]
    return d


def test_permutation_importance_finds_the_only_informative_feature():
    from lightgbm_diag import permutation_importance
    # Hai bộ dữ liệu độc lập (seed khác nhau) làm tập huấn luyện và tập test
    train, test = _synthetic_rows(seed=0), _synthetic_rows(seed=1)
    # Huấn luyện biến thể noclim với một cấu hình cố định
    model = fit_model(train, (8, 200, 0.1), "noclim", SEED)
    # Tính permutation importance, đặt tên đơn vị làm chỉ mục để tra cứu
    out = permutation_importance(model, test, "noclim", n_repeats=20, seed=0).set_index("unit")
    # Xáo trộn y_t phải làm RMSE tăng rất mạnh (> 1000 km²)
    assert out.loc["y_t", "delta_rmse"] > 1000                       # đổi chỗ y_t phá hỏng dự báo
    # Các đặc trưng nhiễu gần như không đổi gì: mức tăng < 10% của y_t
    for f in ("y_t1", "y_t2", "doy_target"):
        assert abs(out.loc[f, "delta_rmse"]) < 0.1 * out.loc["y_t", "delta_rmse"]
    # Kiểm tra cột "kind": "level" là nhóm, "y_t" là đặc trưng riêng lẻ
    assert out.loc["level", "kind"] == "group" and out.loc["y_t", "kind"] == "single"
    # Nhóm "level" chứa y_t nên mức tăng không thể thấp hơn nhiều so với y_t riêng lẻ
    assert out.loc["level", "delta_rmse"] >= out.loc["y_t", "delta_rmse"] * 0.9  # nhóm chứa y_t


def test_permutation_importance_is_zero_for_a_feature_the_model_never_splits_on():
    from lightgbm_diag import permutation_importance
    train, test = _synthetic_rows(seed=0), _synthetic_rows(seed=1)
    # Biến doy_target thành hằng số ở cả hai tập
    train["doy_target"] = 200
    test["doy_target"] = 200  # cột hằng: mô hình không thể chia theo nó, đổi chỗ không đổi gì
    model = fit_model(train, (8, 200, 0.1), "noclim", SEED)
    out = permutation_importance(model, test, "noclim", n_repeats=5, seed=0).set_index("unit")
    # Xáo trộn một cột hằng không thay đổi dự báo nên mức tăng RMSE phải đúng bằng 0
    assert out.loc["doy_target", "delta_rmse"] == pytest.approx(0.0, abs=1e-9)


def test_permutation_importance_is_reproducible_for_the_same_seed():
    from lightgbm_diag import permutation_importance
    train, test = _synthetic_rows(seed=0), _synthetic_rows(seed=1)
    model = fit_model(train, (4, 100, 0.1), "raw", SEED)
    # Chạy hai lần với cùng seed
    a = permutation_importance(model, test, "raw", n_repeats=7, seed=3)
    b = permutation_importance(model, test, "raw", n_repeats=7, seed=3)
    # Hai kết quả phải giống hệt nhau
    pd.testing.assert_frame_equal(a, b)


def test_permutation_summary_known_values():
    from lightgbm_diag import permutation_summary
    # Một đơn vị (y_t) qua 4 mùa test với mức tăng RMSE 100, 300, -50, 150
    t = pd.DataFrame({"model": "lgbm_raw", "h": 2, "unit": "y_t", "kind": "single",
                      "test_season": [2010, 2011, 2012, 2013], "delta_rmse": [100.0, 300.0, -50.0, 150.0]})
    # Chỉ có một nhóm nên lấy dòng đầu
    r = permutation_summary(t).iloc[0]
    # 4 mùa, trong đó 3 mùa có mức tăng > 0
    assert r["folds"] == 4 and r["folds_positive"] == 3
    # Trung bình = (100 + 300 - 50 + 150) / 4 = 125
    assert r["delta_rmse_mean"] == pytest.approx(125.0)
    # Độ lệch chuẩn mẫu (ddof=1)
    assert r["delta_rmse_sd"] == pytest.approx(np.std([100.0, 300.0, -50.0, 150.0], ddof=1))


def test_phase_metrics_known_values():
    from lightgbm_diag import phase_metrics
    # Năm dòng: 2 dòng ngày 200 (rising), 1 dòng ngày 270 (peak), 2 dòng ngày 300 (falling); thực tế đều 1000
    d = pd.DataFrame({"doy_target": [200, 200, 270, 300, 300], "y_target": [1000.0] * 5,
                      "pred": [1010.0, 990.0, 1100.0, 1030.0, 1050.0]})
    m = phase_metrics(d)
    # Rising: sai số +10 và -10 -> RMSE 10, bias 0 (hai sai số triệt tiêu nhau)
    assert m["rmse_rising"] == pytest.approx(10.0) and m["bias_rising"] == pytest.approx(0.0)
    # Peak: một sai số +100 -> RMSE 100, bias +100
    assert m["rmse_peak"] == pytest.approx(100.0) and m["bias_peak"] == pytest.approx(100.0)
    # Falling: sai số +30 và +50 -> RMSE = căn của trung bình bình phương
    assert m["rmse_falling"] == pytest.approx(np.sqrt((30.0**2 + 50.0**2) / 2))
    # Bias của falling = (30 + 50) / 2 = 40
    assert m["bias_falling"] == pytest.approx(40.0)
    # RMSE chung trên cả 5 dòng
    assert m["rmse_all"] == pytest.approx(np.sqrt((100 + 100 + 10000 + 900 + 2500) / 5))


# ---- hai mùa tiêu biểu ------------------------------------------------------------------------

def _season_pred():
    # Danh sách các dòng dự báo giả lập
    rows = []
    # Hai mô hình: lgbm_raw luôn lệch +10, anom_ar1 luôn lệch +20
    for model, err in (("lgbm_raw", 10.0), ("anom_ar1", 20.0)):
        # Hai mùa với mức nền khác nhau
        for season, level in ((2010, 1000.0), (2011, 5000.0)):
            # Hai tầm
            for h in (1, 2):
                # Ba bước gốc
                for t in (3, 4, 5):
                    # ngày đích = ngày đầu mùa 150 + 8 x (bước đích - 1), bước đích = t + h
                    rows.append({"model": model, "h": h, "season": season, "t": t,
                                 "doy_target": 150 + 8 * (t + h - 1),
                                 "y_target": level + 100 * t, "pred": level + 100 * t + err})
    return pd.DataFrame(rows)


def test_season_forecasts_aligns_models_on_the_same_target_step():
    from lightgbm_diag import season_forecasts
    # Mùa 2011, tầm 2, hai mô hình
    w = season_forecasts(_season_pred(), 2011, 2, ["lgbm_raw", "anom_ar1"])
    # Các cột phải đúng thứ tự quy định
    assert list(w.columns) == ["t", "doy_target", "y_target", "lgbm_raw", "anom_ar1"]
    # Có 3 bước gốc nên 3 dòng
    assert len(w) == 3
    # Mỗi mô hình lệch đúng như thiết kế (+10 và +20), chứng tỏ hai mô hình được căn đúng cùng một bước đích
    np.testing.assert_allclose(w["lgbm_raw"] - w["y_target"], 10.0)
    np.testing.assert_allclose(w["anom_ar1"] - w["y_target"], 20.0)
    # Các bước liên tiếp cách nhau đúng 8 ngày
    assert (w["doy_target"].diff().dropna() == 8).all()


def test_season_rmse_table_known_values():
    from lightgbm_diag import season_rmse_table
    # Đỉnh lũ quan sát giả định: mùa 2011 lớn hơn mùa 2010
    obs_peak = {2010: 4000.0, 2011: 9000.0}
    t = season_rmse_table(_season_pred(), "lgbm_raw", "anom_ar1", obs_peak)
    # Lấy dòng của mùa 2011, tầm 2
    r = t[(t["season"] == 2011) & (t["h"] == 2)].iloc[0]
    # RMSE: 10 cho mô hình, 20 cho đối chứng
    assert r["rmse_model"] == pytest.approx(10.0) and r["rmse_base"] == pytest.approx(20.0)
    # Tỉ lệ 10 / 20 = 0,5
    assert r["ratio"] == pytest.approx(0.5)
    # Mùa 2011 có đỉnh 9000 km², hạng 1 (lớn nhất)
    assert r["peak_km2"] == 9000.0 and r["peak_rank"] == 1
    # Mùa 2010 có hạng 2 ở mọi dòng
    assert t[(t["season"] == 2010)]["peak_rank"].eq(2).all()
    # Bảng có 2 mùa x 2 tầm = 4 dòng
    assert len(t) == 2 * 2  # mùa x tầm

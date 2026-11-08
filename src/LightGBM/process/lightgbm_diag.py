"""Chẩn đoán LightGBM (phần A2 của reports/members/thien_lightgbm.md).

Không sửa mô hình của nhóm: module này dùng lại đúng lưới `LGBM_GRID`, đặc trưng, `pick_config`
và `SEED` của `models/ml.py`, rồi ghi thêm những gì `benchmark` không lưu:

1. cấu hình được chọn ở từng fold, từng tầm, từng seed;
2. độ quan trọng đặc trưng (tổng gain) và SHAP trung bình tuyệt đối (TreeSHAP) trên mùa test;
3. độ ổn định theo seed (random_state) của RMSE, so với AnomAR1;
4. lợi thế theo từng mùa ở từng giai đoạn lũ (có đều hay chỉ do vài mùa);
5. (`extras`) permutation importance theo khối mùa (mỗi mùa test là một khối) và thí nghiệm
   đối chứng cấu hình cố định LightGBM so với Random Forest cùng đặc trưng. Các cấu hình trong
   thí nghiệm đối chứng chỉ để MÔ TẢ, không dùng để chọn mô hình hay cấu hình.

Seed chỉ đổi bagging (`subsample=0.8`). Chạy nhiều seed là phân tích độ ổn định, không dùng để
chọn seed tốt nhất. Chạy từ gốc repo:
`python src/LightGBM/process/lightgbm_diag.py [region] [số seed]`,
`python src/LightGBM/process/lightgbm_diag.py extras [region]` và
`python src/LightGBM/process/lightgbm_diag.py seasons [region]`.
Kết quả ghi vào `src/LightGBM/result/{tables,figures}`; `results/tables/probe_predictions.csv`
(do benchmark chung sinh ra) chỉ được đọc.
"""
# Cho phép dùng chú thích kiểu (type hint) mới (vd. `list[int]`) kể cả trên Python 3.10;
# các chú thích kiểu không bị tính toán lúc chạy.
from __future__ import annotations

import sys  # đọc tham số dòng lệnh (sys.argv) và đổi mã hóa in ra màn hình
from pathlib import Path  # làm việc với đường dẫn file/thư mục

import lightgbm as lgb  # thư viện gradient boosting: mô hình đang được chẩn đoán
import numpy as np  # mảng số và các phép toán vector hóa
import pandas as pd  # bảng dữ liệu (DataFrame)
from sklearn.ensemble import RandomForestRegressor  # Random Forest, dùng ở thí nghiệm đối chứng

# ---- Các thành phần dùng chung của cả nhóm (nằm trong src/floodbench/) ----------------------
from floodbench.cv import rolling_origin  # sinh các fold: huấn luyện các mùa trước năm Y, test năm Y
from floodbench.data.regions import load_region, repo_root  # đọc cấu hình vùng; lấy thư mục gốc repo
from floodbench.evaluation import pooled_rmse  # RMSE gộp tất cả dòng theo nhóm chỉ định
from floodbench.features import build_rows, to_wide  # dựng bảng đặc trưng; đổi bảng dài thành ma trận
from floodbench.models.forest import N_TREES, RF_GRID  # số cây và lưới cấu hình của Random Forest nhóm
# LGBM_GRID: 6 cấu hình (num_leaves, n_estimators, learning_rate); SEED: seed mặc định (0);
# _raw/_anom/_noclim: hàm dựng ma trận đặc trưng cho 3 biến thể mô hình
from floodbench.models.ml import LGBM_GRID, SEED, _anom, _noclim, _raw
from floodbench.models.tuning import pick_config  # chọn cấu hình bằng validation trong tập huấn luyện
# BASE = "anom_ar1" (mô hình đối chứng); PHASES = các giai đoạn lũ theo ngày trong năm
from floodbench.report_tables import BASE, PHASES

# Thư mục `result/` nằm cạnh thư mục chứa file này (parents[1] = src/LightGBM/)
RESULT = Path(__file__).resolve().parents[1] / "result"
# Nơi ghi các bảng csv và các hình png
TABLES, FIGURES = RESULT / "tables", RESULT / "figures"

# Các tầm dự báo tính theo số bước; 1 bước = 8 ngày nên (1, 2, 4) ứng với 8, 16, 32 ngày
HORIZONS = (1, 2, 4)
# Ba biến thể mô hình đã khai báo trước khi chạy test (không được chọn lại sau khi xem điểm test)
VARIANTS = ("raw", "anomaly", "noclim")
# Tên các cột đặc trưng của từng biến thể, đúng thứ tự cột của ma trận đưa vào mô hình
FEATURES = {
    # y_t, y_t1, y_t2: diện tích ngập hiện tại và 2 bước trước; c_t: climatology tại bước gốc;
    # c_target: climatology tại bước đích; doy_target: ngày trong năm của bước đích
    "raw": ["y_t", "y_t1", "y_t2", "c_t", "c_target", "doy_target"],
    # a_t, a_t1: anomaly (= giá trị thực trừ climatology) hiện tại và 1 bước trước
    "anomaly": ["a_t", "a_t1", "doy_target"],
    # như raw nhưng bỏ hẳn climatology (ablation: kiểm tra lợi thế có đến từ climatology không)
    "noclim": ["y_t", "y_t1", "y_t2", "doy_target"],
}
# Tra cứu: tên biến thể -> hàm dựng ma trận đặc trưng tương ứng
_DESIGN = {"raw": _raw, "anomaly": _anom, "noclim": _noclim}


def design(rows: pd.DataFrame, variant: str) -> np.ndarray:
    # Biến bảng các dòng (rows) thành ma trận đặc trưng theo biến thể `variant`
    return _DESIGN[variant](rows)


def fit_model(train: pd.DataFrame, cfg, variant: str, seed: int = SEED) -> lgb.LGBMRegressor:
    """Cùng tham số với models/ml.py; chỉ `random_state` được phép đổi."""
    # Tách bộ ba siêu tham số: số lá tối đa mỗi cây, số cây, tốc độ học
    leaves, n_est, lr = cfg
    # Nhãn huấn luyện: biến thể anomaly dự báo (y_target - climatology đích), biến thể khác dự báo y_target
    y = train["y_target"] - (train["c_target"] if variant == "anomaly" else 0)
    # Tạo mô hình LightGBM; các tham số cố định phải giống hệt models/ml.py để tái lập đúng benchmark
    model = lgb.LGBMRegressor(num_leaves=leaves, n_estimators=n_est, learning_rate=lr,
                              # min_child_samples=10: mỗi lá cần >= 10 dòng (chống học thuộc)
                              # subsample=0.8, subsample_freq=1: mỗi cây dùng ngẫu nhiên 80% dòng (bagging)
                              min_child_samples=10, subsample=0.8, subsample_freq=1,
                              # random_state=seed: seed của bagging (tham số duy nhất được đổi);
                              # verbose=-1: tắt thông báo của LightGBM
                              random_state=seed, verbose=-1)
    # Huấn luyện: đầu vào là ma trận đặc trưng, nhãn đổi sang mảng numpy
    model.fit(design(train, variant), y.to_numpy())
    # Trả về mô hình đã huấn luyện
    return model


def predict(model: lgb.LGBMRegressor, test: pd.DataFrame, variant: str) -> np.ndarray:
    # Dự báo thô của mô hình (với biến thể anomaly, đây là dự báo anomaly chứ chưa phải diện tích)
    out = model.predict(design(test, variant))
    # Biến thể anomaly: cộng lại climatology đích để về diện tích ngập (km²); biến thể khác cộng 0
    return out + (test["c_target"].to_numpy() if variant == "anomaly" else 0)


def choose_config(train: pd.DataFrame, variant: str, h: int, seed: int = SEED):
    """Trả về (chỉ số cấu hình trong LGBM_GRID, MSE validation của 6 cấu hình)."""
    # Hàm khớp-và-dự báo theo đúng giao diện mà pick_config yêu cầu: (train, valid, cấu hình, tầm) -> dự báo
    def fit_predict(tr, te, cfg, _h):
        # Khớp trên phần huấn luyện `tr`, dự báo phần validation `te`
        return predict(fit_model(tr, cfg, variant, seed), te, variant)

    # Thử cả 6 cấu hình trong LGBM_GRID; 3 mùa cuối của tập huấn luyện làm validation (không đụng mùa test)
    cfg, scores = pick_config(train, LGBM_GRID, fit_predict, h)
    # Đổi cấu hình thắng cuộc thành chỉ số trong lưới, kèm MSE validation của cả 6 cấu hình
    return LGBM_GRID.index(cfg), scores


def importance_of(model: lgb.LGBMRegressor, test: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Gain chuẩn hóa (tổng = 1) và SHAP trung bình tuyệt đối trên các dòng của mùa test.

    Mô tả mô hình học gì, không phải quan hệ nhân quả.
    """
    # Gain: tổng mức giảm sai số do mỗi đặc trưng tạo ra khi được dùng để chia nút trong mọi cây
    gain = model.booster_.feature_importance(importance_type="gain").astype(float)
    # Chuẩn hóa để tổng bằng 1 (thành "phần đóng góp"); nếu mô hình không chia nút nào thì trả toàn 0
    share = gain / gain.sum() if gain.sum() > 0 else np.zeros_like(gain)
    # pred_contrib=True trả về giá trị SHAP của từng đặc trưng cho từng dòng; cột cuối là giá trị nền nên bỏ
    contrib = model.predict(design(test, variant), pred_contrib=True)[:, :-1]  # bỏ cột giá trị nền
    # Bảng kết quả: mỗi đặc trưng một dòng, gồm phần gain và trung bình trị tuyệt đối của SHAP (km²)
    return pd.DataFrame({"feature": FEATURES[variant], "gain_share": share,
                         "mean_abs_shap": np.abs(contrib).mean(axis=0)})


def run_fold(Y: np.ndarray, cfg, fold, h: int, variants=VARIANTS, seeds=(SEED,)):
    """Một (fold, tầm): trả về (cấu hình đã chọn, độ quan trọng, dự báo) cho mọi biến thể và seed."""
    # Đổi danh sách năm (vd. 2005) thành chỉ số hàng trong ma trận Y (năm - năm đầu tiên)
    idx = lambda seasons: [s - cfg.first_year for s in seasons]
    # Bảng đặc trưng của các mùa huấn luyện (mỗi dòng = một gốc dự báo, nhãn y_target ở bước t + h)
    train = build_rows(Y, cfg, idx(fold.train_seasons), h)
    # Bảng đặc trưng của đúng một mùa test
    test = build_rows(Y, cfg, idx([fold.test_season]), h)
    # Ba danh sách gom kết quả: cấu hình đã chọn, độ quan trọng đặc trưng, dự báo
    cfg_rows, imp_rows, pred_rows = [], [], []
    # Duyệt từng biến thể mô hình
    for variant in variants:
        # Duyệt từng seed (seed chỉ đổi phần bagging ngẫu nhiên)
        for seed in seeds:
            # Chọn cấu hình tốt nhất bằng validation nội bộ: k là chỉ số trong LGBM_GRID, scores là MSE của 6 cấu hình
            k, scores = choose_config(train, variant, h, seed)
            # Huấn luyện lại trên toàn bộ tập huấn luyện với cấu hình đã chọn
            model = fit_model(train, LGBM_GRID[k], variant, seed)
            # Tên mô hình theo quy ước của benchmark (lgbm_raw, lgbm_anomaly, lgbm_noclim)
            name = f"lgbm_{variant}"
            # Giải nén cấu hình đã chọn để ghi vào bảng
            leaves, n_est, lr = LGBM_GRID[k]
            # Ghi lại cấu hình được chọn cùng ngữ cảnh: seed, mô hình, tầm, mùa test, điểm validation, cỡ tập huấn luyện
            cfg_rows.append({"seed": seed, "model": name, "h": h, "test_season": fold.test_season,
                             "cfg_index": k, "num_leaves": leaves, "n_estimators": n_est,
                             "learning_rate": lr, "val_mse": scores[k],
                             "n_train_seasons": len(fold.train_seasons), "n_train_rows": len(train)})
            # Tính độ quan trọng đặc trưng trên mùa test
            imp = importance_of(model, test, variant)
            # Gắn thêm nhãn (seed, mô hình, tầm, mùa test) để về sau gộp bảng vẫn phân biệt được
            imp_rows.append(imp.assign(seed=seed, model=name, h=h, test_season=fold.test_season))
            # Lưu dự báo của mô hình cho từng dòng của mùa test, kèm giá trị thực y_target
            pred_rows.append(test[["season", "t", "doy_target", "y_target"]].assign(
                model=name, h=h, seed=seed, pred=predict(model, test, variant)))
    # Gộp danh sách thành 3 bảng
    return (pd.DataFrame(cfg_rows), pd.concat(imp_rows, ignore_index=True),
            pd.concat(pred_rows, ignore_index=True))


def run(region: str = "mekong", seeds=range(10), variants=VARIANTS):
    # Đọc cấu hình của vùng (năm đầu/cuối, số bước, đường dẫn dữ liệu...)
    cfg = load_region(region)
    # Đọc dữ liệu đã tổng hợp thành chuỗi diện tích ngập, đổi thành ma trận (mùa x bước)
    Y = to_wide(pd.read_csv(cfg.path("processed_level1")), cfg)
    # Danh sách gom kết quả của từng fold
    cfgs, imps, preds = [], [], []
    # Rolling origin: mỗi fold huấn luyện các mùa trước năm test rồi kiểm tra năm test (2010–2020: 11 fold)
    for fold in rolling_origin(cfg):
        # Chạy từng tầm dự báo (8, 16, 32 ngày)
        for h in HORIZONS:
            # Chạy toàn bộ biến thể và seed ở (fold, tầm) này
            c, i, p = run_fold(Y, cfg, fold, h, variants, tuple(seeds))
            # Thêm kết quả vào danh sách (dấu phẩy tạo tuple chỉ để gọi cả ba lệnh append trên một dòng)
            cfgs.append(c), imps.append(i), preds.append(p)
        # Báo tiến độ; flush=True để hiện ngay lên màn hình
        print(f"fold {fold.test_season} xong", flush=True)
    # Gộp mọi fold thành ba bảng lớn: cấu hình, độ quan trọng, dự báo
    return (pd.concat(cfgs, ignore_index=True), pd.concat(imps, ignore_index=True),
            pd.concat(preds, ignore_index=True))


def seed_summary(pred: pd.DataFrame, baseline_rmse: dict, seed0: int = SEED) -> pd.DataFrame:
    """RMSE theo seed -> trung bình, độ lệch chuẩn (ddof=1), khoảng, so với RMSE của AnomAR1.

    baseline_rmse: {(model, h): RMSE của anom_ar1 ở tầm h} (anom_ar1 không ngẫu nhiên).
    """
    # RMSE gộp qua mọi mùa test, tính riêng cho từng (mô hình, tầm, seed): mỗi seed một con số
    per_seed = pooled_rmse(pred, by=("model", "h", "seed"))
    # Danh sách các dòng kết quả
    rows = []
    # Với mỗi (mô hình, tầm): g chứa các RMSE của các seed
    for (model, h), g in per_seed.groupby(["model", "h"]):
        # Mảng RMSE của các seed
        r = g["rmse"].to_numpy()
        # RMSE của AnomAR1 ở cùng tầm, để so sánh
        base = baseline_rmse[(model, h)]
        # Độ lệch chuẩn mẫu (ddof=1: chia n-1); chỉ có 1 seed thì không tính được nên trả NaN
        sd = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
        # Một dòng tổng hợp: số seed, trung bình, độ lệch chuẩn (tuyệt đối và theo %), nhỏ nhất, lớn nhất,
        # RMSE của seed 0 (bản benchmark chính thức), RMSE của AnomAR1,
        # tỉ lệ trung bình/AnomAR1 (< 1 là tốt hơn) và tỉ lệ seed thắng AnomAR1
        rows.append({"model": model, "h": h, "n_seeds": len(r), "rmse_mean": r.mean(), "rmse_sd": sd,
                     "rmse_sd_pct": 100 * sd / r.mean(), "rmse_min": r.min(), "rmse_max": r.max(),
                     "rmse_seed0": float(g.loc[g["seed"] == seed0, "rmse"].iloc[0]),
                     "anom_ar1_rmse": base, "mean_over_base": r.mean() / base,
                     "frac_seeds_better_than_base": float((r < base).mean())})
    # Đổi danh sách dòng thành bảng
    return pd.DataFrame(rows)


def season_phase_table(pred: pd.DataFrame, model: str, base: str = BASE, h: int = 2) -> pd.DataFrame:
    """RMSE của `model` và `base` theo (giai đoạn lũ, mùa test); sse_gain > 0 là mô hình tốt hơn."""
    # Chỉ giữ dự báo ở tầm h của hai mô hình cần so sánh
    d = pred[(pred["h"] == h) & pred["model"].isin([model, base])].copy()
    # Gán giai đoạn lũ theo ngày đích: PHASES cho các mốc [0, 257, 298], thêm 999 làm cận trên;
    # right=False nghĩa là khoảng đóng bên trái, mở bên phải: [0,257) rising, [257,298) peak, [298,999) falling
    d["phase"] = pd.cut(d["doy_target"], [a for a, _, _ in PHASES] + [999], right=False,
                        labels=[n for _, _, n in PHASES])
    # Sai số bình phương của từng dòng dự báo
    d["sq"] = (d["y_target"] - d["pred"]) ** 2
    # Với mỗi (giai đoạn, mùa, mô hình): MSE trung bình ("mean") và số dòng ("size")
    g = d.groupby(["phase", "season", "model"], observed=True)["sq"].agg(["mean", "size"]).reset_index()
    # Ghép hai mô hình cạnh nhau theo (giai đoạn, mùa); hậu tố _m là mô hình, _b là đối chứng
    m = g[g["model"] == model].merge(g[g["model"] == base], on=["phase", "season"], suffixes=("_m", "_b"))
    # Bảng kết quả
    out = pd.DataFrame({
        # giai đoạn (đổi sang chuỗi), mùa, số dòng dự báo trong ô này
        "phase": m["phase"].astype(str), "season": m["season"], "n_rows": m["size_m"],
        # RMSE của từng mô hình = căn bậc hai của MSE trung bình
        "rmse_model": np.sqrt(m["mean_m"]), "rmse_base": np.sqrt(m["mean_b"]),
        # tỉ lệ RMSE mô hình / đối chứng (< 1: mô hình tốt hơn)
        "ratio": np.sqrt(m["mean_m"] / m["mean_b"]),
        # mức giảm tổng sai số bình phương so với đối chứng (> 0: mô hình tốt hơn); cộng được giữa các mùa
        "sse_gain": (m["mean_b"] - m["mean_m"]) * m["size_m"]})
    # Cờ True nếu mô hình tốt hơn đối chứng trong ô (giai đoạn, mùa) này
    out["model_better"] = out["ratio"] < 1
    return out


def gain_concentration(table: pd.DataFrame, phase: str, k: int = 3) -> dict:
    """Tổng mức giảm sai số bình phương và phần do k mùa đóng góp nhiều nhất."""
    # Chỉ lấy các dòng của giai đoạn đang xét
    t = table[table["phase"] == phase]
    # Sắp xếp mức giảm sai số từ lớn đến nhỏ
    gains = t["sse_gain"].sort_values(ascending=False)
    # total: tổng mức giảm của mọi mùa; top: tổng của k mùa đóng góp nhiều nhất
    total, top = float(gains.sum()), float(gains.head(k).sum())
    # Kết quả: số mùa, số mùa mô hình thắng, tổng mức giảm, phần của k mùa hàng đầu (NaN nếu tổng <= 0)
    return {"phase": phase, "seasons": len(t), "seasons_better": int((t["sse_gain"] > 0).sum()),
            "total_gain": total, "top_k": k, "top_k_gain": top,
            "top_k_share": top / total if total > 0 else float("nan")}


# Nhóm đặc trưng cho permutation importance nhóm (đổi chỗ cả nhóm cùng một hoán vị,
# giữ nguyên quan hệ bên trong nhóm vì c_target và doy_target, y_t và y_t1 phụ thuộc nhau)
GROUPS = {
    "raw": {"level": ["y_t", "y_t1", "y_t2"], "climatology": ["c_t", "c_target"], "calendar": ["doy_target"]},
    "anomaly": {"anomaly": ["a_t", "a_t1"], "calendar": ["doy_target"]},
    "noclim": {"level": ["y_t", "y_t1", "y_t2"], "calendar": ["doy_target"]},
}


def permutation_importance(model, test: pd.DataFrame, variant: str, n_repeats: int = 30,
                           seed: int = 0) -> pd.DataFrame:
    """Mức tăng RMSE trên mùa test khi đổi chỗ ngẫu nhiên một đặc trưng (hoặc một nhóm) giữa các dòng.

    Dùng nhãn của mùa test nên chỉ để chẩn đoán sau khi chạy, không để chọn mô hình. Đặc trưng
    tương quan mạnh với nhau làm số này khó diễn giải, vì vậy báo cả bản theo nhóm.
    """
    # Bộ sinh số ngẫu nhiên có seed cố định để kết quả tái lập được
    rng = np.random.default_rng(seed)
    # Ma trận đặc trưng của mùa test
    X = design(test, variant)
    # Biến thể anomaly phải cộng lại climatology đích sau khi dự báo; biến thể khác cộng 0
    offset = test["c_target"].to_numpy() if variant == "anomaly" else 0.0
    # Giá trị thực
    y = test["y_target"].to_numpy()
    # MSE gốc của mô hình trên mùa test khi chưa xáo trộn gì (mốc so sánh)
    base_mse = float(np.mean((y - (model.predict(X) + offset)) ** 2))
    # Tên các cột đặc trưng của biến thể này
    names = FEATURES[variant]
    # Các "đơn vị" cần xáo trộn: (tên, loại, danh sách chỉ số cột). Trước hết là từng đặc trưng riêng lẻ
    units = [(f, "single", [names.index(f)]) for f in names]
    # Rồi thêm từng nhóm đặc trưng (xáo trộn cả nhóm cùng lúc)
    units += [(g, "group", [names.index(f) for f in fs]) for g, fs in GROUPS[variant].items()]
    # Danh sách gom kết quả
    rows = []
    # Với từng đơn vị cần kiểm tra
    for unit, kind, cols in units:
        # MSE sau xáo trộn của từng lần lặp
        mses = []
        # Lặp n_repeats lần để lấy trung bình, giảm nhiễu do ngẫu nhiên
        for _ in range(n_repeats):
            # Một hoán vị ngẫu nhiên của các dòng
            perm = rng.permutation(len(X))
            # Sao chép ma trận để không làm hỏng bản gốc
            Xp = X.copy()
            # Thay các cột đang xét bằng giá trị lấy từ các dòng bị xáo; cả nhóm dùng chung một hoán vị
            # nên quan hệ bên trong nhóm được giữ nguyên, các cột còn lại không đổi
            Xp[:, cols] = X[perm][:, cols]
            # MSE của mô hình trên dữ liệu đã bị xáo trộn
            mses.append(float(np.mean((y - (model.predict(Xp) + offset)) ** 2)))
        # Đổi danh sách thành mảng numpy
        mses = np.array(mses)
        # Mức tăng MSE và tăng RMSE (km²) so với mốc: càng lớn thì mô hình càng phụ thuộc đơn vị này
        rows.append({"unit": unit, "kind": kind, "delta_mse": float(mses.mean() - base_mse),
                     "delta_rmse": float(np.mean(np.sqrt(mses)) - np.sqrt(base_mse))})
    return pd.DataFrame(rows)


def permutation_fold(Y: np.ndarray, cfg, fold, h: int, variants=VARIANTS, seed: int = SEED,
                     n_repeats: int = 30) -> pd.DataFrame:
    """Permutation importance của mô hình mà benchmark chọn (cấu hình chọn bằng validation) ở một fold."""
    # Đổi năm thành chỉ số hàng trong Y (giống run_fold)
    idx = lambda seasons: [s - cfg.first_year for s in seasons]
    # Bảng đặc trưng của các mùa huấn luyện
    train = build_rows(Y, cfg, idx(fold.train_seasons), h)
    # Bảng đặc trưng của mùa test
    test = build_rows(Y, cfg, idx([fold.test_season]), h)
    # Danh sách gom kết quả của từng biến thể
    out = []
    # Với từng biến thể mô hình
    for variant in variants:
        # Chọn cấu hình bằng validation nội bộ, đúng như benchmark làm
        k, _ = choose_config(train, variant, h, seed)
        # Huấn luyện lại trên toàn bộ tập huấn luyện
        model = fit_model(train, LGBM_GRID[k], variant, seed)
        # Tính permutation importance trên mùa test, gắn nhãn mô hình, tầm và mùa test
        out.append(permutation_importance(model, test, variant, n_repeats, seed).assign(
            model=f"lgbm_{variant}", h=h, test_season=fold.test_season))
    return pd.concat(out, ignore_index=True)


def permutation_summary(perm: pd.DataFrame) -> pd.DataFrame:
    """Qua các khối mùa (mỗi mùa test một khối): trung bình, độ lệch chuẩn, số mùa có mức tăng > 0."""
    # Danh sách các dòng kết quả
    rows = []
    # Gộp theo (mô hình, tầm, đơn vị, loại); sort=False giữ nguyên thứ tự xuất hiện
    for (model, h, unit, kind), g in perm.groupby(["model", "h", "unit", "kind"], sort=False):
        # Mức tăng RMSE của từng mùa test (mỗi mùa là một khối độc lập)
        d = g["delta_rmse"].to_numpy()
        # Tổng hợp: số mùa, số mùa có mức tăng > 0 (đơn vị đó thực sự có ích), trung bình và độ lệch chuẩn mẫu
        rows.append({"model": model, "h": h, "unit": unit, "kind": kind, "folds": len(d),
                     "folds_positive": int((d > 0).sum()), "delta_rmse_mean": d.mean(),
                     "delta_rmse_sd": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")})
    return pd.DataFrame(rows)


def phase_metrics(d: pd.DataFrame) -> pd.Series:
    """RMSE chung, RMSE và sai số trung bình (pred - y; dương = dự báo cao hơn thực tế) theo pha lũ."""
    # Sai số có dấu của từng dòng: dự báo trừ thực tế
    err = d["pred"].to_numpy() - d["y_target"].to_numpy()
    # Gán giai đoạn lũ theo ngày đích (cùng cách chia như season_phase_table), đổi thành mảng numpy
    phase = pd.cut(d["doy_target"], [a for a, _, _ in PHASES] + [999], right=False,
                   labels=[n for _, _, n in PHASES]).to_numpy()
    # RMSE trên toàn bộ dòng
    out = {"rmse_all": float(np.sqrt(np.mean(err ** 2)))}
    # Với từng giai đoạn lũ
    for _, _, name in PHASES:
        # Lấy sai số của các dòng thuộc giai đoạn này
        e = err[phase == name]
        # RMSE của giai đoạn (NaN nếu không có dòng nào)
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
        # Bias của giai đoạn: sai số trung bình có dấu (dương = dự báo cao hơn thực tế)
        out[f"bias_{name}"] = float(e.mean()) if len(e) else float("nan")
    return pd.Series(out)


def fixed_config_models() -> dict:
    """Mô hình cố định (không chọn bằng validation), cùng đặc trưng `raw`: nhãn -> (họ, hàm khớp)."""
    # Tạo hàm huấn luyện Random Forest với (độ sâu, số dòng tối thiểu mỗi lá, tỉ lệ đặc trưng mỗi lần chia)
    def rf(depth, leaf, mf):
        # Trả về hàm nhận tập huấn luyện `tr` và trả về mô hình đã khớp; dùng đặc trưng raw, nhãn y_target
        return lambda tr: RandomForestRegressor(n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf,
                                                max_features=mf, random_state=SEED, n_jobs=-1
                                                ).fit(_raw(tr), tr["y_target"].to_numpy())

    # Tạo hàm huấn luyện LightGBM với (số lá, số cây, tốc độ học, min_child_samples); các tham số khác như benchmark
    def gbm(leaves, n, lr, mcs=10):
        # Trả về hàm nhận tập huấn luyện `tr` và trả về mô hình LightGBM đã khớp trên đặc trưng raw
        return lambda tr: lgb.LGBMRegressor(num_leaves=leaves, n_estimators=n, learning_rate=lr,
                                            min_child_samples=mcs, subsample=0.8, subsample_freq=1,
                                            random_state=SEED, verbose=-1
                                            ).fit(_raw(tr), tr["y_target"].to_numpy())

    # Từ điển kết quả: nhãn hiển thị -> (tên họ mô hình, hàm khớp)
    m = {}
    # Các cấu hình Random Forest trong lưới của nhóm
    for i, (d, leaf, mf) in enumerate(RF_GRID):
        m[f"RF lưới {i}: depth={d}, leaf={leaf}, max_features={mf}"] = ("RF (lưới nhóm)", rf(d, leaf, mf))
    # Thêm hai Random Forest nông (ngoài lưới) để xem cây nông có giúp không
    m["RF depth=2, leaf=5"] = ("RF (thêm, nông)", rf(2, 5, 1.0))
    m["RF depth=3, leaf=5"] = ("RF (thêm, nông)", rf(3, 5, 1.0))
    # Các cấu hình LightGBM trong lưới của nhóm
    for i, (a, b, c) in enumerate(LGBM_GRID):
        m[f"LightGBM lưới {i}: leaves={a}, n={b}, lr={c}"] = ("LightGBM (lưới nhóm)", gbm(a, b, c))
    # Thêm hai LightGBM sâu (31 lá, min_child_samples=5) ngoài lưới để xem mô hình phức tạp hơn có khác không
    m["LightGBM sâu: leaves=31, n=300, lr=0.05, mcs=5"] = ("LightGBM (thêm, sâu)", gbm(31, 300, 0.05, 5))
    m["LightGBM sâu: leaves=31, n=100, lr=0.3, mcs=5"] = ("LightGBM (thêm, sâu)", gbm(31, 100, 0.3, 5))
    return m


def fixed_config_table(Y: np.ndarray, cfg, h: int = 2) -> pd.DataFrame:
    """Chạy mọi mô hình cố định qua các fold rolling-origin, gộp dự báo rồi tính phase_metrics."""
    # Tất cả mô hình cố định cần thử
    models = fixed_config_models()
    # Đổi năm thành chỉ số hàng trong Y
    idx = lambda seasons: [s - cfg.first_year for s in seasons]
    # Mỗi mô hình có một danh sách chứa dự báo của các fold
    preds = {k: [] for k in models}
    # Duyệt từng fold rolling-origin
    for fold in rolling_origin(cfg):
        # Đặc trưng của các mùa huấn luyện
        train = build_rows(Y, cfg, idx(fold.train_seasons), h)
        # Đặc trưng của mùa test
        test = build_rows(Y, cfg, idx([fold.test_season]), h)
        # Với từng mô hình: nhãn và (họ, hàm khớp)
        for label, (_, fit) in models.items():
            # Khớp trên tập huấn luyện, dự báo mùa test, lưu kèm mùa, ngày đích và giá trị thực
            preds[label].append(test[["season", "doy_target", "y_target"]].assign(
                pred=fit(train).predict(_raw(test))))
    # Danh sách các dòng kết quả
    rows = []
    # Với từng mô hình: gộp dự báo của mọi fold rồi tính RMSE chung/theo giai đoạn/bias
    for label, (family, _) in models.items():
        rows.append({"model": label, "family": family, "h": h,
                     **phase_metrics(pd.concat(preds[label])).to_dict()})
    return pd.DataFrame(rows)


# Ba mô hình được vẽ trong hình "mùa tiêu biểu": LightGBM raw, AnomAR1 (đối chứng), Clim5 (climatology 5 mùa)
SEASON_MODELS = ("lgbm_raw", "anom_ar1", "clim5")
# Kiểu vẽ của từng mô hình: (nhãn chú giải, màu, kiểu nét)
SEASON_STYLE = {"lgbm_raw": ("LightGBM raw", "#2E7D32", "-"), "anom_ar1": ("AnomAR1", "#C62828", "--"),
                "clim5": ("Clim5", "#757575", ":")}


def season_forecasts(pred: pd.DataFrame, season: int, h: int, models=SEASON_MODELS) -> pd.DataFrame:
    """Giá trị thực và dự báo của từng mô hình, căn theo cùng bước đích, trong một mùa và một tầm."""
    # Chỉ giữ dự báo của mùa, tầm và các mô hình cần vẽ
    d = pred[(pred["season"] == season) & (pred["h"] == h) & pred["model"].isin(models)]
    # Xoay bảng: mỗi mô hình thành một cột dự báo, mỗi dòng là một bước (cùng t, ngày đích, giá trị thực)
    w = d.pivot(index=["t", "doy_target", "y_target"], columns="model", values="pred").reset_index()
    # Xóa tên "model" của chỉ mục cột do pivot để lại
    w.columns.name = None
    # Sắp cột theo thứ tự cố định và sắp dòng theo bước thời gian t
    return w[["t", "doy_target", "y_target", *models]].sort_values("t").reset_index(drop=True)


def season_rmse_table(pred: pd.DataFrame, model: str, base: str, peak_by_season: dict) -> pd.DataFrame:
    """RMSE theo (mùa, tầm) của `model` và `base`, tỉ lệ, đỉnh lũ quan sát và hạng đỉnh (1 = lớn nhất)."""
    # Chỉ giữ hai mô hình, thêm cột sai số bình phương
    d = pred[pred["model"].isin([model, base])].assign(sq=lambda x: (x["y_target"] - x["pred"]) ** 2)
    # RMSE theo (mùa, tầm, mô hình) = căn bậc hai của trung bình; unstack đưa mỗi mô hình thành một cột
    r = d.groupby(["season", "h", "model"])["sq"].mean().pow(0.5).unstack("model").reset_index()
    # Bảng gọn: mùa, tầm, RMSE mô hình, RMSE đối chứng và tỉ lệ (< 1: mô hình tốt hơn)
    out = pd.DataFrame({"season": r["season"], "h": r["h"], "rmse_model": r[model], "rmse_base": r[base],
                        "ratio": r[model] / r[base]})
    # Đỉnh lũ quan sát của từng mùa (km²), chuyển thành Series để tra cứu theo mùa
    peaks = pd.Series(peak_by_season)
    # Hạng của đỉnh: 1 = mùa có đỉnh lớn nhất
    ranks = peaks.rank(ascending=False).astype(int)
    # Gắn đỉnh lũ vào từng dòng theo mùa
    out["peak_km2"] = out["season"].map(peaks)
    # Gắn hạng đỉnh vào từng dòng theo mùa
    out["peak_rank"] = out["season"].map(ranks)
    return out


def plot_seasons(pred: pd.DataFrame, seasons, path, models=SEASON_MODELS) -> None:
    # Nhập matplotlib trong hàm để chỉ cần cài khi thật sự vẽ hình
    import matplotlib
    # Dùng backend "Agg": vẽ ra file, không mở cửa sổ (chạy được trên máy không có màn hình)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Lưới ô vẽ: mỗi hàng một mùa, mỗi cột một tầm; squeeze=False để axes luôn là mảng 2 chiều
    fig, axes = plt.subplots(len(seasons), len(HORIZONS), figsize=(13, 3.6 * len(seasons)), squeeze=False)
    # Duyệt từng mùa (hàng)
    for i, season in enumerate(seasons):
        # Duyệt từng tầm dự báo (cột)
        for j, h in enumerate(HORIZONS):
            # Ô vẽ tại hàng i, cột j
            ax = axes[i][j]
            # Bảng thực tế và dự báo của mùa/tầm này
            w = season_forecasts(pred, season, h, models)
            # Đường giá trị thực: đen, nét dày
            ax.plot(w["doy_target"], w["y_target"], color="black", lw=2, label="Thực tế")
            # Vẽ đường dự báo của từng mô hình
            for m in models:
                # Lấy nhãn, màu, kiểu nét của mô hình
                label, color, ls = SEASON_STYLE[m]
                ax.plot(w["doy_target"], w[m], color=color, ls=ls, lw=1.6, label=label)
            # RMSE của từng mô hình trên riêng mùa/tầm này, để ghi trong tiêu đề
            rm = {m: float(np.sqrt(np.mean((w[m] - w["y_target"]) ** 2))) for m in models}
            # Tiêu đề: mùa, tầm tính theo ngày (h bước x 8 ngày), RMSE của LightGBM và AnomAR1
            ax.set_title(f"Mùa {season}, tầm {h * 8} ngày\nRMSE: LightGBM {rm['lgbm_raw']:,.0f}"
                         f" | AnomAR1 {rm['anom_ar1']:,.0f}", fontsize=10)
            # Nhãn trục x
            ax.set_xlabel("Ngày trong năm của bước đích")
            # Nhãn trục y
            ax.set_ylabel("Diện tích ngập (km²)")
            # Chỉ vẽ chú giải ở ô đầu tiên để khỏi lặp lại
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    # Tự điều chỉnh khoảng cách để các tiêu đề/nhãn không chồng lên nhau
    fig.tight_layout()
    # Lưu hình ra file
    _save(fig, path)


def _save(fig, path):
    import matplotlib.pyplot as plt

    # Lưu hình với độ phân giải 150 dpi
    fig.savefig(path, dpi=150)
    # Đóng hình để giải phóng bộ nhớ
    plt.close(fig)
    return path


def make_figures(tables: dict, out_dir) -> list:
    # Nhập và cấu hình matplotlib như plot_seasons
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Tạo thư mục đầu ra nếu chưa có
    out_dir.mkdir(parents=True, exist_ok=True)
    # Danh sách đường dẫn các hình đã lưu
    saved = []

    # ---- Hình 1: SHAP trung bình tuyệt đối của từng đặc trưng, ba biến thể, tầm 16 ngày ----
    imp = tables["importance"]
    # 1 hàng x 3 ô (mỗi biến thể một ô)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    # Ghép từng ô vẽ với tên mô hình tương ứng
    for ax, name in zip(axes, [f"lgbm_{v}" for v in VARIANTS]):
        # Chỉ lấy dòng của mô hình này, tầm h = 2 và seed mặc định
        s = imp[(imp["model"] == name) & (imp["h"] == 2) & (imp["seed"] == SEED)]
        # Trung bình qua các mùa test cho mỗi đặc trưng, sắp tăng dần để thanh dài nhất nằm trên cùng
        s = s.groupby("feature")["mean_abs_shap"].mean().sort_values()
        # Biểu đồ thanh ngang
        ax.barh(s.index, s.values, color="#4C78A8")
        ax.set_title(f"{name} (h = 2)")
        ax.set_xlabel("SHAP trung bình tuyệt đối (km²)")
    fig.tight_layout()
    saved.append(_save(fig, out_dir / "shap_h2.png"))

    # ---- Hình 2: phân bố RMSE giữa các seed (hộp râu) so với đường AnomAR1 ----
    # pred: dự báo mọi seed; base: RMSE của AnomAR1 theo (mô hình, tầm)
    pred, base = tables["seed_predictions"], tables["baseline_rmse"]
    # RMSE của từng (mô hình, tầm, seed)
    per = pooled_rmse(pred, by=("model", "h", "seed"))
    # 1 hàng x 3 ô (mỗi tầm một ô)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, h in zip(axes, HORIZONS):
        # Với mỗi biến thể: mảng RMSE của các seed ở tầm h
        data = [per[(per["model"] == f"lgbm_{v}") & (per["h"] == h)]["rmse"] for v in VARIANTS]
        # Hộp râu cho ba biến thể (tick_labels là tên tham số của matplotlib mới)
        ax.boxplot(data, tick_labels=[f"lgbm_{v}" for v in VARIANTS])
        # Đường ngang đứt nét: RMSE của AnomAR1 (nếu hộp nằm dưới đường này thì LightGBM tốt hơn)
        ax.axhline(base[("lgbm_raw", h)], color="#E45756", ls="--", label="anom_ar1")
        ax.set_title(f"Tầm {h * 8} ngày")
        ax.set_ylabel("RMSE (km²)")
        ax.legend()
        # Xoay nhãn trục x 20 độ cho khỏi chồng chữ
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    saved.append(_save(fig, out_dir / "seed_rmse.png"))

    # ---- Hình 3: tỉ lệ RMSE lgbm_raw / anom_ar1 theo mùa, chia theo giai đoạn lũ ----
    ph = tables["phase_by_season"]
    # 1 hàng x 3 ô (rising, peak, falling); sharey=True để cùng thang trục y, dễ so sánh
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (_, _, name) in zip(axes, PHASES):
        # Các dòng của giai đoạn này, sắp theo mùa
        s = ph[ph["phase"] == name].sort_values("season")
        # Cột xanh nếu tỉ lệ < 1 (LightGBM tốt hơn), đỏ nếu >= 1
        ax.bar(s["season"].astype(str), s["ratio"],
               color=["#54A24B" if r < 1 else "#E45756" for r in s["ratio"]])
        # Đường mốc tỉ lệ = 1 (hai mô hình bằng nhau)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_title(f"{name} (h = 2)")
        ax.tick_params(axis="x", labelrotation=60)
    # Nhãn trục y chỉ cần ở ô đầu
    axes[0].set_ylabel("RMSE lgbm_raw / anom_ar1")
    fig.tight_layout()
    saved.append(_save(fig, out_dir / "phase_ratio_by_season_h2.png"))
    return saved


def main(region: str = "mekong", n_seeds: int = 10) -> None:
    # Đảm bảo thư mục ghi bảng tồn tại
    TABLES.mkdir(parents=True, exist_ok=True)
    # Chạy toàn bộ rolling-origin với n_seeds seed: ra bảng cấu hình, độ quan trọng, dự báo
    cfgs, imps, preds = run(region, seeds=range(n_seeds))
    # Đọc dự báo chính thức của benchmark chung (chỉ đọc, không ghi)
    official = pd.read_csv(repo_root() / "results" / "tables" / "probe_predictions.csv")
    # Bỏ các mô hình có hậu tố _w7 (bản cửa sổ huấn luyện 7 mùa, không phải bản chính)
    official = official[~official["model"].str.endswith("_w7")]
    # RMSE của AnomAR1 theo tầm, từ bản benchmark chính thức
    base_rmse = pooled_rmse(official[official["model"] == BASE], by=("model", "h"))
    # Đổi thành từ điển {(tên mô hình LightGBM, tầm): RMSE AnomAR1}; AnomAR1 dùng chung cho cả 3 biến thể
    base = {(m, h): float(r) for m in (f"lgbm_{v}" for v in VARIANTS)
            for h, r in zip(base_rmse["h"], base_rmse["rmse"])}

    # Bảng độ ổn định theo seed, làm tròn 4 chữ số
    summary = seed_summary(preds, base).round(4)
    # Bảng lợi thế theo (giai đoạn lũ, mùa) của lgbm_raw so với AnomAR1 ở tầm 16 ngày
    phase = season_phase_table(official, "lgbm_raw").round(4)
    # Ghi các bảng ra csv (index=False: không ghi cột chỉ số của pandas)
    cfgs.round(4).to_csv(TABLES / "lightgbm_chosen_configs.csv", index=False)
    imps.round(6).to_csv(TABLES / "lightgbm_feature_importance.csv", index=False)
    preds.round(4).to_csv(TABLES / "lightgbm_seed_predictions.csv", index=False)
    summary.to_csv(TABLES / "lightgbm_seed_summary.csv", index=False)
    phase.to_csv(TABLES / "lightgbm_phase_by_season_h2.csv", index=False)

    # In bảng độ ổn định theo seed ra màn hình
    print("\nRMSE theo seed (km², cột rmse_*), so với anom_ar1:\n", summary.to_string(index=False))
    # In mức tập trung của lợi thế: lợi thế có đều giữa các mùa hay chỉ do vài mùa, cho từng giai đoạn lũ
    for name in ("rising", "peak", "falling"):
        print("\n", gain_concentration(phase, name))
    # Vẽ ba hình và lưu vào thư mục hình
    figs = make_figures({"importance": imps, "seed_predictions": preds, "baseline_rmse": base,
                         "phase_by_season": phase}, FIGURES)
    # In đường dẫn các hình, mỗi hình một dòng (sep="\n  " xuống dòng và thụt 2 khoảng trắng)
    print("\nHình:", *[str(f) for f in figs], sep="\n  ")


def main_extras(region: str = "mekong") -> None:
    """Permutation importance (seed 0, mô hình benchmark chọn) và thí nghiệm cấu hình cố định."""
    # Đảm bảo thư mục ghi bảng tồn tại
    TABLES.mkdir(parents=True, exist_ok=True)
    # Đọc cấu hình vùng và dữ liệu, dựng ma trận (mùa x bước) như hàm run
    cfg = load_region(region)
    Y = to_wide(pd.read_csv(cfg.path("processed_level1")), cfg)
    # Permutation importance cho mọi (fold, tầm), gộp thành một bảng
    perm = pd.concat([permutation_fold(Y, cfg, fold, h) for fold in rolling_origin(cfg) for h in HORIZONS],
                     ignore_index=True)
    # Ghi bảng chi tiết theo từng fold
    perm.round(4).to_csv(TABLES / "lightgbm_permutation_importance.csv", index=False)
    # Tổng hợp qua các mùa test và ghi bảng tóm tắt
    summ = permutation_summary(perm).round(3)
    summ.to_csv(TABLES / "lightgbm_permutation_summary.csv", index=False)
    # In bảng tóm tắt theo nhóm đặc trưng
    print("Permutation importance (nhóm, tăng RMSE km² qua 11 mùa test):\n",
          summ[summ["kind"] == "group"].to_string(index=False))
    # Thí nghiệm đối chứng: LightGBM và Random Forest với cấu hình cố định ở tầm 16 ngày
    fixed = fixed_config_table(Y, cfg, h=2).round(1)
    fixed.to_csv(TABLES / "lightgbm_vs_rf_fixed_configs_h2.csv", index=False)
    print("\nCấu hình cố định, cùng đặc trưng raw, h = 2 (chỉ để mô tả):\n", fixed.to_string(index=False))


def main_seasons(region: str = "mekong", seasons=(2011, 2015)) -> None:
    """Hai mùa tiêu biểu: bảng RMSE theo mùa và hình quan sát so với dự báo (bản benchmark, seed 0).

    Chọn theo tiêu chí ghi sẵn: 2011 là mùa có đỉnh lớn nhất và là mùa LightGBM kém AnomAR1
    ở cả ba tầm; 2015 là mùa có đỉnh thấp nhất. Tiêu chí kiểm bằng bảng, không chọn theo cảm tính.
    """
    # Đảm bảo thư mục ghi bảng tồn tại
    TABLES.mkdir(parents=True, exist_ok=True)
    # Đọc dự báo chính thức của benchmark rồi bỏ các mô hình cửa sổ 7 mùa (_w7), như hàm main
    official = pd.read_csv(repo_root() / "results" / "tables" / "probe_predictions.csv")
    official = official[~official["model"].str.endswith("_w7")]
    # Đỉnh lũ quan sát của từng mùa: lấy giá trị thực lớn nhất (các mô hình đều chứa cùng y_target nên dùng clim5)
    peaks = official[official["model"] == "clim5"].groupby("season")["y_target"].max().to_dict()
    # Bảng RMSE theo (mùa, tầm) của lgbm_raw so với AnomAR1, kèm đỉnh lũ và hạng đỉnh
    table = season_rmse_table(official, "lgbm_raw", BASE, peaks).round(4)
    table.to_csv(TABLES / "lightgbm_season_rmse.csv", index=False)
    # In riêng các mùa được chọn (2011 và 2015)
    print(table[table["season"].isin(seasons)].to_string(index=False))
    # Đường dẫn file hình, tạo thư mục chứa nếu chưa có
    out = FIGURES / "season_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Vẽ hình so sánh thực tế với dự báo của các mô hình cho các mùa được chọn
    plot_seasons(official, seasons, out)
    print("Hình:", out)


# Khối này chỉ chạy khi chạy trực tiếp file (python lightgbm_diag.py ...), không chạy khi bị import
if __name__ == "__main__":
    # Đặt mã hóa UTF-8 cho đầu ra để tiếng Việt không lỗi trên Windows
    sys.stdout.reconfigure(encoding="utf-8")
    # Tham số đầu tiên chọn chế độ: "seasons", "extras"; không có thì chạy chế độ chính
    if sys.argv[1:2] == ["seasons"]:
        # Tham số thứ hai (nếu có) là tên vùng, mặc định "mekong"
        main_seasons(*(sys.argv[2:3] or ["mekong"]))
    elif sys.argv[1:2] == ["extras"]:
        main_extras(*(sys.argv[2:3] or ["mekong"]))
    else:
        # Chế độ chính: tham số 1 = tên vùng, tham số 2 = số seed (mặc định 10)
        main(*(sys.argv[1:2] or ["mekong"]), *([int(a) for a in sys.argv[2:3]] or [10]))

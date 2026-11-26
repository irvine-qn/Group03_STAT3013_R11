"""Mô hình GAM, viết lại từng bước cho dễ đọc.

Mô hình giữ đúng đặc tả GAM của benchmark thăm dò của nhóm (cùng đặc trưng, cùng lưới 6 cấu hình,
cùng cách chọn cấu hình), nên dự báo trùng với bản đã chạy trước đây. Test `tests/GAM/test_gam_model.py`
khóa lại các dự báo đó; đổi mô hình thì phải chạy lại mọi bước và báo cả nhóm.

------------------------------------------------------------------------------------------------
Ý tưởng trong 4 dòng
------------------------------------------------------------------------------------------------
1. "Mức bình thường" c = trung bình diện tích ngập của 5 mùa liền trước, tại cùng bước.
2. "Anomaly" a = y - c : hôm nay ngập nhiều hơn (a > 0) hay ít hơn (a < 0) mức bình thường.
3. GAM dự báo anomaly của bước đích:
       a_{t+h} ≈ β0 + s(ngày đích) + β1·a_t + β2·a_{t-1}
   - s(·) là một hàm TRƠN (spline phạt) của ngày trong năm: mô hình tự học hình dạng của nó.
   - β1, β2 là hệ số TUYẾN TÍNH của anomaly hôm nay và 8 ngày trước.
4. Dự báo diện tích = c_{t+h} + anomaly dự báo.

Vì sao dự báo anomaly mà không dự báo thẳng y? Vì phần lớn biến động của y là chu kỳ mùa
(lên - đỉnh - rút) và c đã chứa sẵn chu kỳ đó. Mô hình chỉ còn phải học phần lệch khỏi chu kỳ,
nhỏ hơn và gần dừng hơn. AnomAR1 (mốc so sánh) dùng đúng cách tách này.

------------------------------------------------------------------------------------------------
Chọn siêu tham số (giống mọi mô hình khác của nhóm)
------------------------------------------------------------------------------------------------
GAM có 2 siêu tham số của hàm trơn s(·):
  n_splines : số hàm cơ sở (độ "mềm" tối đa của đường cong)
  lam       : hệ số phạt độ cong (lớn -> đường cong phẳng hơn)
Lưới cố định 6 cấu hình (ngân sách bằng mọi mô hình). Trong MỖI fold và MỖI tầm, cấu hình được
chọn bằng `gam_protocol.pick_config`: học trên các mùa huấn luyện trừ 3 mùa cuối,
chấm MSE trên 3 mùa cuối đó, lấy cấu hình có MSE nhỏ nhất, rồi học lại trên toàn bộ mùa huấn luyện.
Mùa test KHÔNG BAO GIỜ được nhìn thấy khi chọn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pygam import LinearGAM, l, s  # LinearGAM: GAM với nhiễu chuẩn; s = hàm trơn; l = tuyến tính

from gam_protocol import pick_config  # chọn cấu hình bằng 3 mùa cuối của tập học

# 6 cấu hình (n_splines, lam), cùng lưới với benchmark thăm dò (ghi trong nhật ký tinh chỉnh của nhóm).
GAM_GRID = [(6, 0.1), (6, 1.0), (8, 0.6), (8, 5.0), (12, 1.0), (12, 10.0)]

# Tên 3 cột của ma trận đầu vào, đúng thứ tự cột trong X
FEATURE_NAMES = ["doy_target", "a_t", "a_t1"]


# ---- 1. Từ bảng dòng dự báo sang ma trận X và mục tiêu ---------------------------------------
def design_matrix(rows: pd.DataFrame) -> np.ndarray:
    """X = [ngày đích, a_t, a_{t-1}] với a = y - c (lệch so với mức bình thường).

    `rows` là bảng của `gam_protocol.build_rows`: mỗi dòng là một gốc dự báo (mùa, bước t).
    """
    a_t = rows["y_t"] - rows["c_t"]      # anomaly hôm nay
    a_t1 = rows["y_t1"] - rows["c_t1"]   # anomaly 8 ngày trước
    return np.column_stack([rows["doy_target"].to_numpy(), a_t.to_numpy(), a_t1.to_numpy()])


def anomaly_target(rows: pd.DataFrame) -> np.ndarray:
    """Mục tiêu học: anomaly ở bước đích, y_{t+h} - c_{t+h}."""
    return (rows["y_target"] - rows["c_target"]).to_numpy()


# ---- 2. Học và dự báo với MỘT cấu hình -------------------------------------------------------
def fit_gam(train: pd.DataFrame, config: tuple[int, float]) -> LinearGAM:
    """Học GAM: s(cột 0 = ngày đích) + l(cột 1 = a_t) + l(cột 2 = a_{t-1}) (+ hệ số chặn)."""
    n_splines, lam = config
    terms = s(0, n_splines=n_splines, lam=lam) + l(1) + l(2)
    return LinearGAM(terms).fit(design_matrix(train), anomaly_target(train))


def predict(model: LinearGAM, rows: pd.DataFrame) -> np.ndarray:
    """Diện tích dự báo = mức bình thường ở bước đích + anomaly dự báo."""
    return rows["c_target"].to_numpy() + model.predict(design_matrix(rows))


def fit_predict(train, test, config, h):
    """Chữ ký (train, valid, config, h) mà `pick_config` yêu cầu. h không dùng: mỗi tầm có
    bảng dòng riêng (y_target đã là giá trị sau h bước), nên mô hình không cần biết h."""
    return predict(fit_gam(train, config), test)


# ---- 3. Toàn bộ quy trình cho một fold, một tầm ------------------------------------------------
@dataclass
class GamFit:
    pred: np.ndarray              # dự báo diện tích (km²) cho các dòng test
    config: tuple[int, float]     # cấu hình được chọn
    val_mse: list[float]          # MSE validation của 6 cấu hình (cùng thứ tự GAM_GRID)
    model: LinearGAM              # mô hình đã học lại trên toàn bộ mùa huấn luyện


def gam_forecast(train: pd.DataFrame, test: pd.DataFrame, h: int) -> GamFit:
    """Chọn cấu hình bằng validation trong tập huấn luyện, học lại, rồi dự báo mùa test."""
    config, val_mse = pick_config(train, GAM_GRID, fit_predict, h)
    model = fit_gam(train, config)
    return GamFit(pred=predict(model, test), config=config, val_mse=val_mse, model=model)


# ---- 4. Đọc mô hình đã học ---------------------------------------------------------------------
# Thứ tự hệ số trong pyGAM: [n_splines hệ số của s(ngày)] [β1] [β2] [β0 = hệ số chặn]
def coefficients(model: LinearGAM) -> dict:
    stats_ = model.statistics_
    return {
        "beta0_intercept": float(model.coef_[-1]),
        "beta1_a_t": float(model.coef_[-3]),
        "beta2_a_t1": float(model.coef_[-2]),
        "edof_total": float(stats_["edof"]),
        # Bậc tự do hiệu dụng của riêng hàm trơn: tổng trừ 3 tham số (β0, β1, β2).
        # ~1 nghĩa là gần như đường thẳng; càng lớn càng uốn lượn.
        "edof_smooth": float(stats_["edof"] - 3),
        "r2_pseudo": float(stats_["pseudo_r2"]["explained_deviance"]),
    }


def seasonal_curve(model: LinearGAM, doys) -> np.ndarray:
    """β0 + s(ngày): anomaly dự báo khi hôm nay ĐÚNG mức bình thường (a_t = a_{t-1} = 0).

    Đây là phần "sửa lệch theo mùa vụ" mà GAM học thêm so với AnomAR1. Hàm trơn s(·) riêng
    lẻ không xác định được mức (nó và β0 có thể đổi chỗ cho nhau), nên vẽ tổng β0 + s(·).
    """
    X = np.column_stack([np.asarray(doys, float), np.zeros(len(doys)), np.zeros(len(doys))])
    return model.predict(X)


def contributions(model: LinearGAM, rows: pd.DataFrame) -> pd.DataFrame:
    """Tách một dự báo thành từng phần cộng lại: c_target + β0 + s(ngày) + β1·a_t + β2·a_{t-1}."""
    X = design_matrix(rows)
    out = pd.DataFrame({
        "c_target": rows["c_target"].to_numpy(),
        "beta0": model.coef_[-1],
        "s_doy": model.partial_dependence(term=0, X=X),
        "beta1_a_t": model.partial_dependence(term=1, X=X),
        "beta2_a_t1": model.partial_dependence(term=2, X=X),
    })
    out["pred"] = out.sum(axis=1)
    return out

"""Kiểm tra phụ (THĂM DÒ) – fine-tune có giúp Chronos-2 bắt kịp AnomAR1 không?

Chạy (cần GPU, sau chronos2_forecast.py):  python src/Chronus-2/chronos2_finetune_check.py
Ghi:  results/Chronus-2/tables/finetune_check_predictions.csv  (dự báo của 2 bản fine-tune)
      results/Chronus-2/tables/finetune_check.csv              (RMSE so với AnomAR1 và zero-shot)

Cách làm: với mỗi mùa test Y, sao chép Chronos-2 rồi fine-tune trên anomaly của các mùa học
2003..Y−1 (mỗi mùa là một chuỗi 27 bước), sau đó dự báo mùa Y đúng như bản zero-shot.
Không có mùa test nào lọt vào dữ liệu học.

VÌ SAO CHỈ LÀ THĂM DÒ, không phải mô hình chính thức:
- Hai cấu hình dưới đây được chọn sau khi đã thử trên mùa 2020, tức là có nhìn mùa test;
  giao thức chung yêu cầu chọn bằng 3 mùa cuối của tập học, với 6 cấu hình như mọi mô hình.
- Mỗi cấu hình chỉ chạy một lần, trên GPU; quá trình huấn luyện trên GPU không tái lập được tuyệt đối.
Vì vậy bảng này chỉ dùng để trả lời "nếu fine-tune thì kết luận có đổi không", không đưa vào
bảng so sánh chính và không kiểm định.
"""
from __future__ import annotations

import sys
import tempfile

import numpy as np
import pandas as pd

from chronos2_data import REPO_ROOT, load_config, load_flood_matrix
from chronos2_evaluate import TABLES, add_season_part, pooled_rmse
from chronos2_forecast import PREDICTIONS_PATH, chronos_season
from chronos2_model import Chronos2ZeroShot, load_pipeline
from chronos2_protocol import HORIZON_DAYS, HORIZONS, TEST_SEASONS, anomaly, training_seasons

# Hai cách fine-tune: "full" sửa mọi trọng số (bước học rất nhỏ); "lora" chỉ học thêm vài ma trận nhỏ.
FINETUNE_CONFIGS = {
    "chronos2_ft_full": {"finetune_mode": "full", "num_steps": 50, "learning_rate": 1e-6},
    "chronos2_ft_lora": {"finetune_mode": "lora", "num_steps": 50, "learning_rate": 1e-5},
}
CHECK_PREDICTIONS_PATH = TABLES / "finetune_check_predictions.csv"
CHECK_TABLE_PATH = TABLES / "finetune_check.csv"


def training_series(Y: np.ndarray, test_season: int, first_year: int) -> list[np.ndarray]:
    """Dữ liệu học cho mùa test: anomaly của từng mùa học, mỗi mùa là một chuỗi riêng."""
    seasons = training_seasons(test_season, first_year)
    return [anomaly(Y, season - first_year).astype(np.float32) for season in seasons]


def finetuned_predictions(Y, config: dict, base, name: str, settings: dict) -> pd.DataFrame:
    """Fine-tune lại từ bản gốc cho TỪNG mùa test rồi dự báo mùa đó (rolling origin)."""
    rows = []
    with tempfile.TemporaryDirectory() as checkpoint_dir:  # checkpoint tạm, không ghi vào repo
        for test_season in TEST_SEASONS:
            series = training_series(Y, test_season, config["first_year"])
            tuned = base.fit(series, prediction_length=max(HORIZONS), batch_size=32,
                             output_dir=checkpoint_dir, remove_printer_callback=True, **settings)
            rows += chronos_season(Y, config, Chronos2ZeroShot(tuned), test_season)
            print(f"  {name} mùa {test_season}: xong", flush=True)
    return pd.DataFrame(rows).assign(model=name)


def summary_table(pred: pd.DataFrame) -> pd.DataFrame:
    """RMSE theo tầm, thêm hai cột tách gốc đầu–giữa mùa và cuối mùa ở tầm 32 ngày."""
    by_h = pooled_rmse(pred, ["model", "h"]).pivot(index="model", columns="h", values="rmse")
    table = pd.DataFrame({f"{HORIZON_DAYS[h]} days": by_h.get(h) for h in HORIZONS})
    h4 = add_season_part(pred[pred["h"] == 4])
    parts = pooled_rmse(h4, ["model", "season_part"]).pivot(index="model", columns="season_part", values="rmse")
    for part in parts.columns:
        table[f"32 days, {part}"] = parts[part]
    return table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_config("mekong")
    Y = load_flood_matrix(config)
    base = load_pipeline(device="cuda")
    tuned = [finetuned_predictions(Y, config, base, name, settings)
             for name, settings in FINETUNE_CONFIGS.items()]
    tuned = pd.concat(tuned, ignore_index=True)
    tuned.to_csv(CHECK_PREDICTIONS_PATH, index=False)

    reference = pd.read_csv(PREDICTIONS_PATH)  # AnomAR1 và Chronos-2 zero-shot (CPU)
    table = summary_table(pd.concat([reference, tuned], ignore_index=True))
    table.round(0).to_csv(CHECK_TABLE_PATH)
    print(table.round(0).to_string())
    print(f"-> {CHECK_TABLE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

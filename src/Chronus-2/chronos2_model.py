"""Chronos-2 zero-shot: dùng model nền (foundation model) đã huấn luyện sẵn, không học lại.

"Zero-shot" nghĩa là: không huấn luyện, không tinh chỉnh siêu tham số trên dữ liệu lũ.
Model chỉ nhìn một đoạn chuỗi (ngữ cảnh) rồi dự báo vài bước tiếp theo.

Vì sao đưa vào ANOMALY thay vì diện tích thô: chu kỳ mùa lũ (lên – đỉnh – rút) đã biết qua
climatology. Bỏ nó đi thì model chỉ cần dự báo phần "lệch khỏi bình thường"; sau đó cộng lại
climatology để ra km². Mốc AnomAR1 cũng làm đúng như vậy, nên hai bên so sánh công bằng.
"""
from __future__ import annotations

import numpy as np

from chronos2_protocol import climatology

MODEL_ID = "amazon/chronos-2"
MEDIAN = 0.5  # lấy phân vị 0,5 (trung vị) của phân phối dự báo làm dự báo điểm


def build_anomaly_context(Y: np.ndarray, season_idx: int, origin: int) -> np.ndarray:
    """Ngữ cảnh tại gốc t: anomaly của chính mùa đang dự báo, từ bước 1 đến bước t.

    Chỉ dùng mùa hiện tại vì các mùa cách nhau ~150 ngày khô, không nối thành một chuỗi liền.
    """
    c = climatology(Y, season_idx)
    return Y[season_idx, :origin] - c[:origin]


def load_pipeline(model_id: str = MODEL_ID, device: str = "cpu"):
    """Tải trọng số Chronos-2 (lần đầu tải từ Hugging Face, các lần sau lấy từ bộ nhớ đệm).

    device: "cpu" (mặc định, kết quả chính thức chạy trên CPU) hoặc "cuda" (GPU, dùng cho
    fine-tune và để đối chiếu; chạy zero-shot trên GPU RTX 4050 lệch CPU tối đa 0,002 km²).
    """
    from chronos import Chronos2Pipeline  # import ở đây để test không cần tải model

    return Chronos2Pipeline.from_pretrained(model_id, device_map=device)


class Chronos2ZeroShot:
    """Bọc pipeline Chronos-2 thành một hàm dự báo anomaly dễ dùng.

    pipeline được truyền từ ngoài vào: khi chạy thật là load_pipeline(), khi test là bản giả.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def forecast_anomaly(self, anomaly_context: np.ndarray, n_ahead: int) -> np.ndarray:
        """Dự báo trung vị anomaly cho n_ahead bước tới; phần tử thứ h − 1 là tầm h."""
        import torch

        series = torch.tensor(anomaly_context, dtype=torch.float32)
        quantiles, _mean = self.pipeline.predict_quantiles(
            [series], prediction_length=n_ahead, quantile_levels=[MEDIAN]
        )
        # quantiles[0] có dạng (số biến = 1, số bước tới, số phân vị = 1)
        return quantiles[0][0, :, 0].numpy()

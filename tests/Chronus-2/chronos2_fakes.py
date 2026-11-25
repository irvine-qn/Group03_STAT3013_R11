"""Pipeline giả dùng chung cho các test: thay cho Chronos-2 thật để test chạy nhanh, không cần mạng."""
import torch


class FakePipeline:
    """Giả lập Chronos2Pipeline.predict_quantiles.

    Ghi lại mọi lần gọi và trả dự báo step_value × (1, 2, 3, ...) cho các bước tới.
    step_value = 0 nghĩa là "anomaly dự báo bằng 0", tức dự báo đúng bằng climatology.
    """

    def __init__(self, step_value=10.0):
        self.step_value = step_value
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.calls.append(
            {"inputs": inputs, "prediction_length": prediction_length, "quantile_levels": quantile_levels}
        )
        values = self.step_value * torch.arange(1, prediction_length + 1, dtype=torch.float32)
        quantiles = [values.reshape(1, prediction_length, 1)]  # (biến, bước, phân vị)
        return quantiles, [values.reshape(1, prediction_length)]

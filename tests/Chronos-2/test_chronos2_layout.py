"""Quy ước tên file: mọi module trong src/Chronos-2 bắt đầu bằng "chronos2_".

Các thư mục mô hình khác cũng chèn đường dẫn của mình vào sys.path khi chạy pytest chung;
tên chung chung như chronos2_evaluate.py hay chronos2_protocol.py sẽ trùng nhau và một bộ test có thể vô tình
kiểm tra code của mô hình khác.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "Chronos-2"


def test_modules_have_chronos2_prefix():
    names = sorted(p.name for p in SRC.glob("*.py"))
    assert names and all(name.startswith("chronos2_") for name in names), names

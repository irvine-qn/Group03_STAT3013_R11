"""Cho phép test import trực tiếp các file trong src/Chronos-2/.

Thư mục có dấu gạch ngang nên không import được như một package Python;
vì vậy chèn nó vào sys.path (cùng cách làm với thư mục test của LightGBM).
"""
import sys
from pathlib import Path

# parents[2] của tests/Chronos-2/conftest.py là thư mục gốc repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "Chronos-2"))

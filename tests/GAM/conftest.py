"""Cho phép test import trực tiếp các file trong src/GAM/ (gam_model, run_model, ...).

pytest tự nạp conftest.py trước các test cùng thư mục. parents[2] của file này là gốc repo
(parents[0] = tests/GAM/, parents[1] = tests/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "GAM"))

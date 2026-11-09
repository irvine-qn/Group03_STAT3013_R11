"""Cho phép test import trực tiếp các file trong src/GAM/process/ (gam_model, run_model, ...).

pytest tự nạp conftest.py trước các test cùng thư mục. parents[1] của file này là src/GAM/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "process"))

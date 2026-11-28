"""Cho phép test import trực tiếp các file trong src/Comparison/."""
import sys
from pathlib import Path

# parents[2] của tests/Comparison/conftest.py là thư mục gốc repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "Comparison"))

"""Cho phép test import trực tiếp các file trong src/AnomAR1/."""
import sys
from pathlib import Path

# parents[2] của tests/AnomAR1/conftest.py là thư mục gốc repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "AnomAR1"))

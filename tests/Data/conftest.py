"""Cho phép test import trực tiếp các file trong src/Data/ (level1, regions, schema)."""
import sys
from pathlib import Path

# parents[2] của tests/Data/conftest.py là thư mục gốc repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "Data"))

"""Cho phép test import `lightgbm_diag` trực tiếp từ src/LightGBM/."""
# pytest tự nạp file conftest.py trước khi chạy các test trong cùng thư mục
import sys  # sys.path: danh sách thư mục Python tìm module khi `import`
from pathlib import Path  # làm việc với đường dẫn

# Chèn src/LightGBM/ vào đầu sys.path để `from lightgbm_diag import ...` tìm thấy file lightgbm_diag.py.
# __file__ là conftest.py; parents[0] là tests/LightGBM/, parents[1] là tests/, parents[2] là gốc repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "LightGBM"))

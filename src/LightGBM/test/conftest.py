"""Cho phép test import `lightgbm_diag` trực tiếp từ src/LightGBM/process/."""
# pytest tự nạp file conftest.py trước khi chạy các test trong cùng thư mục
import sys  # sys.path: danh sách thư mục Python tìm module khi `import`
from pathlib import Path  # làm việc với đường dẫn

# Chèn thư mục process/ vào đầu sys.path để `from lightgbm_diag import ...` tìm thấy file lightgbm_diag.py.
# __file__ là conftest.py; parents[1] là src/LightGBM/ (parents[0] là test/); từ đó vào process/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "process"))

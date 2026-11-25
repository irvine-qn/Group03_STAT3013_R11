"""Test đọc dữ liệu đầu vào (flood_data.py)."""
import numpy as np
import pandas as pd
import pytest

from flood_data import REPO_ROOT, load_config, load_flood_matrix, step_to_doy


def test_real_matrix_shape():
    Y = load_flood_matrix(load_config())
    assert Y.shape == (21, 27)  # 21 mùa (2000–2020) × 27 bước 8 ngày
    assert np.isfinite(Y).all()
    assert Y[0, 0] == 2403.7  # mùa 2000, bước 1 (ngày 153)


def test_step_to_doy():
    config = load_config()
    assert step_to_doy(config, 1) == 153
    assert step_to_doy(config, 27) == 361


def test_missing_cell_raises(tmp_path):
    config = load_config()
    full = pd.read_csv(REPO_ROOT / config["processed_level1"])
    broken = full[~((full["season"] == 2000) & (full["step"] == 5))]
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="2000"):
        load_flood_matrix(config, csv_path=path)

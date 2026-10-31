import numpy as np
import pandas as pd
import pytest

from floodbench.models.baselines import BASELINES, anom_ar1


def rows(**cols):
    return pd.DataFrame(cols)


def test_persistence_and_clim_columns():
    test = rows(y_t=[5.0], c_target=[7.0], y_prev_season=[3.0], clim_all_target=[9.0])
    train = rows(y_t=[1.0], y_t1=[1.0], c_t=[1.0], c_t1=[1.0])
    assert BASELINES["persistence"](train, test, 1)[0] == 5.0
    assert BASELINES["clim5"](train, test, 1)[0] == 7.0
    assert BASELINES["seasonal_naive"](train, test, 1)[0] == 3.0
    assert BASELINES["clim_all"](train, test, 1)[0] == 9.0


def test_anom_ar1_recovers_phi_and_decays_with_horizon():
    phi = 0.8
    a_prev = np.array([100.0, -50.0, 30.0, 200.0])
    train = rows(y_t=10 + phi * a_prev, y_t1=10 + a_prev, c_t=[10.0] * 4, c_t1=[10.0] * 4)
    test = rows(y_t=[110.0], c_t=[10.0], c_target=[20.0])  # anomaly today = 100
    assert anom_ar1(train, test, 1)[0] == pytest.approx(20 + 0.8 * 100)
    assert anom_ar1(train, test, 2)[0] == pytest.approx(20 + 0.8**2 * 100)
    assert anom_ar1(train, test, 4)[0] == pytest.approx(20 + 0.8**4 * 100)

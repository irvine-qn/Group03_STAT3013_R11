import pytest

from floodbench.data.regions import load_region, repo_root


def test_load_mekong_config():
    cfg = load_region("mekong")
    assert cfg.region == "mekong"
    assert (cfg.first_year, cfg.last_year) == (2000, 2020)
    assert (cfg.first_doy, cfg.step_days, cfg.n_steps) == (153, 8, 27)
    assert cfg.target_column == "water_km2"
    assert list(cfg.seasons) == list(range(2000, 2021))


def test_step_and_doy_are_inverse():
    cfg = load_region("mekong")
    assert cfg.doy_of(1) == 153
    assert cfg.doy_of(27) == 361
    assert cfg.step_of(233) == 11
    assert all(cfg.step_of(cfg.doy_of(s)) == s for s in range(1, 28))


def test_step_of_rejects_doy_off_the_calendar():
    cfg = load_region("mekong")
    with pytest.raises(ValueError):
        cfg.step_of(154)
    with pytest.raises(ValueError):
        cfg.step_of(369)


def test_unknown_region_raises():
    with pytest.raises(FileNotFoundError):
        load_region("atlantis")


def test_repo_root_contains_configs():
    assert (repo_root() / "configs" / "regions" / "mekong.yaml").exists()

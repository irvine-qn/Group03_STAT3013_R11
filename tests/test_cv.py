from floodbench.cv import rolling_origin
from floodbench.data.regions import load_region

CFG = load_region("mekong")


def test_every_training_season_precedes_the_test_season():
    folds = list(rolling_origin(CFG, first_test=2010, last_test=2020))
    assert [f.test_season for f in folds] == list(range(2010, 2021))
    for f in folds:
        assert max(f.train_seasons) < f.test_season
        assert f.train_seasons == sorted(f.train_seasons)


def test_training_window_keeps_only_the_most_recent_seasons():
    fold = next(iter(rolling_origin(CFG, first_test=2015, last_test=2015, window=4)))
    assert fold.train_seasons == [2011, 2012, 2013, 2014]


def test_first_training_season_leaves_room_for_climatology():
    fold = next(iter(rolling_origin(CFG, first_test=2010, last_test=2010)))
    assert fold.train_seasons[0] == 2003 and fold.train_seasons[-1] == 2009

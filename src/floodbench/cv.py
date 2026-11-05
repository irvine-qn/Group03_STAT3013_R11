"""Rolling origin theo năm: huấn luyện các mùa trước năm Y, kiểm tra năm Y."""
from __future__ import annotations

from dataclasses import dataclass

from floodbench.data.regions import RegionConfig


@dataclass(frozen=True)
class Fold:
    train_seasons: list[int]
    test_season: int


def rolling_origin(cfg: RegionConfig, first_test: int = 2010, last_test: int = 2020,
                   min_prior: int = 3, window: int | None = None):
    """Mùa huấn luyện đầu tiên là first_year + min_prior (để mỗi dòng có đủ climatology).

    window: chỉ giữ `window` mùa gần nhất (để giữ lượng dữ liệu học không đổi giữa các fold).
    """
    start = cfg.first_year + min_prior
    for test in range(first_test, last_test + 1):
        train = list(range(start, test))
        if window is not None:
            train = train[-window:]
        yield Fold(train_seasons=train, test_season=test)

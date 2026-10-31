"""Chạy benchmark rolling-origin cho một vùng: python -m floodbench.benchmark [region].

Ghi results/tables/probe_predictions.csv (mỗi dòng một dự báo) để mọi bảng/hình sau này
sinh từ file này, không dán số tay.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from floodbench.cv import rolling_origin
from floodbench.data.regions import load_region, repo_root
from floodbench.features import build_rows, to_wide
from floodbench.models.baselines import BASELINES
from floodbench.models.ml import Chronos2Zero, lgbm_anomaly, lgbm_noclim, lgbm_raw
from floodbench.models.stat import gam

HORIZONS = (1, 2, 4)
WINDOW = 7  # số mùa huấn luyện cố định cho biến thể "_w7"
TRAINED = {"gam": gam, "lgbm_raw": lgbm_raw, "lgbm_anomaly": lgbm_anomaly, "lgbm_noclim": lgbm_noclim,
           "anom_ar1": BASELINES["anom_ar1"]}


def run(region: str = "mekong") -> pd.DataFrame:
    cfg = load_region(region)
    Y = to_wide(pd.read_csv(cfg.path("processed_level1")), cfg)
    idx = lambda seasons: [s - cfg.first_year for s in seasons]
    chronos = Chronos2Zero(Y, cfg)
    out = []
    for window, suffix in ((None, ""), (WINDOW, "_w7")):
        for fold in rolling_origin(cfg, window=window):
            t0 = time.time()
            for h in HORIZONS:
                train = build_rows(Y, cfg, idx(fold.train_seasons), h)
                test = build_rows(Y, cfg, idx([fold.test_season]), h)
                models = dict(TRAINED)
                if window is None:
                    models.update({k: v for k, v in BASELINES.items() if k != "anom_ar1"})
                    models["chronos2"] = chronos
                for name, fn in models.items():
                    pred = fn(train, test, h)
                    out.append(test[["season", "t", "doy_target", "y_target"]].assign(
                        model=name + suffix, h=h, pred=pred))
            print(f"{fold.test_season} window={window} {time.time() - t0:.0f}s", flush=True)
    result = pd.concat(out, ignore_index=True)
    dst = repo_root() / "results" / "tables" / "probe_predictions.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(dst, index=False)
    print(f"{len(result)} predictions -> {dst}")
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run(*sys.argv[1:2])

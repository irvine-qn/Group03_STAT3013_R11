# Group 03

Flood-extent forecasting for the Vietnamese Mekong Delta: four models compared
against the AnomAR1 baseline on the same protocol (11 test seasons 2010–2020,
yearly rolling origin, horizons of 8, 16 and 32 days).

The project is intended to run locally on Windows.

## Local setup

Use Python 3.12.10 or later in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Models

Each model is self-contained: its code is in `src/<Model>/`, its tests in
`tests/<Model>/`, its outputs in `results/<Model>/` and its overview notebook in
`notebooks/<Model>/`. Start from the notebook; the `README.md` in each `src/`
folder lists the scripts in the order to run them.

| Model | Family | Source | Overview notebook |
|---|---|---|---|
| AnomAR1 | statistical baseline | [`src/AnomAR1/`](src/AnomAR1/) | [`notebooks/AnomAR1/`](notebooks/AnomAR1/) |
| GAM | statistical | [`src/GAM/`](src/GAM/) | [`notebooks/GAM/`](notebooks/GAM/) |
| LightGBM | machine learning | [`src/LightGBM/`](src/LightGBM/) | [`notebooks/LightGBM/`](notebooks/LightGBM/) |
| Chronos-2 | foundation model, zero-shot | [`src/Chronos-2/`](src/Chronos-2/) | [`notebooks/Chronos-2/`](notebooks/Chronos-2/) |

### Shared result tables

Every model writes the same five tables to `results/<Model>/tables/`, so they
can be combined:

| File | Columns |
|---|---|
| `predictions.csv` | `season, t, doy_target, y_target, model, h, pred` (one row per forecast) |
| `rmse.csv` | `model, 1, 2, 4` (pooled RMSE in km² for horizons h = 1, 2, 4 steps = 8, 16, 32 days) |
| `vs_anom_ar1.csv` | `model, h, n_seasons, rmse_ratio, wins, mse_diff_pct, mse_diff_pct_ci_low, mse_diff_pct_ci_high, p_t, p_wilcoxon, p_tost, p_t_holm` (season-level tests against AnomAR1, n = 11) |
| `rmse_by_phase.csv` | `h, model, rising, peak, falling` (pooled RMSE by flood phase of the target day: before day 257, days 257–297, from day 298) |
| `rmse_by_size.csv` | `h, model, small, medium, large` (pooled RMSE by flood size: the 11 test seasons split into terciles of their peak) |

Other tables in each folder are specific to that model.

The cross-model table that the report and the video quote is
`results/Comparison/tables/vs_anom_ar1_main.csv`: the 9 main comparisons, unrounded, with the columns of
`vs_anom_ar1.csv` except that `p_t_holm` becomes `p_t_holm_model` (Holm within that model's folder) and
`p_t_holm_main` (Holm over the main family) is added.

### Multiple comparisons

The **main family** is fixed in advance, not chosen from the results: each member's main model (`gam`,
`lgbm_raw`, `chronos2`) against `anom_ar1` at h = 1, 2, 4, i.e. 9 comparisons. Holm is applied to their 9
season-level paired t-test p-values (n = 11), recomputed from `predictions.csv` so nothing is rounded before
the adjustment. This is `p_t_holm_main`.

Variants (`anom_ar1_w7`, `gam_w7`, `lgbm_anomaly`, `lgbm_noclim`) and the simple baselines (`clim5`,
`persistence`) are secondary analyses. They are not in the main family and keep the Holm adjustment within
their own model folder (`p_t_holm` in `results/<Model>/tables/vs_anom_ar1.csv`).

### Comparison across models

[`src/Comparison/`](src/Comparison/) reads the four `predictions.csv` files, checks that every folder uses the
same AnomAR1 forecasts, and writes the combined tables and figures to `results/Comparison/`. Its season-level
tests apply Holm over all 9 main comparisons (3 models × 3 horizons). Start from
[`notebooks/Comparison/`](notebooks/Comparison/).

```powershell
python src/Comparison/comparison_tables.py
python src/Comparison/comparison_figures.py
```

Result over 11 test seasons: no model beats AnomAR1 significantly; zero-shot Chronos-2 is significantly worse at
32 days (Holm p = 0.045); GAM is equivalent to AnomAR1 at 8 days within ±5% RMSE.

## Interactive report

Reads the shared tables of all four models:

```powershell
streamlit run app/streamlit_app.py
```

## Tests

Run from the project root:

```powershell
python -m pytest
```

## Project layout

```text
Group03_STAT3013_R11/
├── app/
│   └── streamlit_app.py      # Local Streamlit report UI
├── configs/
│   └── regions/
│       └── mekong.yaml       # Per-region settings (paths relative to project root)
├── data/
│   ├── raw/                  # Source datasets (not committed)
│   └── processed/            # Derived datasets (level1_series.csv is the shared input)
├── docs/                     # Project documentation
├── notebooks/                # Overview notebook of each model and of the comparison (Vietnamese and English)
├── results/                  # Outputs, one folder per model plus Comparison/
├── src/                      # Source code, one folder per model plus Comparison/
├── tests/                    # Tests, one folder per model plus Comparison/
├── pyproject.toml            # Project metadata and pytest settings
├── requirements.txt          # Pinned runtime dependencies
└── README.md
```

Note: Git does not track empty folders. Add a `.gitkeep` file to any empty
folder above that should exist in the repository.

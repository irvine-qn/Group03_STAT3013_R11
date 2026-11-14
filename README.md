# Group 03

The project is intended to run locally on Windows

## Local setup

Use Python 3.12.10 or later in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Once the source package has been added under `src/floodbench/`, install it in
editable mode:

```powershell
python -m pip install -e .
```

After the source package, data, and generated result tables are in place, run
the interactive report with:

```powershell
streamlit run app/streamlit_app.py
```

Run tests from the project root with:

```powershell
python -m pytest
```

## Project layout

- `src/floodbench/`: reusable data-processing, modeling, and reporting code.
- `configs/regions/`: per-region YAML settings; paths in these files are
  relative to the project root.
- `data/raw/`: locally stored source datasets; raw files are not committed.
- `data/interim/` and `data/processed/`: derived datasets.
- `results/tables/`: generated CSV tables read by the report app.
- `app/`: local Streamlit report UI.
- `tests/`: automated tests.

The scaffold is not runnable until the source package, input data, and required
result tables have been added.

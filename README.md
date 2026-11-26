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

```text
Group03_STAT3013_R11/
├── app/
│   └── streamlit_app.py      # Local Streamlit report UI
├── configs/
│   └── regions/
│       └── mekong.yaml       # Per-region settings (paths relative to project root)
├── data/
│   ├── raw/                  # Source datasets (not committed)
│   └── processed/            # Derived datasets
├── docs/                     # Project documentation
├── notebooks/                # Exploratory notebooks (nếu chốt chạy streamlit thì sẽ không cần notebook)
├── results/                  # Model outputs, one folder per model
│   ├── anomAR1/
│   ├── Chronus-2/
│   ├── GAM/
│   └── LightGBM/
├── src/                      # Model source code, one folder per model
│   ├── anomAR1/
│   ├── Chronus-2/
│   ├── GAM/
│   └── LightGBM/
├── tests/                    # Bộ test chung, đối với test riêng thì cập nhật vào thư mục test lớn, test riêng của model thì ở thư mục con của nó
├── pyproject.toml            # Project metadata
├── requirements.txt          # Pinned runtime dependencies
└── README.md
```

Note: Git does not track empty folders. Add a `.gitkeep` file to any empty
folder above that should exist in the repository.

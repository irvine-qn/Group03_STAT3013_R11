import pandas as pd

from floodbench.report_md import build_markdown, label, load, md_table


def test_label_handles_window_suffix():
    assert label("gam_w7") == "GAM [7 mùa gần nhất]"
    assert label("unknown") == "unknown"


def test_md_table_shape():
    out = md_table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})).splitlines()
    assert len(out) == 4 and out[1] == "|---|---|"


def test_report_numbers_come_from_tables():
    t = load()
    md = build_markdown(t)
    gam = t["probe_vs_anom_ar1"].query("model == 'gam' and h == 1").iloc[0]
    assert f"{gam.p_tost:.4f}" in md
    assert f"{t['probe_rmse'].set_index('model').loc['anom_ar1', '4']:,.0f}" in md

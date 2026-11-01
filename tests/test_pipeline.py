import pandas as pd
from src.generate_data import generate
from src.model import clean, run


def test_generate_shape_and_reproducible():
    a, b = generate(500, seed=1), generate(500, seed=1)
    assert len(a) == 500
    pd.testing.assert_frame_equal(a, b)
    assert set(a.churned.unique()) <= {0, 1}


def test_clean_fills_missing_totals():
    df = generate(2000, seed=3)
    assert df.total_charges.isna().any()
    assert not clean(df).total_charges.isna().any()


def test_model_beats_baseline(tmp_path):
    path = tmp_path / "c.csv"
    generate(3000, seed=7).to_csv(path, index=False)
    m = run(str(path), str(tmp_path / "out"))
    assert m["roc_auc"] > 0.75
    assert m["recall"] > 0.5
    assert (tmp_path / "out" / "SUMMARY.md").exists()


def test_model_recovers_true_drivers(tmp_path):
    path = tmp_path / "c.csv"
    generate(5000, seed=11).to_csv(path, index=False)
    m = run(str(path), str(tmp_path / "out"))
    risk = " ".join(m["top_risk_drivers"]).lower()
    assert "month-to-month" in risk or "fiber" in risk

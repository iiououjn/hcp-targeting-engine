import tempfile

import pandas as pd

from helpers import data
from hcp_engine.data import (HCPData, generate_synthetic, load_csv_dir, save_csv_dir,
                             validate)


def test_synthetic_is_valid():
    d = data()
    validate(d, 12, 3)
    assert d.profile["hcp_id"].is_unique
    assert d.rx_monthly["month"].between(1, 15).all()
    assert (d.rx_monthly["brand_trx"] <= d.rx_monthly["market_trx"]).all()


def test_synthetic_is_reproducible():
    a, b = generate_synthetic(200, seed=3), generate_synthetic(200, seed=3)
    pd.testing.assert_frame_equal(a.rx_monthly, b.rx_monthly)
    pd.testing.assert_frame_equal(a.engagement, b.engagement)


def test_validate_rejects_brand_above_market():
    d = generate_synthetic(50, seed=1)
    rx = d.rx_monthly.copy()
    rx.loc[0, "brand_trx"] = rx.loc[0, "market_trx"] + 5
    bad = HCPData(d.profile.copy(), rx, d.engagement.copy())
    try:
        validate(bad, 12, 3)
    except ValueError as e:
        assert "brand_trx > market_trx" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_validate_rejects_missing_months():
    d = generate_synthetic(50, seed=1)
    bad = HCPData(d.profile.copy(), d.rx_monthly[d.rx_monthly["month"] <= 12].copy(),
                  d.engagement.copy())
    try:
        validate(bad, 12, 3)
    except ValueError as e:
        assert "months" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_csv_roundtrip():
    d = generate_synthetic(80, seed=2)
    with tempfile.TemporaryDirectory() as tmp:
        save_csv_dir(d, tmp)
        back = load_csv_dir(tmp)
    validate(back, 12, 3)
    assert len(back.profile) == len(d.profile)
    assert back.truth is not None

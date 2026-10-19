import numpy as np
import pandas as pd

from helpers import data, features, labels
from hcp_engine.data import HCPData
from hcp_engine.features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, _wls_slope,
                                 build_features, build_labels, describe_feature)


def test_features_do_not_depend_on_outcome_months():
    """LEAKAGE GUARD: scrambling months 13-15 must not change a single feature."""
    d = data()
    rx = d.rx_monthly.copy()
    late = rx["month"] > 12
    rx.loc[late, "market_trx"] = 999
    rx.loc[late, "brand_trx"] = 500
    tampered = HCPData(d.profile, rx, d.engagement)
    pd.testing.assert_frame_equal(features(), build_features(tampered, 12))


def test_labels_do_not_depend_on_lookback_months():
    d = data()
    rx = d.rx_monthly.copy()
    early = rx["month"] <= 12
    rx.loc[early, "brand_trx"] = 0
    rx.loc[early, "market_trx"] = 0
    tampered = HCPData(d.profile, rx, d.engagement)
    pd.testing.assert_frame_equal(labels(), build_labels(tampered, 12, 3, 6))


def test_feature_schema_and_ranges():
    f = features()
    assert list(f.columns) == CATEGORICAL_FEATURES + NUMERIC_FEATURES
    assert f.index.is_unique
    assert f["brand_share"].between(0, 1).all()
    assert (f["addressable_trx_3m"] >= 0).all()
    assert f["brand_momentum"].between(-1, 5).all()
    nan_cols = set(f.columns[f.isna().any()])
    assert nan_cols <= {"years_in_practice", "formulary_coverage"}  # only raw-profile gaps


def test_wls_slope_recovers_known_slope():
    t = np.tile(np.arange(12, dtype=float), (3, 1))
    y = 0.5 + 0.02 * t
    slope = _wls_slope(y, t, np.ones_like(y))
    assert np.allclose(slope, 0.02)


def test_wls_slope_is_zero_without_evidence():
    t = np.tile(np.arange(12, dtype=float), (1, 1))
    y = np.zeros((1, 12))
    w = np.zeros((1, 12))
    w[0, :2] = 5  # only two active months
    assert _wls_slope(y, t, w)[0] == 0.0


def test_label_definition():
    y = labels()
    assert set(y["adopted"].unique()) <= {0, 1}
    assert ((y["brand_trx_outcome"] >= 6).astype(int) == y["adopted"]).all()


def test_describe_feature():
    assert describe_feature("email_open_rate", 0.623) == "Email open rate = 62%"
    assert describe_feature("region", "West") == "Region: West"
    assert describe_feature("formulary_coverage", float("nan")).endswith("missing")

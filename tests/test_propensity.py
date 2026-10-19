import numpy as np

from helpers import prospect_xy
from hcp_engine.propensity import PropensityModel, evaluate_scores, gains_table


def test_cv_beats_chance_and_oof_is_complete():
    X, y = prospect_xy()
    cv = PropensityModel(seed=1).cross_validate(X, y, n_splits=3)
    assert len(cv.oof) == len(y)
    assert ((cv.oof >= 0) & (cv.oof <= 1)).all()
    assert cv.metrics["roc_auc"] > 0.70
    assert cv.metrics["lift_top_decile"] > 1.5
    assert len(cv.fold_auc) == 3


def test_fit_predict_and_feature_matrix():
    X, y = prospect_xy()
    m = PropensityModel(seed=1).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()
    Xm = m.transform(X)
    assert not Xm.isna().any().any()                       # imputation worked
    assert any(c.startswith("region_") for c in Xm.columns)  # one-hot present
    assert np.allclose(m.raw_score(X), np.log(p / (1 - p)), atol=1e-6)


def test_excluded_features_are_not_used():
    X, y = prospect_xy()
    m = PropensityModel(exclude_features=("years_in_practice", "region"), seed=1).fit(X, y)
    cols = list(m.transform(X).columns)
    assert "years_in_practice" not in cols
    assert not any(c.startswith("region_") for c in cols)


def test_evaluate_scores_on_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    p = np.r_[np.linspace(0.99, 0.9, 10), np.linspace(0.1, 0.01, 90)]
    m = evaluate_scores(y, p)
    assert m["roc_auc"] == 1.0
    assert m["precision_top_decile"] == 1.0
    assert abs(m["lift_top_decile"] - 10.0) < 1e-9


def test_gains_table_is_consistent():
    X, y = prospect_xy()
    cv = PropensityModel(seed=1).cross_validate(X, y, n_splits=3)
    g = gains_table(y, cv.oof)
    assert g["n"].sum() == len(y)
    assert g["positives"].sum() == y.sum()
    assert abs(g["cum_capture"].iloc[-1] - 1) < 1e-9
    assert g["lift"].iloc[0] > g["lift"].iloc[-1]

import importlib.util

import numpy as np
import pandas as pd
import pytest

from helpers import prospect_xy
from hcp_engine.explain import group_by_original, to_2d, top_contributors
from hcp_engine.propensity import PropensityModel

HAS_SHAP = importlib.util.find_spec("shap") is not None


def test_to_2d_handles_all_shap_shapes():
    a = np.arange(12, dtype=float).reshape(4, 3)
    assert to_2d(a).shape == (4, 3)
    assert to_2d([a * 0, a]).shape == (4, 3)               # list of per-class arrays
    assert (to_2d([a * 0, a]) == a).all()
    assert (to_2d(np.stack([a * 0, a], axis=-1)) == a).all()  # (n, f, classes)


def test_group_by_original_sums_one_hot_columns():
    sv = pd.DataFrame({"region_West": [0.2, -0.1], "region_South": [0.1, 0.3],
                       "email_open_rate": [0.5, 0.0], "specialty_NP/PA": [0.0, 0.4]})
    g = group_by_original(sv, ["region", "specialty"])
    assert set(g.columns) == {"region", "email_open_rate", "specialty"}
    assert np.allclose(g["region"], [0.3, 0.2])
    assert np.allclose(g.sum(axis=1), sv.sum(axis=1))      # total contribution preserved


def test_top_contributors_text_and_thresholds():
    grouped = pd.DataFrame({"email_open_rate": [0.6, -0.4], "region": [0.3, 0.01],
                            "brand_trx": [-0.2, 0.5]}, index=["a", "b"])
    X = pd.DataFrame({"email_open_rate": [0.62, 0.1], "region": ["West", "South"],
                      "brand_trx": [2, 5]}, index=["a", "b"])
    up = top_contributors(grouped, X, k=2, positive=True)
    assert up["a"] == "Email open rate = 62%; Region: West"
    assert up["b"] == "Brand TRx (lookback) = 5"           # 0.01 < min_abs, so region omitted
    down = top_contributors(grouped, X, k=2, positive=False)
    assert down["a"] == "Brand TRx (lookback) = 2"
    assert down["b"] == "Email open rate = 10%"


@pytest.mark.skipif(not HAS_SHAP, reason="shap not installed")
def test_shap_values_are_additive():
    from hcp_engine.explain import PropensityExplainer
    X, y = prospect_xy()
    model = PropensityModel(seed=1).fit(X, y)
    ex = PropensityExplainer(model).explain(X.iloc[:200])
    recon = ex.shap_values.sum(axis=1).to_numpy() + ex.base_value
    assert np.allclose(recon, model.raw_score(X.iloc[:200]), atol=1e-4)
    assert not ex.global_importance().empty

from __future__ import annotations

import pytest

from helpers import prospect_xy
from hcp_engine_plus.tuning import tune_on_training


def test_bounded_tuning_uses_training_only():
    X, y = prospect_xy()
    result = tune_on_training(X, y, folds=2,
                              grid={"gbm__n_estimators": [20, 30], "gbm__max_depth": [2]})
    assert result.candidates_tested == 2
    assert 0.5 <= result.training_cv_auc <= 1.0
    assert result.model.predict_proba(X.iloc[:2]).shape == (2,)
    with pytest.raises(ValueError, match="gbm__"):
        tune_on_training(X, y, grid={"prep__random": [1]})

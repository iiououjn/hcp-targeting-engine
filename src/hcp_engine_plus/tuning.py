"""Optional, bounded tuning using ONLY the training period; holdouts stay untouched."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from hcp_engine.propensity import PropensityModel


@dataclass
class TuningResult:
    model: PropensityModel
    best_params: dict
    training_cv_auc: float
    candidates_tested: int


def tune_on_training(
    X: pd.DataFrame, y: pd.Series, *, seed: int = 42, folds: int = 3,
    grid: dict | None = None,
) -> TuningResult:
    """Select booster hyperparameters from an EARLIER labeled training window only.

    Never choose settings from the final temporal test cohort; random CV here is
    for parameter selection, NOT an unbiased estimate of forward performance.
    """
    if y.nunique() < 2 or min(y.value_counts()) < folds:
        raise ValueError("Need at least folds examples of each outcome class")
    model = PropensityModel(seed=seed)
    candidates = grid or {
        "gbm__n_estimators": [150, 300],
        "gbm__learning_rate": [0.03, 0.07],
        "gbm__max_depth": [2, 3],
    }
    if not candidates or any(not str(k).startswith("gbm__") for k in candidates):
        raise ValueError("Tuning grid may only contain gbm__ prefixed parameters")
    folds_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = GridSearchCV(model._build_gbm(), candidates, scoring="roc_auc",
                          cv=folds_cv, n_jobs=1, refit=False, error_score="raise")
    search.fit(X[model.columns], y)
    model.gbm_params.update({k.removeprefix("gbm__"): v for k, v in search.best_params_.items()})
    model.fit(X, y)
    return TuningResult(model, search.best_params_, float(search.best_score_),
                        len(search.cv_results_["params"]))

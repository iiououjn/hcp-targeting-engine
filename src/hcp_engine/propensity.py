"""Propensity-to-prescribe model.

* Gradient boosting (scikit-learn) on engineered behavioral + profile features.
* Stratified K-fold CV yields **out-of-fold (OOF)** probabilities, so every scored HCP
  receives a prediction from a model that never saw that HCP's label. Use OOF scores
  for ranking the training universe; use ``predict_proba`` for genuinely new HCPs.
* A regularized logistic regression is cross-validated alongside as a sanity baseline:
  if boosting can't beat it clearly, prefer the simpler model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hcp_engine.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def evaluate_scores(y, p, top_frac: float = 0.10) -> dict[str, float]:
    """Discrimination, calibration and targeting-oriented metrics for probability scores."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    prev = float(y.mean())
    n_top = max(1, int(round(len(y) * top_frac)))
    top = np.argsort(-p, kind="stable")[:n_top]
    prec = float(y[top].mean())
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "prevalence": prev,
        "precision_top_decile": prec,
        "lift_top_decile": prec / prev if prev > 0 else float("nan"),
    }


def gains_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Decile gains: rate, lift and cumulative share of positives captured."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    df = df.sort_values("p", ascending=False, kind="stable").reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    g = df.groupby("decile").agg(n=("y", "size"), positives=("y", "sum"), mean_score=("p", "mean"))
    g["rate"] = g["positives"] / g["n"]
    g["lift"] = g["rate"] / df["y"].mean()
    g["cum_capture"] = g["positives"].cumsum() / df["y"].sum()
    return g.reset_index()


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed rate in equal-count score bins."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"),
                                 n=("y", "size")).reset_index()


@dataclass
class CVResult:
    oof: np.ndarray
    metrics: dict[str, float]
    baseline_oof: np.ndarray
    baseline_metrics: dict[str, float]
    fold_auc: list[float] = field(default_factory=list)


class PropensityModel:
    def __init__(self, exclude_features: tuple[str, ...] = (), seed: int = 42,
                 gbm_params: dict | None = None):
        self.seed = seed
        drop = set(exclude_features)
        self.numeric = [c for c in NUMERIC_FEATURES if c not in drop]
        self.categorical = [c for c in CATEGORICAL_FEATURES if c not in drop]
        self.columns = self.categorical + self.numeric
        self.gbm_params = {
            "n_estimators": 400, "learning_rate": 0.04, "max_depth": 3, "subsample": 0.8,
            "min_samples_leaf": 25, "validation_fraction": 0.15, "n_iter_no_change": 25,
            **(gbm_params or {}),
        }

    # ------------------------------------------------------------------- builders
    def _onehot(self) -> OneHotEncoder:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def _build_gbm(self) -> Pipeline:
        prep = ColumnTransformer(
            [("num", SimpleImputer(strategy="median"), self.numeric),
             ("cat", self._onehot(), self.categorical)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        clf = GradientBoostingClassifier(random_state=self.seed, **self.gbm_params)
        return Pipeline([("prep", prep), ("gbm", clf)])

    def _build_baseline(self) -> Pipeline:
        prep = ColumnTransformer(
            [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), self.numeric),
             ("cat", self._onehot(), self.categorical)],
        )
        return Pipeline([("prep", prep),
                         ("lr", LogisticRegression(C=0.5, max_iter=2000))])

    # ------------------------------------------------------------------------- CV
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> CVResult:
        X = X[self.columns]
        y = pd.Series(np.asarray(y).astype(int), index=X.index)
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(y))
        base = np.zeros(len(y))
        fold_auc: list[float] = []
        for tr, te in cv.split(X, y):
            m = self._build_gbm().fit(X.iloc[tr], y.iloc[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
            fold_auc.append(float(roc_auc_score(y.iloc[te], oof[te])))
            b = self._build_baseline().fit(X.iloc[tr], y.iloc[tr])
            base[te] = b.predict_proba(X.iloc[te])[:, 1]
        return CVResult(oof, evaluate_scores(y, oof), base, evaluate_scores(y, base), fold_auc)

    # ------------------------------------------------------------------ fit/predict
    def fit(self, X: pd.DataFrame, y: pd.Series) -> PropensityModel:
        self.pipeline_ = self._build_gbm().fit(X[self.columns], np.asarray(y).astype(int))
        return self

    @property
    def estimator_(self) -> GradientBoostingClassifier:
        return self.pipeline_.named_steps["gbm"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Model-ready feature matrix (imputed numerics + one-hot categoricals)."""
        return self.pipeline_.named_steps["prep"].transform(X[self.columns])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X[self.columns])[:, 1]

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Log-odds output of the booster (the space SHAP values add up in)."""
        return self.estimator_.decision_function(self.transform(X))

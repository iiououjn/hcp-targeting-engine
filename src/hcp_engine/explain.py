"""SHAP explanations for the propensity model.

Design notes
------------
* ``TreeExplainer`` is run on the fitted gradient-boosting model. SHAP values are in
  log-odds space and satisfy ``base_value + sum(shap) == raw_score`` per HCP.
* The model sees one-hot columns (``region_West`` ...). For *reporting*, contributions are
  summed back to the original feature (``region``) so reason codes read naturally.
* The grouping and reason-code functions are pure (no ``shap`` import), so they are
  unit-testable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hcp_engine.features import describe_feature
from hcp_engine.propensity import PropensityModel


def to_2d(values) -> np.ndarray:
    """Normalize the many shapes ``shap`` returns for binary models to (n_samples, n_features).

    Depending on the shap version / model, the result can be a list of per-class arrays or
    a 3-D array (samples, features, classes). We always want the positive class.
    """
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    return values


def group_by_original(sv: pd.DataFrame, categorical: Sequence[str]) -> pd.DataFrame:
    """Sum SHAP values of one-hot columns back to their original categorical feature."""
    cats = sorted(categorical, key=len, reverse=True)
    mapping: dict[str, str] = {}
    for col in sv.columns:
        owner = next((c for c in cats if col.startswith(c + "_")), col)
        mapping[col] = owner
    return sv.T.groupby(mapping).sum().T


def top_contributors(grouped: pd.DataFrame, X: pd.DataFrame, k: int = 3,
                     positive: bool = True, min_abs: float = 0.05) -> pd.Series:
    """Per-HCP text of the ``k`` strongest drivers pushing the score up (or down).

    ``min_abs`` (log-odds) suppresses negligible contributions so reason codes stay honest.
    """
    vals = grouped.to_numpy()
    cols = grouped.columns.to_numpy()
    raw = {c: X.loc[grouped.index, c].to_numpy() for c in cols}
    out: list[str] = []
    for i in range(len(grouped)):
        row = vals[i]
        order = np.argsort(-row if positive else row)[:k]
        keep = [j for j in order
                if (row[j] > min_abs if positive else row[j] < -min_abs)]
        out.append("; ".join(describe_feature(cols[j], raw[cols[j]][i]) for j in keep))
    return pd.Series(out, index=grouped.index)


@dataclass
class Explanation:
    shap_values: pd.DataFrame     # model space (one-hot), log-odds
    grouped: pd.DataFrame         # original-feature space, log-odds
    base_value: float
    X_model: pd.DataFrame         # model-ready matrix the SHAP values refer to

    def global_importance(self) -> pd.Series:
        """Mean |SHAP| per original feature (log-odds), descending."""
        imp = self.grouped.abs().mean().sort_values(ascending=False)
        return imp.rename("mean_abs_shap").rename_axis("feature")


class PropensityExplainer:
    def __init__(self, model: PropensityModel):
        try:
            import shap
        except ImportError as e:  # pragma: no cover
            raise ImportError("SHAP explanations need the `shap` package: pip install shap "
                              "(or run with Config(explain=False) / --no-explain).") from e
        self.model = model
        self.explainer = shap.TreeExplainer(model.estimator_)

    def explain(self, X: pd.DataFrame) -> Explanation:
        Xm = self.model.transform(X)
        sv = to_2d(self.explainer.shap_values(Xm, check_additivity=False))
        sv = pd.DataFrame(sv, index=X.index, columns=Xm.columns)
        base = float(np.ravel(self.explainer.expected_value)[-1])
        return Explanation(sv, group_by_original(sv, self.model.categorical), base, Xm)

"""Holdout-only probability calibration and operationally constrained call plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    """Post-fit calibration; calibrator MUST be fit on a disjoint labeled window.

    Train base model on period A; calibrate on period B; evaluate on period C.
    Reusing the training or evaluation cohort for fitting inflates reported quality.
    """

    method: str = "sigmoid"

    def fit(self, probabilities, outcomes) -> ProbabilityCalibrator:
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(outcomes, dtype=int)
        if (p.ndim != 1 or y.ndim != 1 or len(y) != len(p) or len(y) < 20
                or set(np.unique(y)) != {0, 1} or not np.isfinite(p).all()
                or np.any((p < 0) | (p > 1))):
            raise ValueError("Require >=20 valid held-out probabilities with both classes")
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        elif self.method == "sigmoid":
            clipped = np.clip(p, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            self.model_ = LogisticRegression().fit(logits, y)
        else:
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        return self

    def predict(self, probabilities) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("Call fit on held-out calibration data first")
        p = np.asarray(probabilities, dtype=float)
        if self.method == "isotonic":
            return np.asarray(self.model_.predict(p))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return self.model_.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def constrained_call_plan(
    candidates: pd.DataFrame,
    *,
    max_calls: int,
    territory_capacity: dict[str, int] | None = None,
    rep_capacity: dict[str, int] | None = None,
    max_cost: float | None = None,
    min_days_between_calls: int = 30,
) -> pd.DataFrame:
    """Transparent greedy baseline; not an optimal solver or a causal-value estimate.

    Required: hcp_id, propensity, addressable_trx_3m. Optional territory, rep_id,
    cost_per_call, days_since_last_call. If a constraint is requested, the
    corresponding column MUST exist: no silent bypass of territorial restrictions.
    Rows with missing / invalid scores or monetary costs are rejected.
    """
    if max_calls < 0 or min_days_between_calls < 0:
        raise ValueError("Capacity and call interval must be nonnegative")
    df = candidates.copy()
    required = {"hcp_id", "propensity", "addressable_trx_3m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["hcp_id"].duplicated().any():
        raise ValueError("Duplicate HCP IDs; each HCP can only appear once")
    for flag, col in [(territory_capacity is not None, "territory"),
                      (rep_capacity is not None, "rep_id"),
                      (max_cost is not None, "cost_per_call")]:
        if flag and col not in df:
            raise ValueError(f"Need {col} for requested constraint")
    if "days_since_last_call" not in df:
        raise ValueError("Need days_since_last_call to enforce call-frequency policy")
    df = df[df["propensity"].notna()].copy()
    if not (df["propensity"].between(0, 1).all()
            and df["addressable_trx_3m"].ge(0).all()):
        raise ValueError("Invalid propensity or addressable TRx")
    df = df[df["days_since_last_call"].ge(min_days_between_calls)]
    if "cost_per_call" not in df:
        df["cost_per_call"] = 0.0
    if df["cost_per_call"].isna().any() or (df["cost_per_call"] < 0).any():
        raise ValueError("Costs must be finite, nonnegative and known")
    df["ranking_value"] = df["propensity"] * df["addressable_trx_3m"]
    # Greedy net value only as a RANKING heuristic, not treatment uplift.
    df = df.sort_values(["ranking_value", "hcp_id"], ascending=[False, True])
    territory_used: dict[str, int] = {}
    rep_used: dict[str, int] = {}
    budget_used = 0.0
    rows: list[int] = []
    for idx, row in df.iterrows():
        if len(rows) >= max_calls:
            break
        terr, rep = str(row.get("territory", "")), str(row.get("rep_id", ""))
        if territory_capacity is not None and territory_used.get(terr, 0) >= territory_capacity.get(terr, 0):
            continue
        if rep_capacity is not None and rep_used.get(rep, 0) >= rep_capacity.get(rep, 0):
            continue
        cost = float(row["cost_per_call"])
        if max_cost is not None and budget_used + cost > max_cost + 1e-10:
            continue
        rows.append(idx)
        budget_used += cost
        territory_used[terr] = territory_used.get(terr, 0) + 1
        rep_used[rep] = rep_used.get(rep, 0) + 1
    out = df.loc[rows].copy()
    out["call_plan_rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)

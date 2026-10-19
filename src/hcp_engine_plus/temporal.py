"""Leakage-guarded as-of scoring and genuinely forward-time holdout validation.

Month integers denote a sequential calendar month index, NOT YYYYMM. Engagement
must be a point-in-time aggregate computed no later than each as-of cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hcp_engine.data import HCPData, validate
from hcp_engine.features import build_features, build_labels
from hcp_engine.propensity import PropensityModel, evaluate_scores


@dataclass
class TemporalHoldout:
    train_as_of: int
    test_as_of: int
    n_train: int
    n_test: int
    metrics: dict[str, float]
    predictions: pd.DataFrame
    model: PropensityModel


def asof_snapshot(
    data: HCPData,
    as_of_month: int,
    lookback_months: int = 12,
    outcome_months: int = 0,
    *,
    require_complete: bool = True,
) -> HCPData:
    """Slice absolute sequential Rx months into relative months 1..L(+O).

    The supplied engagement table MUST already be aggregated as of ``as_of_month``;
    no valid rolling reconstruction is possible from the original aggregate schema.
    A strictly complete HCP-month panel is required by default to prevent treating
    unknown missing Rx values as documented zeros.
    """
    if not isinstance(as_of_month, int) or lookback_months < 6 or outcome_months < 0:
        raise ValueError("Use integer as_of_month, lookback_months >= 6, outcome_months >= 0")
    start = as_of_month - lookback_months + 1
    stop = as_of_month + outcome_months
    if start < 1:
        raise ValueError("Not enough history before the as-of date")
    rx = data.rx_monthly.copy()
    if not pd.api.types.is_integer_dtype(rx["month"]):
        raise ValueError("month must be an integer sequential index (not YYYYMM)")
    if rx.duplicated(["hcp_id", "month"]).any():
        raise ValueError("Duplicate HCP-month rows; deduplicate/aggregate explicitly")
    rx = rx.loc[rx["month"].between(start, stop)].copy()
    profile = data.profile.copy()
    engagement = data.engagement.copy()
    if require_complete:
        months = set(range(start, stop + 1))
        if months - set(rx["month"]):
            raise ValueError("Missing whole calendar month(s) from the required window")
        expected = len(profile) * len(months)
        # require each HCP to have all months; zeros must be explicit.
        if len(rx) != expected or set(rx["hcp_id"].astype(str)) != set(profile["hcp_id"].astype(str)):
            raise ValueError("Incomplete HCP-month panel: supply explicit zero rows or fix missing data")
    rx["month"] = rx["month"] - start + 1
    snapshot = HCPData(profile, rx, engagement)
    if outcome_months:
        validate(snapshot, lookback_months, outcome_months)
    else:
        # The original validator expects a labeled forward period, so separately
        # validate a scoring-only snapshot's ID and domain constraints.
        if profile["hcp_id"].astype(str).duplicated().any():
            raise ValueError("Duplicate HCP profile IDs")
        if engagement["hcp_id"].astype(str).duplicated().any():
            raise ValueError("Duplicate HCP engagement IDs")
        if set(profile["hcp_id"].astype(str)) != set(engagement["hcp_id"].astype(str)):
            raise ValueError("Profile/engagement HCP IDs do not match")
        if (rx[["market_trx", "brand_trx"]].isna().any().any()
                or (rx[["market_trx", "brand_trx"]] < 0).any().any()
                or (rx["brand_trx"] > rx["market_trx"]).any()):
            raise ValueError("Invalid or missing Rx counts")
    return snapshot


def score_asof(
    model: PropensityModel,
    data: HCPData,
    as_of_month: int,
    lookback_months: int = 12,
    established_writer_trx: int = 6,
) -> pd.DataFrame:
    """Predict later HCPs without reading outcomes; model is fit *before* this call.

    Returns probabilities only for prospects; established brand writers remain NaN.
    These are uncalibrated scores unless the caller passed a calibrated model wrapper.
    """
    snap = asof_snapshot(data, as_of_month, lookback_months)
    features = build_features(snap, lookback_months)
    prospect = features["brand_trx_3m"] < established_writer_trx
    df = features[["region", "specialty", "brand_trx_3m", "addressable_trx_3m"]].copy()
    df["propensity"] = np.nan
    if prospect.any():
        df.loc[prospect, "propensity"] = model.predict_proba(features.loc[prospect])
    df["status"] = np.where(prospect, "Prospect", "Established writer")
    df["expected_value"] = df["propensity"] * df["addressable_trx_3m"]
    df["as_of_month"] = as_of_month
    return df.reset_index().sort_values("expected_value", ascending=False, na_position="last")


def temporal_holdout(
    train_data: HCPData,
    test_data: HCPData,
    train_as_of: int,
    test_as_of: int,
    *,
    lookback_months: int = 12,
    outcome_months: int = 3,
    adoption_trx: int = 6,
    established_writer_trx: int = 6,
    seed: int = 42,
) -> TemporalHoldout:
    """Fit on earlier labels, evaluate on a later, unseen forward outcome window.

    Enforces that the training labels end BEFORE the test feature cutoff.
    Engagement for each snapshot must be captured *at that snapshot's cutoff*.
    Same HCP IDs across time are valid; this evaluates prospective deployment,
    not cold-start generalization. No temporal holdout is inferred from a single quarter.
    """
    if test_as_of <= train_as_of + outcome_months:
        raise ValueError("Test as-of must be AFTER the last training outcome month")
    train = asof_snapshot(train_data, train_as_of, lookback_months, outcome_months)
    test = asof_snapshot(test_data, test_as_of, lookback_months, outcome_months)
    X_train = build_features(train, lookback_months)
    y_train = build_labels(train, lookback_months, outcome_months, adoption_trx)["adopted"]
    X_test = build_features(test, lookback_months)
    y_test = build_labels(test, lookback_months, outcome_months, adoption_trx)["adopted"]
    tr = X_train["brand_trx_3m"] < established_writer_trx
    te = X_test["brand_trx_3m"] < established_writer_trx
    if y_train.loc[tr].nunique() < 2 or y_test.loc[te].nunique() < 2:
        raise ValueError("Both training and test prospects must have positive and negative labels")
    model = PropensityModel(seed=seed).fit(X_train.loc[tr], y_train.loc[tr])
    p = model.predict_proba(X_test.loc[te])
    metrics = evaluate_scores(y_test.loc[te], p)
    pred = pd.DataFrame({
        "hcp_id": X_test.loc[te].index,
        "as_of_month": test_as_of,
        "adopted": y_test.loc[te].to_numpy(),
        "propensity": p,
    })
    return TemporalHoldout(train_as_of, test_as_of, int(tr.sum()), int(te.sum()),
                           metrics, pred, model)

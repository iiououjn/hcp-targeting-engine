"""Behavioral feature engineering and label construction.

Leakage guard: ``build_features`` only reads months ``1..lookback_months`` of the
Rx panel; ``build_labels`` only reads the months after that. The two never overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcp_engine.data import HCPData

RECENT_MONTHS = 3  # "recent" window used for 3-month features

CATEGORICAL_FEATURES = ["specialty", "practice_setting", "region"]
NUMERIC_FEATURES = [
    "years_in_practice", "formulary_coverage",
    "market_trx", "market_trx_3m", "market_growth", "market_cv",
    "brand_trx", "brand_trx_3m", "brand_share", "brand_share_3m",
    "share_trend", "brand_momentum", "months_with_brand_rx", "addressable_trx_3m",
    "calls_per_month", "rep_calls_recent", "days_since_last_call",
    "samples_requested", "samples_per_call",
    "emails_sent", "email_open_rate", "digital_events", "speaker_programs",
]

# Human-readable labels + format strings, used for SHAP "reason codes".
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "years_in_practice": ("Years in practice", "{:.0f}"),
    "formulary_coverage": ("Formulary coverage", "{:.0%}"),
    "market_trx": ("Class TRx (lookback)", "{:,.0f}"),
    "market_trx_3m": ("Class TRx (last 3m)", "{:,.0f}"),
    "market_growth": ("Class volume growth/mo", "{:+.1%}"),
    "market_cv": ("Prescribing volatility", "{:.2f}"),
    "brand_trx": ("Brand TRx (lookback)", "{:,.0f}"),
    "brand_trx_3m": ("Brand TRx (last 3m)", "{:,.0f}"),
    "brand_share": ("Brand share (lookback)", "{:.1%}"),
    "brand_share_3m": ("Brand share (last 3m)", "{:.1%}"),
    "share_trend": ("Brand share trend/mo", "{:+.2%}"),
    "brand_momentum": ("Brand momentum (last 3m vs prior 3m)", "{:+.0%}"),
    "months_with_brand_rx": ("Months with a brand Rx", "{:.0f}"),
    "addressable_trx_3m": ("Addressable TRx (last 3m)", "{:,.0f}"),
    "calls_per_month": ("Rep calls/month", "{:.1f}"),
    "rep_calls_recent": ("Rep calls (last 3m)", "{:.0f}"),
    "days_since_last_call": ("Days since last rep call", "{:.0f}"),
    "samples_requested": ("Samples requested", "{:.0f}"),
    "samples_per_call": ("Samples per call", "{:.1f}"),
    "emails_sent": ("Emails sent", "{:.0f}"),
    "email_open_rate": ("Email open rate", "{:.0%}"),
    "digital_events": ("Digital events attended", "{:.0f}"),
    "speaker_programs": ("Speaker programs attended", "{:.0f}"),
}
CATEGORICAL_LABELS = {"specialty": "Specialty", "practice_setting": "Setting", "region": "Region"}


def describe_feature(name: str, value) -> str:
    """Render ``(feature, value)`` as e.g. ``'Email open rate = 62%'``."""
    if name in CATEGORICAL_LABELS:
        return f"{CATEGORICAL_LABELS[name]}: {value}"
    label, fmt = FEATURE_LABELS.get(name, (name, "{:.2f}"))
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f"{label} = missing"
        return f"{label} = {fmt.format(float(value))}"
    except (TypeError, ValueError):
        return f"{label} = {value}"


# --------------------------------------------------------------------------- #
def _wls_slope(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise weighted least-squares slope of ``y`` on ``t`` (rows: HCPs).

    Rows with fewer than 3 positively-weighted points get slope 0 (no evidence of trend).
    """
    sw = w.sum(axis=1)
    safe = np.where(sw > 0, sw, 1.0)
    tbar = (w * t).sum(axis=1) / safe
    ybar = (w * y).sum(axis=1) / safe
    num = (w * (t - tbar[:, None]) * (y - ybar[:, None])).sum(axis=1)
    den = (w * (t - tbar[:, None]) ** 2).sum(axis=1)
    slope = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    slope[(w > 0).sum(axis=1) < 3] = 0.0
    return slope


def _wide(rx: pd.DataFrame, value: str, ids: np.ndarray, months: range) -> np.ndarray:
    wide = rx.pivot_table(index="hcp_id", columns="month", values=value, aggfunc="sum")
    return wide.reindex(index=ids, columns=list(months)).fillna(0.0).to_numpy(dtype=float)


def build_features(data: HCPData, lookback_months: int = 12) -> pd.DataFrame:
    """One row per HCP, indexed by ``hcp_id``, using only lookback-window information."""
    L, R = lookback_months, RECENT_MONTHS
    prof = data.profile.copy()
    prof["hcp_id"] = prof["hcp_id"].astype(str)
    ids = prof["hcp_id"].to_numpy()

    rx = data.rx_monthly.copy()
    rx["hcp_id"] = rx["hcp_id"].astype(str)
    rx = rx[(rx["month"] >= 1) & (rx["month"] <= L)]        # <-- leakage guard
    market = _wide(rx, "market_trx", ids, range(1, L + 1))
    brand = _wide(rx, "brand_trx", ids, range(1, L + 1))
    t = np.arange(L, dtype=float)[None, :] * np.ones((len(ids), 1))

    market_12 = market.sum(1)
    brand_12 = brand.sum(1)
    market_3 = market[:, -R:].sum(1)
    brand_3 = brand[:, -R:].sum(1)
    prev_3 = brand[:, -2 * R:-R].sum(1)

    share_m = np.divide(brand, market, out=np.zeros_like(brand), where=market > 0)
    share_trend = _wls_slope(share_m, t, market)            # weighted by monthly volume
    ones = np.ones_like(market)
    vol_slope = _wls_slope(market, t, ones)
    mean_mkt = market.mean(1)
    market_growth = np.divide(vol_slope, mean_mkt, out=np.zeros_like(mean_mkt), where=mean_mkt > 0)
    market_cv = np.divide(market.std(1), mean_mkt, out=np.zeros_like(mean_mkt), where=mean_mkt > 0)

    eng = data.engagement.copy()
    eng["hcp_id"] = eng["hcp_id"].astype(str)
    eng = eng.set_index("hcp_id").reindex(ids)
    calls = eng["rep_calls"].fillna(0).to_numpy(float)
    sent = eng["emails_sent"].fillna(0).to_numpy(float)
    opened = eng["emails_opened"].fillna(0).to_numpy(float)
    samples = eng["samples_requested"].fillna(0).to_numpy(float)

    feats = pd.DataFrame(index=pd.Index(ids, name="hcp_id"))
    for c in CATEGORICAL_FEATURES:
        feats[c] = prof[c].to_numpy()
    feats["years_in_practice"] = prof["years_in_practice"].to_numpy(float)
    feats["formulary_coverage"] = prof["formulary_coverage"].to_numpy(float)

    feats["market_trx"] = market_12
    feats["market_trx_3m"] = market_3
    feats["market_growth"] = market_growth
    feats["market_cv"] = market_cv
    feats["brand_trx"] = brand_12
    feats["brand_trx_3m"] = brand_3
    feats["brand_share"] = np.divide(brand_12, market_12, out=np.zeros_like(market_12),
                                     where=market_12 > 0)
    feats["brand_share_3m"] = np.divide(brand_3, market_3, out=np.zeros_like(market_3),
                                        where=market_3 > 0)
    feats["share_trend"] = share_trend
    feats["brand_momentum"] = np.clip((brand_3 - prev_3) / (prev_3 + 1.0), -1.0, 5.0)
    feats["months_with_brand_rx"] = (brand > 0).sum(1)
    feats["addressable_trx_3m"] = market_3 - brand_3

    feats["calls_per_month"] = calls / L
    feats["rep_calls_recent"] = eng["rep_calls_recent"].fillna(0).to_numpy(float)
    feats["days_since_last_call"] = eng["days_since_last_call"].fillna(365).to_numpy(float)
    feats["samples_requested"] = samples
    feats["samples_per_call"] = samples / np.maximum(calls, 1.0)
    feats["emails_sent"] = sent
    feats["email_open_rate"] = np.divide(opened, sent, out=np.zeros_like(sent), where=sent > 0)
    feats["digital_events"] = eng["digital_events"].fillna(0).to_numpy(float)
    feats["speaker_programs"] = eng["speaker_programs"].fillna(0).to_numpy(float)
    return feats[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def build_labels(data: HCPData, lookback_months: int = 12, outcome_months: int = 3,
                 adoption_trx: int = 6) -> pd.DataFrame:
    """Forward-window outcome: brand TRx in months ``L+1..L+O`` and the binary label."""
    rx = data.rx_monthly.copy()
    rx["hcp_id"] = rx["hcp_id"].astype(str)
    lo, hi = lookback_months + 1, lookback_months + outcome_months
    fwd = rx[(rx["month"] >= lo) & (rx["month"] <= hi)]
    out = fwd.groupby("hcp_id")["brand_trx"].sum()
    ids = data.profile["hcp_id"].astype(str)
    out = out.reindex(ids).fillna(0).astype(int)
    out.index.name = "hcp_id"
    return pd.DataFrame({"brand_trx_outcome": out,
                         "adopted": (out >= adoption_trx).astype(int)})

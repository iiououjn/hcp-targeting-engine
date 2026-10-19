"""Turn segments + propensity scores into an actionable targeting table.

Logic
-----
* **Established writers** (already above the writer threshold) are not scored; they are
  routed to "Retain & defend", with an at-risk flag if their momentum is falling.
* **Prospects** get ``expected_value = propensity x addressable_trx_3m`` -- a transparent
  ranking heuristic (expected addressable scripts), not a forecast of incremental Rx.
* Prospects are placed on a propensity x potential matrix to suggest a play, tiered by
  expected-value rank, and the top ``call_capacity`` form the call plan.
* A simple channel hint compares each HCP's digital vs rep engagement percentile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcp_engine.config import Config

ACTIONS = {
    (True, True): "Prioritize: high propensity, high potential",
    (True, False): "Nurture: high propensity, low potential (low-touch)",
    (False, True): "Persuade: high potential, low propensity (evidence / peer influence)",
    (False, False): "Deprioritize",
}


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def channel_hint(features: pd.DataFrame) -> pd.Series:
    """Digital-first vs rep-led vs omnichannel, from relative engagement percentiles."""
    digital = (_pct_rank(features["email_open_rate"]) + _pct_rank(features["digital_events"])) / 2
    rep = (_pct_rank(features["calls_per_month"]) + _pct_rank(features["samples_requested"])) / 2
    diff = digital - rep
    return pd.Series(np.where(diff > 0.15, "Digital-first",
                              np.where(diff < -0.15, "Rep-led", "Omnichannel")),
                     index=features.index, name="channel_hint")


def build_target_table(features: pd.DataFrame, segments: pd.Series, segment_names: pd.Series,
                       propensity: pd.Series, cfg: Config,
                       reasons: pd.Series | None = None,
                       barriers: pd.Series | None = None) -> pd.DataFrame:
    """One row per HCP; ``propensity`` is NaN for HCPs outside the scoring universe."""
    df = features[["specialty", "practice_setting", "region", "market_trx_3m",
                   "brand_trx_3m", "brand_share_3m", "addressable_trx_3m",
                   "brand_momentum", "share_trend"]].copy()
    df["segment_id"] = segments
    df["segment"] = segment_names
    df["propensity"] = propensity.reindex(df.index)
    established = df["brand_trx_3m"] >= cfg.established_writer_trx
    df["status"] = np.where(established, "Established writer", "Prospect")
    prospect = ~established

    df["expected_value"] = df["propensity"] * df["addressable_trx_3m"]

    # Propensity x potential matrix (thresholds defined on the prospect population)
    p_thr = df.loc[prospect, "propensity"].quantile(cfg.high_propensity_quantile)
    v_thr = df.loc[prospect, "addressable_trx_3m"].quantile(cfg.high_potential_quantile)
    hi_p = df["propensity"] >= p_thr
    hi_v = df["addressable_trx_3m"] >= v_thr
    action = pd.Series([ACTIONS[(bool(a), bool(b))] for a, b in zip(hi_p, hi_v, strict=True)],
                       index=df.index)
    at_risk = established & (df["brand_momentum"] < -0.25) & (df["share_trend"] < 0)
    action[established] = "Retain & defend"
    action[at_risk] = "Retain & defend: at-risk writer (declining momentum)"
    df["recommended_play"] = action

    # Tiers by expected-value rank among prospects
    rank = df.loc[prospect, "expected_value"].rank(ascending=False, method="first")
    pct = rank / prospect.sum()
    tier = pd.Series("Defend", index=df.index, dtype=object)
    tier[prospect] = np.where(pct <= cfg.tier1_pct, "Tier 1",
                              np.where(pct <= cfg.tier2_pct, "Tier 2", "Tier 3"))
    df["tier"] = tier
    df["priority_rank"] = rank.reindex(df.index)
    df["in_call_plan"] = df["priority_rank"] <= cfg.call_capacity

    df["channel_hint"] = channel_hint(features)
    df["why_target"] = reasons.reindex(df.index) if reasons is not None else ""
    df["watch_outs"] = barriers.reindex(df.index) if barriers is not None else ""
    return df.reset_index().sort_values(["priority_rank", "hcp_id"], na_position="last",
                                        ignore_index=True)


def segment_opportunity(table: pd.DataFrame) -> pd.DataFrame:
    """Segment-level rollup: size, prospect mix, mean propensity and call-plan share."""
    t = table.assign(_est=table["status"] == "Established writer")
    g = t.groupby(["segment_id", "segment"])
    prospects = t[t["status"] == "Prospect"].groupby(["segment_id", "segment"])
    out = pd.DataFrame({
        "n_hcps": g.size(),
        "n_prospects": prospects.size(),
        "n_established": g["_est"].sum().astype(int),
        "mean_propensity": prospects["propensity"].mean(),
        "total_expected_value": prospects["expected_value"].sum(),
        "in_call_plan": g["in_call_plan"].sum(),
    }).fillna({"n_prospects": 0, "mean_propensity": np.nan, "total_expected_value": 0.0})
    out["n_prospects"] = out["n_prospects"].astype(int)
    return out.reset_index().sort_values("total_expected_value", ascending=False,
                                         ignore_index=True)

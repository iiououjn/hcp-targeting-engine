"""Descriptive HCP data quality, group score audit and snapshot drift utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcp_engine.data import HCPData


def quality_report(data: HCPData, expected_months: range | None = None) -> dict:
    """Report ID mismatches, duplicates and missing panel cells without mutating data."""
    p = data.profile
    e = data.engagement
    rx = data.rx_monthly
    ids = set(p["hcp_id"].astype(str))
    months = (set(expected_months) if expected_months is not None
              else set(range(int(rx["month"].min()), int(rx["month"].max()) + 1)))
    present = set(zip(rx["hcp_id"].astype(str), rx["month"], strict=True))
    return {
        "n_hcps": len(ids),
        "n_months_expected": len(months),
        "duplicate_profile_ids": int(p["hcp_id"].astype(str).duplicated().sum()),
        "duplicate_engagement_ids": int(e["hcp_id"].astype(str).duplicated().sum()),
        "duplicate_hcp_months": int(rx.duplicated(["hcp_id", "month"]).sum()),
        "unknown_rx_ids": len(set(rx["hcp_id"].astype(str)) - ids),
        "missing_engagement_ids": len(ids - set(e["hcp_id"].astype(str))),
        "missing_hcp_month_cells": sum((hid, month) not in present for hid in ids for month in months),
        "missing_rx_values": int(rx[["market_trx", "brand_trx"]].isna().sum().sum()),
    }


def group_score_audit(table: pd.DataFrame, by: str, min_group_size: int = 20) -> pd.DataFrame:
    """Descriptive disparity audit; does NOT establish demographic fairness.

    Filters to scored records. Review with approved demographic/proxy policies,
    clinically justified comparisons and minimum sample sizes.
    """
    if by not in table or "propensity" not in table:
        raise ValueError(f"Need {by!r} and 'propensity' columns")
    t = table.loc[table["propensity"].notna(), [by, "propensity"]].copy()
    groups = t.groupby(by, dropna=False)["propensity"].agg(n="size", mean_score="mean",
                                                            median_score="median")
    groups = groups.loc[groups["n"] >= min_group_size]
    if groups.empty:
        return groups.assign(mean_ratio_to_overall=pd.Series(dtype=float)).reset_index()
    denom = max(float(t["propensity"].mean()), 1e-12)
    groups["mean_ratio_to_overall"] = groups["mean_score"] / denom
    return groups.reset_index().sort_values("mean_score")


def population_stability_index(reference: pd.Series, current: pd.Series,
                               n_bins: int = 10) -> float:
    """PSI using reference quantiles; clip empty bins to avoid infinities.

    PSI is a heuristic and must be interpreted with sample sizes and data lineage.
    """
    ref = np.asarray(reference.dropna(), dtype=float)
    cur = np.asarray(current.dropna(), dtype=float)
    if len(ref) < n_bins or len(cur) < n_bins:
        raise ValueError("Not enough records to estimate PSI")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        # Constant distributions: use a small two-sided interval around baseline.
        center = float(ref[0])
        edges = np.array([-np.inf, center, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    rp = np.histogram(ref, bins=edges)[0] / len(ref)
    cp = np.histogram(cur, bins=edges)[0] / len(cur)
    rp, cp = np.clip(rp, 1e-6, None), np.clip(cp, 1e-6, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def segment_migration(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Counts HCP transitions across snapshots; segment IDs must share a fitted model."""
    for t in (previous, current):
        if not {"hcp_id", "segment_id"}.issubset(t.columns):
            raise ValueError("Both snapshots need hcp_id and segment_id")
        if t["hcp_id"].duplicated().any():
            raise ValueError("Each snapshot must have unique HCP IDs")
    joined = previous[["hcp_id", "segment_id"]].merge(
        current[["hcp_id", "segment_id"]], on="hcp_id", suffixes=("_from", "_to"))
    return (joined.groupby(["segment_id_from", "segment_id_to"])
            .size().rename("n_hcps").reset_index())

"""Central configuration. Every tunable lives here so runs are reproducible."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42

    # --- time windows (in months) -------------------------------------------------
    # Features are built ONLY from months 1..lookback_months.
    # The label is built ONLY from the following `outcome_months`. No overlap => no leakage.
    lookback_months: int = 12
    outcome_months: int = 3

    # --- target definition --------------------------------------------------------
    # HCPs already writing >= established_writer_trx brand scripts in the last 3
    # lookback months are "established writers": they are segmented but excluded
    # from the propensity universe (the question "will they start?" is moot).
    established_writer_trx: int = 6
    # Label = 1 if a prospect writes >= adoption_trx brand scripts in the outcome window.
    adoption_trx: int = 6

    # --- segmentation -------------------------------------------------------------
    k: int | None = None            # None => choose k automatically
    k_min: int = 4
    k_max: int = 8
    min_stability: float = 0.75     # min bootstrap ARI for a k to be eligible

    # --- propensity model ---------------------------------------------------------
    cv_folds: int = 5
    # Drop features you do not want the model to use (e.g. fairness/compliance review).
    exclude_features: tuple[str, ...] = ()

    # --- targeting ----------------------------------------------------------------
    call_capacity: int = 300            # how many prospects the field force can cover
    tier1_pct: float = 0.10             # top 10% of prospects by expected value
    tier2_pct: float = 0.30             # next 20% (cumulative 30%)
    high_propensity_quantile: float = 0.60
    high_potential_quantile: float = 0.50
    top_reasons: int = 3                # SHAP reason codes per HCP

    # --- toggles ------------------------------------------------------------------
    explain: bool = True                # compute SHAP explanations / reason codes

    def to_dict(self) -> dict:
        return asdict(self)

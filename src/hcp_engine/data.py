"""Data container, schema validation, CSV I/O and a synthetic data generator.

Three input tables are expected (see docs/data_schema.md):

* ``profile``     one row per HCP (specialty, setting, region, ...)
* ``rx_monthly``  one row per HCP-month with market (class) and brand TRx
* ``engagement``  one row per HCP with commercial-touch aggregates over the lookback window

The synthetic generator draws HCPs from five latent behavioral archetypes. The
archetype labels are kept in ``HCPData.truth`` for validation only and are never
used as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_COLS = [
    "hcp_id", "specialty", "practice_setting", "region",
    "years_in_practice", "formulary_coverage",
]
RX_COLS = ["hcp_id", "month", "market_trx", "brand_trx"]
ENGAGEMENT_COLS = [
    "hcp_id", "rep_calls", "rep_calls_recent", "days_since_last_call",
    "samples_requested", "emails_sent", "emails_opened",
    "digital_events", "speaker_programs",
]
FILES = {"profile": "hcp_profile.csv", "rx_monthly": "rx_monthly.csv",
         "engagement": "engagement.csv"}


@dataclass
class HCPData:
    profile: pd.DataFrame
    rx_monthly: pd.DataFrame
    engagement: pd.DataFrame
    truth: pd.DataFrame | None = None   # synthetic only: latent archetype per HCP


# --------------------------------------------------------------------------- #
# Validation & I/O
# --------------------------------------------------------------------------- #
def validate(data: HCPData, lookback_months: int = 12, outcome_months: int = 3) -> HCPData:
    """Raise ``ValueError`` with a readable message if the input tables are malformed."""
    problems: list[str] = []
    for name, cols in [("profile", PROFILE_COLS), ("rx_monthly", RX_COLS),
                       ("engagement", ENGAGEMENT_COLS)]:
        missing = set(cols) - set(getattr(data, name).columns)
        if missing:
            problems.append(f"{name}: missing columns {sorted(missing)}")
    if problems:
        raise ValueError("Invalid HCP data: " + "; ".join(problems))

    for name in ("profile", "rx_monthly", "engagement"):
        df = getattr(data, name)
        df["hcp_id"] = df["hcp_id"].astype(str)

    if data.profile["hcp_id"].duplicated().any():
        problems.append("profile: duplicate hcp_id")
    if data.engagement["hcp_id"].duplicated().any():
        problems.append("engagement: duplicate hcp_id")
    if data.rx_monthly.duplicated(["hcp_id", "month"]).any():
        problems.append("rx_monthly: duplicate (hcp_id, month)")
    unknown = set(data.rx_monthly["hcp_id"]) - set(data.profile["hcp_id"])
    if unknown:
        problems.append(f"rx_monthly: {len(unknown)} hcp_id(s) not in profile")
    rx = data.rx_monthly
    if (rx[["market_trx", "brand_trx"]] < 0).any().any():
        problems.append("rx_monthly: negative TRx")
    if (rx["brand_trx"] > rx["market_trx"]).any():
        problems.append("rx_monthly: brand_trx > market_trx (brand must be part of the market)")
    need = set(range(1, lookback_months + outcome_months + 1))
    if not need.issubset(set(rx["month"].unique())):
        problems.append(f"rx_monthly: months {min(need)}..{max(need)} must all be present "
                        f"(lookback={lookback_months}, outcome={outcome_months})")
    cov = data.profile["formulary_coverage"].dropna()
    if ((cov < 0) | (cov > 1)).any():
        problems.append("profile: formulary_coverage must be in [0, 1]")
    if problems:
        raise ValueError("Invalid HCP data: " + "; ".join(problems))
    return data


def save_csv_dir(data: HCPData, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    data.profile.to_csv(path / FILES["profile"], index=False)
    data.rx_monthly.to_csv(path / FILES["rx_monthly"], index=False)
    data.engagement.to_csv(path / FILES["engagement"], index=False)
    if data.truth is not None:
        data.truth.to_csv(path / "synthetic_truth.csv", index=False)
    return path


def load_csv_dir(path: str | Path) -> HCPData:
    path = Path(path)
    tables = {k: pd.read_csv(path / f, dtype={"hcp_id": str}) for k, f in FILES.items()}
    truth_path = path / "synthetic_truth.csv"
    truth = pd.read_csv(truth_path, dtype={"hcp_id": str}) if truth_path.exists() else None
    return HCPData(**tables, truth=truth)


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #
# Per-archetype parameters.
#   vol_*    : log of mean monthly class TRx (lognormal across HCPs)
#   share_*  : logit of brand share at month 1
#   trend_*  : monthly change in brand-share logit
#   resp_mu  : latent "responsiveness to the brand" (hidden; surfaces via engagement)
#   calls    : rep calls / month;  samples: samples per call
#   open     : logit of email open probability;  digital / speaker: events per year
ARCHETYPES: dict[str, dict[str, float]] = {
    "Early adopter": dict(w=0.16, vol_mu=3.0, vol_sd=0.5, share_mu=-3.4, share_sd=0.5,
                          trend_mu=0.08, trend_sd=0.04, resp_mu=0.7, calls=1.5, samples=0.6,
                          open=0.3, digital=6.0, speaker=0.8),
    "Brand loyalist": dict(w=0.14, vol_mu=3.3, vol_sd=0.5, share_mu=-0.4, share_sd=0.6,
                           trend_mu=0.0, trend_sd=0.03, resp_mu=0.4, calls=2.0, samples=0.8,
                           open=0.0, digital=2.0, speaker=1.0),
    "Competitor loyalist": dict(w=0.20, vol_mu=3.2, vol_sd=0.5, share_mu=-4.2, share_sd=0.5,
                                trend_mu=-0.03, trend_sd=0.03, resp_mu=-0.5, calls=0.8,
                                samples=0.3, open=-1.2, digital=0.5, speaker=0.1),
    "Rep-dependent": dict(w=0.22, vol_mu=2.7, vol_sd=0.5, share_mu=-3.4, share_sd=0.5,
                          trend_mu=0.03, trend_sd=0.03, resp_mu=0.2, calls=3.5, samples=1.5,
                          open=-1.8, digital=0.3, speaker=0.2),
    "Low-volume / disengaged": dict(w=0.28, vol_mu=1.8, vol_sd=0.5, share_mu=-3.6, share_sd=0.5,
                                    trend_mu=0.0, trend_sd=0.03, resp_mu=-0.2, calls=0.4,
                                    samples=0.3, open=-0.8, digital=0.8, speaker=0.05),
}
SPECIALTIES = ["Primary Care", "Endocrinology", "Cardiology", "NP/PA"]
SPECIALTY_P = [0.45, 0.15, 0.20, 0.20]
SPECIALTY_VOL = {"Primary Care": 1.2, "Endocrinology": 1.6, "Cardiology": 1.0, "NP/PA": 0.7}
SETTINGS = ["Private practice", "Hospital-affiliated", "Academic", "Community health center"]
SETTING_P = [0.45, 0.30, 0.10, 0.15]
REGIONS = ["Northeast", "South", "Midwest", "West"]
REGION_P = [0.20, 0.35, 0.20, 0.25]
REGION_COVERAGE = {"Northeast": 0.70, "South": 0.55, "Midwest": 0.62, "West": 0.68}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic(n_hcps: int = 4000, seed: int = 42,
                       lookback_months: int = 12, outcome_months: int = 3) -> HCPData:
    """Generate a fully synthetic HCP dataset (no real people, no PHI)."""
    rng = np.random.default_rng(seed)
    n, T = n_hcps, lookback_months + outcome_months
    names = list(ARCHETYPES)
    p = np.array([ARCHETYPES[a]["w"] for a in names])
    arch = rng.choice(len(names), size=n, p=p / p.sum())

    def par(key: str) -> np.ndarray:
        return np.array([ARCHETYPES[a][key] for a in names])[arch]

    ids = np.array([f"HCP{i:06d}" for i in range(1, n + 1)])
    specialty = rng.choice(SPECIALTIES, n, p=SPECIALTY_P)
    setting = rng.choice(SETTINGS, n, p=SETTING_P)
    region = rng.choice(REGIONS, n, p=REGION_P)
    cov_mean = np.array([REGION_COVERAGE[r] for r in region])
    coverage = rng.beta(cov_mean * 25, (1 - cov_mean) * 25)
    years = np.clip(rng.normal(18, 9, n), 1, 45).round()   # deliberately uninformative

    resp = par("resp_mu") + 0.5 * rng.standard_normal(n)    # latent, never exported
    spec_mult = np.array([SPECIALTY_VOL[s] for s in specialty])
    vol = np.exp(par("vol_mu") + par("vol_sd") * rng.standard_normal(n)) * spec_mult

    # --- monthly brand share (logit-linear trend + noise) ---
    academic = (setting == "Academic").astype(float)
    a = par("share_mu") + par("share_sd") * rng.standard_normal(n) + 0.3 * academic
    b = par("trend_mu") + par("trend_sd") * rng.standard_normal(n)
    t = np.arange(T)
    logit = a[:, None] + b[:, None] * t[None, :] + 0.3 * rng.standard_normal((n, T))

    # --- commercial engagement over the lookback window only ---
    calls_rate = par("calls") * np.exp(0.15 * resp) * (vol / vol.mean()) ** 0.25
    rep_calls = rng.poisson(calls_rate * lookback_months)
    recent = rng.binomial(rep_calls, np.clip(rng.normal(0.25, 0.08, n), 0, 1))
    days_since = np.where(recent > 0, rng.integers(1, 90, n),
                          np.where(rep_calls > 0, rng.integers(91, 365, n), 365))
    samples = rng.poisson(rep_calls * par("samples"))
    sent = np.where(rng.random(n) < 0.08, 0, rng.poisson(24, n))
    opened = rng.binomial(sent, _sigmoid(par("open") + 0.5 * resp))
    digital = rng.poisson(par("digital") * np.exp(0.3 * resp))
    speaker = rng.poisson(par("speaker") * np.exp(0.3 * resp))

    # --- outcome-window "adoption push" ---
    # Hidden drivers: latent receptivity, payer access, an access x receptivity
    # interaction, saturating returns on recent rep contact, and plenty of noise.
    cov_c = coverage - 0.6
    jump = (0.9 * resp + 3.0 * cov_c + 4.0 * resp * cov_c
            + 0.5 * np.tanh(recent / 2.0) + 1.3 * rng.standard_normal(n))
    logit[:, lookback_months:] += jump[:, None]
    share = _sigmoid(logit)

    # --- monthly class volume ---
    season = 1 + 0.08 * np.sin(2 * np.pi * t / 12)
    growth = 1 + rng.normal(0.004, 0.01, n)[:, None] * t[None, :]
    lam = vol[:, None] * season[None, :] * growth * rng.gamma(10, 0.1, (n, T))
    market = rng.poisson(lam)
    brand = rng.binomial(market, share)

    rx = pd.DataFrame({
        "hcp_id": np.repeat(ids, T),
        "month": np.tile(np.arange(1, T + 1), n),
        "market_trx": market.ravel(),
        "brand_trx": brand.ravel(),
    })

    engagement = pd.DataFrame({
        "hcp_id": ids, "rep_calls": rep_calls, "rep_calls_recent": recent,
        "days_since_last_call": days_since, "samples_requested": samples,
        "emails_sent": sent, "emails_opened": opened,
        "digital_events": digital, "speaker_programs": speaker,
    })

    # --- realistic missingness ---
    cov_obs = coverage.copy()
    cov_obs[rng.random(n) < 0.03] = np.nan
    years_obs = years.copy()
    years_obs[rng.random(n) < 0.02] = np.nan
    profile = pd.DataFrame({
        "hcp_id": ids, "specialty": specialty, "practice_setting": setting,
        "region": region, "years_in_practice": years_obs,
        "formulary_coverage": cov_obs.round(3),
    })
    truth = pd.DataFrame({"hcp_id": ids, "archetype": np.array(names)[arch]})
    return HCPData(profile, rx, engagement, truth)

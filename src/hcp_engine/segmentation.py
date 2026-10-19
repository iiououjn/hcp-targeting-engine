"""Behavioral segmentation of HCPs.

Pipeline: impute -> log/sqrt transforms for skewed counts -> winsorize -> standardize -> KMeans.

``k`` is chosen automatically by combining internal validity (silhouette) with
*stability* (mean adjusted Rand index between the reference clustering and
clusterings fitted on bootstrap resamples). A k is only eligible if it is stable;
among eligible values the smallest k within 0.01 of the best silhouette wins.
The search starts at k=4 by default: silhouette is biased toward very small k, and
a 3-way split rarely supports differentiated field strategy.

Segments are ordered by size (segment 1 = largest) and given rule-based names derived
from their most distinctive features. Treat the names as a starting point for review
by brand/analytics teams, not as ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEGMENT_FEATURES = [
    "brand_share", "share_trend", "market_trx", "market_cv", "brand_momentum",
    "calls_per_month", "samples_per_call", "email_open_rate",
    "digital_events", "speaker_programs",
]
_LOG1P = ["market_trx", "calls_per_month", "samples_per_call", "digital_events", "speaker_programs"]
_SQRT = ["brand_share"]

# (low phrase, high phrase) used to auto-name segments.
PHRASES: dict[str, tuple[str, str]] = {
    "brand_share": ("Brand-averse", "Brand-loyal"),
    "share_trend": ("Declining share", "Rising share"),
    "market_trx": ("Low-volume", "High-volume"),
    "market_cv": ("Steady prescriber", "Erratic prescriber"),
    "brand_momentum": ("Fading momentum", "Building momentum"),
    "calls_per_month": ("Rarely detailed", "Heavily detailed"),
    "samples_per_call": ("Sample-light", "Sample-heavy"),
    "email_open_rate": ("Email-unresponsive", "Email-responsive"),
    "digital_events": ("Digitally passive", "Digitally engaged"),
    "speaker_programs": ("Few peer events", "Peer-event active"),
}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] training quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class HCPSegmenter:
    def __init__(self, k: int | None = None, k_range: tuple[int, int] = (4, 8),
                 min_stability: float = 0.75, n_boot: int = 8, seed: int = 42,
                 features: list[str] | None = None):
        self.k = k
        self.k_range = k_range
        self.min_stability = min_stability
        self.n_boot = n_boot
        self.seed = seed
        self.features = list(features or SEGMENT_FEATURES)

    # ---------------------------------------------------------------- preprocessing
    def _stateless(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X[self.features].astype(float).copy()
        for c in _LOG1P:
            if c in d:
                d[c] = np.log1p(d[c].clip(lower=0))
        for c in _SQRT:
            if c in d:
                d[c] = np.sqrt(d[c].clip(lower=0))
        return d

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return the standardized feature space the clustering lives in."""
        return self.prep_.transform(self._stateless(X))

    # ------------------------------------------------------------------ k selection
    def select_k(self, Z: np.ndarray) -> pd.DataFrame:
        n = len(Z)
        rng = np.random.default_rng(self.seed)
        rows = []
        for k in range(self.k_range[0], self.k_range[1] + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(Z)
            lab = km.labels_
            aris = []
            for b in range(self.n_boot):
                idx = rng.integers(0, n, n)
                kb = KMeans(n_clusters=k, n_init=3, random_state=self.seed + b + 1).fit(Z[idx])
                aris.append(adjusted_rand_score(lab, kb.predict(Z)))
            rows.append({
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(Z, lab, sample_size=min(n, 3000),
                                                     random_state=self.seed)),
                "calinski_harabasz": float(calinski_harabasz_score(Z, lab)),
                "davies_bouldin": float(davies_bouldin_score(Z, lab)),
                "stability_ari": float(np.mean(aris)),
            })
        return pd.DataFrame(rows)

    def _choose_k(self, sel: pd.DataFrame, tol: float = 0.01) -> int:
        """Smallest stable k whose silhouette is within ``tol`` of the best stable k."""
        ok = sel[sel["stability_ari"] >= self.min_stability]
        if ok.empty:  # nothing is stable enough: fall back to the most stable k
            return int(sel.loc[sel["stability_ari"].idxmax(), "k"])
        near_best = ok[ok["silhouette"] >= ok["silhouette"].max() - tol]
        return int(near_best["k"].min())

    # -------------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame) -> HCPSegmenter:
        base = self._stateless(X)
        self.prep_ = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer()),
            ("scale", StandardScaler()),
        ]).fit(base)
        Z = self.prep_.transform(base)

        self.selection_: pd.DataFrame | None = None
        k = self.k
        if k is None:
            self.selection_ = self.select_k(Z)
            k = self._choose_k(self.selection_)
        self.k_ = int(k)

        km = KMeans(n_clusters=self.k_, n_init=20, random_state=self.seed).fit(Z)
        order = np.argsort(-np.bincount(km.labels_, minlength=self.k_), kind="stable")
        self._remap = np.empty(self.k_, dtype=int)
        self._remap[order] = np.arange(self.k_)
        self.kmeans_ = km
        self.labels_ = pd.Series(self._remap[km.labels_] + 1, index=X.index, name="segment_id")

        self.z_profile_ = pd.DataFrame(km.cluster_centers_[order], columns=self.features,
                                       index=pd.Index(np.arange(1, self.k_ + 1), name="segment_id"))
        raw_means = X[self.features].groupby(self.labels_).mean()
        sizes = self.labels_.value_counts().sort_index()
        self.profile_ = pd.concat(
            [pd.DataFrame({"n_hcps": sizes, "pct_hcps": sizes / len(X)}), raw_means], axis=1)
        self.names_ = self._name_segments()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        lab = self._remap[self.kmeans_.predict(self.transform(X))] + 1
        return pd.Series(lab, index=X.index, name="segment_id")

    # ------------------------------------------------------------------------ naming
    def _name_segments(self, min_abs_z: float = 0.4, n_terms: int = 2) -> dict[int, str]:
        names: dict[int, str] = {}
        for seg, z in self.z_profile_.iterrows():
            top = z.reindex(z.abs().sort_values(ascending=False).index)
            terms = [PHRASES[f][int(v > 0)] for f, v in top.items()
                     if abs(v) >= min_abs_z and f in PHRASES][:n_terms]
            names[int(seg)] = f"S{seg}: " + (", ".join(terms) if terms else "Average profile")
        return names

    def name_of(self, seg_ids: pd.Series) -> pd.Series:
        return seg_ids.map(self.names_).rename("segment")

"""Matplotlib figures. All functions save a PNG and close the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from hcp_engine.propensity import calibration_table, gains_table  # noqa: E402

PALETTE = ["#2E6F95", "#E07A5F", "#3D9970", "#8E5BA8", "#C9A227", "#5C6B73", "#B5446E", "#1B998B"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_k_selection(sel: pd.DataFrame, chosen_k: int, path: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, col, title in zip(ax, ["inertia", "silhouette", "stability_ari"],
                             ["Inertia (elbow)", "Silhouette (higher = better)",
                              "Bootstrap stability, ARI"], strict=True):
        a.plot(sel["k"], sel[col], "o-", color=PALETTE[0])
        a.axvline(chosen_k, color=PALETTE[1], ls="--", label=f"chosen k={chosen_k}")
        a.set_title(title)
        a.set_xlabel("k")
    ax[2].axhline(0.75, color="grey", ls=":", lw=1)
    ax[0].legend()
    return _save(fig, path)


def plot_segment_profiles(z: pd.DataFrame, names: dict[int, str], sizes: pd.Series,
                          path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(1.1 * len(z.columns) + 3, 0.7 * len(z) + 2.2))
    lim = max(1.5, float(np.abs(z.to_numpy()).max()))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(z.columns)), z.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(z)), [f"{names[i]}  (n={sizes[i]:,})" for i in z.index])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.iat[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="z-score vs all HCPs", shrink=0.8)
    ax.set_title("Segment behavioral profiles")
    return _save(fig, path)


def plot_segment_map(Z: np.ndarray, labels: pd.Series, names: dict[int, str], path: Path,
                     seed: int = 42, max_points: int = 4000) -> Path:
    xy = PCA(n_components=2, random_state=seed).fit_transform(Z)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(len(Z), max_points), replace=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, seg in enumerate(sorted(labels.unique())):
        m = labels.to_numpy()[idx] == seg
        ax.scatter(xy[idx][m, 0], xy[idx][m, 1], s=7, alpha=0.55,
                   color=PALETTE[i % len(PALETTE)], label=names[seg])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("HCP segments in PCA space")
    ax.legend(markerscale=2.5, fontsize=8, frameon=False, loc="best")
    return _save(fig, path)


def plot_calibration(y, p, path: Path) -> Path:
    cal = calibration_table(y, p)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(cal["mean_predicted"], cal["observed"], "o-", color=PALETTE[0], label="model (OOF)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed adoption rate")
    ax.set_title("Calibration")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_gains(y, p, path: Path) -> Path:
    g = gains_table(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(g["decile"], g["lift"], color=PALETTE[0])
    ax[0].axhline(1, color="grey", ls="--")
    ax[0].set_xlabel("Score decile (1 = highest)")
    ax[0].set_ylabel("Lift")
    ax[0].set_title("Lift by decile")
    x = np.r_[0, g["decile"] / 10]
    ax[1].plot(x, np.r_[0, g["cum_capture"]], "o-", color=PALETTE[1], label="model")
    ax[1].plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax[1].set_xlabel("Share of prospects contacted")
    ax[1].set_ylabel("Share of adopters captured")
    ax[1].set_title("Cumulative gains")
    ax[1].legend(frameon=False)
    return _save(fig, path)


def plot_importance_bar(imp: pd.Series, path: Path, top: int = 15) -> Path:
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(s) + 1.3))
    ax.barh(s.index, s.to_numpy(), color=PALETTE[0])
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("What drives propensity (global)")
    return _save(fig, path)


def plot_shap_beeswarm(shap_values: pd.DataFrame, X_model: pd.DataFrame, path: Path,
                       max_display: int = 15) -> Path:
    import shap  # lazy: optional at import time

    plt.figure()
    shap.summary_plot(shap_values.to_numpy(), X_model, feature_names=list(X_model.columns),
                      max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 0.38 * max_display + 1.8)
    return _save(fig, path)


def plot_segment_opportunity(opp: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    lab = opp["segment_id"].map(lambda i: f"S{i}")
    ax[0].bar(lab, opp["mean_propensity"].fillna(0), color=PALETTE[0])
    ax[0].set_title("Mean propensity (prospects)")
    ax[0].set_ylabel("Probability")
    ax[1].bar(lab, opp["total_expected_value"], color=PALETTE[1])
    ax[1].set_title("Total expected value (propensity x addressable TRx)")
    for a in ax:
        a.set_xlabel("Segment")
    return _save(fig, path)

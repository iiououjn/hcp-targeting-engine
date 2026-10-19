"""End-to-end orchestration: data -> features -> segments -> propensity -> targets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from hcp_engine.config import Config
from hcp_engine.data import HCPData, generate_synthetic, validate
from hcp_engine.features import build_features, build_labels
from hcp_engine.propensity import CVResult, PropensityModel, gains_table
from hcp_engine.segmentation import HCPSegmenter
from hcp_engine.targeting import build_target_table, segment_opportunity

log = logging.getLogger("hcp_engine")


@dataclass
class PipelineResult:
    config: Config
    features: pd.DataFrame
    labels: pd.DataFrame
    segmenter: HCPSegmenter
    propensity_model: PropensityModel
    cv: CVResult
    target_table: pd.DataFrame
    segment_opportunity: pd.DataFrame
    metrics: dict
    importance: pd.Series | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)


def run_pipeline(cfg: Config | None = None, data: HCPData | None = None,
                 out_dir: str | Path | None = "outputs", n_hcps: int = 4000,
                 make_plots: bool = True) -> PipelineResult:
    """Run the full engine. Pass ``out_dir=None`` to skip writing artifacts."""
    cfg = cfg or Config()
    if data is None:
        log.info("Generating synthetic data (n=%d, seed=%d)", n_hcps, cfg.seed)
        data = generate_synthetic(n_hcps, cfg.seed, cfg.lookback_months, cfg.outcome_months)
    validate(data, cfg.lookback_months, cfg.outcome_months)

    # 1) Features (lookback only) and labels (outcome window only)
    feats = build_features(data, cfg.lookback_months)
    labels = build_labels(data, cfg.lookback_months, cfg.outcome_months, cfg.adoption_trx)

    # 2) Behavioral segmentation on ALL HCPs
    seg = HCPSegmenter(k=cfg.k, k_range=(cfg.k_min, cfg.k_max),
                       min_stability=cfg.min_stability, seed=cfg.seed).fit(feats)
    seg_names = seg.name_of(seg.labels_)
    log.info("Segmentation: k=%d", seg.k_)

    # 3) Propensity model on the prospect universe (non-established writers)
    prospect_mask = feats["brand_trx_3m"] < cfg.established_writer_trx
    Xp = feats[prospect_mask]
    yp = labels.loc[Xp.index, "adopted"]
    if yp.nunique() < 2 or yp.sum() < 30:
        raise ValueError(
            f"Too few positive labels to train ({int(yp.sum())} of {len(yp)}). Revisit "
            "`adoption_trx`, `established_writer_trx` or the outcome window.")
    model = PropensityModel(cfg.exclude_features, cfg.seed)
    cv = model.cross_validate(Xp, yp, cfg.cv_folds)
    model.fit(Xp, yp)
    oof = pd.Series(cv.oof, index=Xp.index, name="propensity")   # honest scores for the universe
    log.info("Propensity CV AUC=%.3f (baseline LR %.3f)", cv.metrics["roc_auc"],
             cv.baseline_metrics["roc_auc"])

    # 4) Explanations (SHAP)
    reasons = barriers = importance = None
    explanation = None
    if cfg.explain:
        from hcp_engine.explain import PropensityExplainer, top_contributors
        explanation = PropensityExplainer(model).explain(Xp)
        reasons = top_contributors(explanation.grouped, Xp, cfg.top_reasons, positive=True)
        barriers = top_contributors(explanation.grouped, Xp, cfg.top_reasons, positive=False)
        importance = explanation.global_importance()

    # 5) Targeting
    table = build_target_table(feats, seg.labels_, seg_names, oof, cfg, reasons, barriers)
    opp = segment_opportunity(table)

    # 6) Metrics
    sizes = seg.labels_.value_counts().sort_index()
    metrics: dict = {
        "config": cfg.to_dict(),
        "n_hcps": int(len(feats)),
        "n_prospects": int(prospect_mask.sum()),
        "n_established_writers": int((~prospect_mask).sum()),
        "segmentation": {
            "k": seg.k_,
            "segment_sizes": {int(k): int(v) for k, v in sizes.items()},
            "segment_names": {int(k): v for k, v in seg.names_.items()},
        },
        "propensity_cv": {
            "gbm": cv.metrics,
            "logistic_baseline": cv.baseline_metrics,
            "fold_roc_auc": cv.fold_auc,
        },
    }
    if seg.selection_ is not None:
        metrics["segmentation"]["k_selection"] = seg.selection_.round(4).to_dict("records")
    if data.truth is not None:  # synthetic only: how well were latent archetypes recovered?
        tr = data.truth.assign(hcp_id=data.truth["hcp_id"].astype(str)).set_index("hcp_id")
        metrics["segmentation"]["ari_vs_synthetic_truth"] = float(
            adjusted_rand_score(tr.loc[feats.index, "archetype"], seg.labels_))

    result = PipelineResult(cfg, feats, labels, seg, model, cv, table, opp, metrics, importance)

    # 7) Persist
    if out_dir is not None:
        result.artifacts = _write_outputs(result, Path(out_dir), yp, explanation, make_plots)
    return result


def _write_outputs(r: PipelineResult, out: Path, y_prospect: pd.Series, explanation,
                   make_plots: bool) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    a: dict[str, Path] = {}

    a["targets"] = out / "hcp_targeting_table.csv"
    r.target_table.round(4).to_csv(a["targets"], index=False)
    a["segment_profiles"] = out / "segment_profiles.csv"
    prof = r.segmenter.profile_.copy()
    prof.insert(0, "segment", pd.Series(r.segmenter.names_))
    prof.round(4).to_csv(a["segment_profiles"])
    a["segment_opportunity"] = out / "segment_opportunity.csv"
    r.segment_opportunity.round(4).to_csv(a["segment_opportunity"], index=False)
    a["metrics"] = out / "metrics.json"
    a["metrics"].write_text(json.dumps(r.metrics, indent=2, default=float))
    if r.importance is not None:
        a["importance"] = out / "shap_global_importance.csv"
        r.importance.round(5).to_csv(a["importance"])
    a["segmenter_model"] = out / "models" / "segmenter.joblib"
    joblib.dump(r.segmenter, a["segmenter_model"])
    a["propensity_model"] = out / "models" / "propensity.joblib"
    joblib.dump(r.propensity_model, a["propensity_model"])
    a["lift_table"] = out / "lift_table.csv"
    gains_table(y_prospect, r.cv.oof).round(4).to_csv(a["lift_table"], index=False)

    if make_plots:
        from hcp_engine import viz
        fig = out / "figures"
        seg = r.segmenter
        sizes = seg.profile_["n_hcps"]
        if seg.selection_ is not None:
            a["fig_k"] = viz.plot_k_selection(seg.selection_, seg.k_, fig / "k_selection.png")
        a["fig_profiles"] = viz.plot_segment_profiles(seg.z_profile_, seg.names_, sizes,
                                                      fig / "segment_profiles.png")
        a["fig_map"] = viz.plot_segment_map(seg.transform(r.features), seg.labels_, seg.names_,
                                            fig / "segment_map.png", seed=r.config.seed)
        a["fig_calibration"] = viz.plot_calibration(y_prospect, r.cv.oof,
                                                    fig / "calibration.png")
        a["fig_gains"] = viz.plot_gains(y_prospect, r.cv.oof, fig / "lift_gains.png")
        a["fig_opportunity"] = viz.plot_segment_opportunity(r.segment_opportunity,
                                                            fig / "segment_opportunity.png")
        if explanation is not None:
            a["fig_importance"] = viz.plot_importance_bar(r.importance, fig / "shap_importance.png")
            a["fig_beeswarm"] = viz.plot_shap_beeswarm(explanation.shap_values,
                                                       explanation.X_model,
                                                       fig / "shap_beeswarm.png")
    return a

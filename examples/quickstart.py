"""Minimal programmatic use of the engine.

    python examples/quickstart.py
"""

from hcp_engine import Config, run_pipeline

cfg = Config(call_capacity=250, explain=True)   # set explain=False to skip SHAP
res = run_pipeline(cfg, out_dir="outputs", n_hcps=4000)

print(res.metrics["propensity_cv"]["gbm"])
print(res.segment_opportunity)

top = res.target_table.query("in_call_plan").head(10)
print(top[["hcp_id", "segment", "propensity", "tier", "channel_hint", "why_target"]])

# Score brand-new HCPs later (features must be built the same way):
#   new_scores = res.propensity_model.predict_proba(new_features)
#   new_segments = res.segmenter.predict(new_features)

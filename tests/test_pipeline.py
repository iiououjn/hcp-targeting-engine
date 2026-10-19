import importlib.util
import json
import tempfile
from pathlib import Path

from helpers import data
from hcp_engine.cli import main
from hcp_engine.config import Config
from hcp_engine.pipeline import run_pipeline

HAS_SHAP = importlib.util.find_spec("shap") is not None


def test_end_to_end_writes_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        res = run_pipeline(Config(seed=7, explain=HAS_SHAP, call_capacity=50),
                           data=data(), out_dir=tmp)
        out = Path(tmp)
        for f in ["hcp_targeting_table.csv", "segment_profiles.csv", "segment_opportunity.csv",
                  "metrics.json", "lift_table.csv", "models/segmenter.joblib",
                  "models/propensity.joblib", "figures/segment_profiles.png",
                  "figures/lift_gains.png"]:
            assert (out / f).exists(), f
        m = json.loads((out / "metrics.json").read_text())
        assert m["propensity_cv"]["gbm"]["roc_auc"] > 0.7
        assert m["segmentation"]["ari_vs_synthetic_truth"] > 0.5
        assert m["n_hcps"] == len(res.target_table)
        if HAS_SHAP:
            assert res.target_table["why_target"].str.len().gt(0).any()


def test_cli_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        code = main(["run", "--n-hcps", "800", "--out", tmp, "--no-explain", "--no-plots"])
        assert code == 0
        assert (Path(tmp) / "hcp_targeting_table.csv").exists()

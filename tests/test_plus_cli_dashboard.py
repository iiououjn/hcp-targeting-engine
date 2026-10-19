from __future__ import annotations

import json

import pandas as pd

from helpers import data, prospect_xy
from hcp_engine.data import save_csv_dir
from hcp_engine.propensity import PropensityModel
from hcp_engine_plus.cli import main
from hcp_engine_plus.dashboard import write_dashboard


def test_dashboard_has_escaped_html_and_search(tmp_path):
    table = pd.DataFrame({"hcp_id": ["<script>alert(1)</script>", "two"],
                          "propensity": [0.5, 0.7], "in_call_plan": [True, False]})
    path = write_dashboard(table, tmp_path / "report.html")
    content = path.read_text()
    assert "&lt;script&gt;" in content
    assert "<script>alert(1)</script>" not in content
    assert "search.addEventListener" in content


def test_cli_audit_score_and_dashboard(tmp_path):
    input_dir = save_csv_dir(data(), tmp_path / "data")
    assert main(["audit-data", "--input-dir", str(input_dir), "--out", str(tmp_path / "audit.json")]) == 0
    assert json.loads((tmp_path / "audit.json").read_text())["duplicate_profile_ids"] == 0
    X, y = prospect_xy()
    model = PropensityModel(seed=3, gbm_params={"n_estimators": 30}).fit(X, y)
    import joblib
    joblib.dump(model, tmp_path / "model.joblib")
    assert main(["score-asof", "--input-dir", str(input_dir), "--model", str(tmp_path / "model.joblib"),
                 "--as-of", "12", "--out", str(tmp_path / "scores.csv")]) == 0
    assert (tmp_path / "scores.csv").exists()
    assert main(["dashboard", "--targets", str(tmp_path / "scores.csv"),
                 "--out", str(tmp_path / "dashboard.html")]) == 0
    assert (tmp_path / "dashboard.html").is_file()


def test_cli_temporal_holdout_and_original_generate_path(tmp_path):
    from hcp_engine.cli import main as original_main
    from hcp_engine.data import HCPData
    source = data()
    earlier = save_csv_dir(source, tmp_path / "train")
    later = HCPData(source.profile, source.rx_monthly.assign(month=lambda x: x.month + 4),
                    source.engagement)
    later_path = save_csv_dir(later, tmp_path / "later")
    assert main(["temporal-holdout", "--train-dir", str(earlier), "--train-as-of", "12",
                 "--test-dir", str(later_path), "--test-as-of", "16",
                 "--out", str(tmp_path / "holdout.json")]) == 0
    assert "roc_auc" in json.loads((tmp_path / "holdout.json").read_text())["metrics"]
    assert (tmp_path / "holdout.csv").exists()
    assert original_main(["generate-data", "--n-hcps", "10", "--out", str(tmp_path / "generated")]) == 0
    assert (tmp_path / "generated" / "hcp_profile.csv").exists()

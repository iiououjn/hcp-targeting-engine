# Build and Verification Report

**Scope:** This report distinguishes local execution from unverified CI/external activities. The project remains a synthetic-data prototype.

## Locally verified (19 October 2026)

- Python 3.13.5, pytest 9.0.2, SHAP 0.50.0, pandas 2.2.3, scikit-learn 1.8.0, NumPy 2.3.5, matplotlib 3.10.8, joblib 1.5.3.
- `pip install -e . --no-deps --no-build-isolation` succeeded for this already-provisioned environment; the CLI entry points `hcp-engine` and `hcp-engine-plus` are exposed.
- `pytest -q`: **46 passed**. The original 32 tests, including the actual `test_shap_values_are_additive`, executed successfully. Fourteen new extension tests passed.
- `python -m compileall -q src tests` succeeded.
- Ran the original full pipeline with *real SHAP enabled* on 800 synthetic HCPs: `hcp-engine run --n-hcps 800 --k 5 --capacity 90 --out OUTPUT_DIR`. The observed synthetic-only cross-validated ROC-AUC was 0.857; the corresponding synthetic logistic baseline AUC was 0.836. No interpretation as real-world performance is warranted.
- Generated SHAP importance and SHAP beeswarm PNGs with that real execution. Copied them to `docs/images/`.
- Ran `hcp-engine generate-data` to create 800 synthetic HCPs; ran `hcp-engine-plus audit-data` (0 missing/duplicate panel rows), `score-asof` (produced scores), `audit-scores` and `dashboard` (generated an HTML file).
- Exercised `temporal-holdout` by test code on synthetic fixtures with separated as-of windows; tests reject overlapping windows. This is **not** a longitudinal real-data validation.
- Compared SHA-256 digests of all 11 original `src/hcp_engine/` Python modules against the uploaded ZIP: **11/11 unchanged**. The existing CLI and core behavior were not modified.

## Not locally verified / blocked

- `ruff check` was not executed because Ruff is absent and package installation could not reach the dependency index. A full `ruff` job remains in the original CI workflow. Manually checked Python source line widths, but that is not equivalent to Ruff.
- `mypy` was not executed for the same reason; an optional manual GitHub Actions typecheck workflow is included.
- The oldest supported versions (pandas 2.0.3, scikit-learn 1.3.2, etc.) were not installed locally; the new compatibility GitHub Actions matrix is configured but **has not run**. Python 3.10–3.12 are likewise untested here.
- Docker image build was not performed. Dockerfile and documented commands are provided for a standard Docker-enabled host.
- No production HCP panel, dated engagement records, competitor/brand-specific Rx, outreach treatment assignment, or independent evaluation labels were supplied. Therefore temporal generalization on real data, calibration generalization, demographic fairness and true uplift remain unverified.
- No authenticated GitHub repository or deployment was modified; publishing and CI checks require your own GitHub connection.

## Before using outside a demo

1. Publish to GitHub and check both workflows, fixing any Ruff, mypy, Python-version or dependency-resolution failures.
2. Inspect and explicitly validate quality of your own licensed point-in-time HCP data.
3. Assemble non-overlapping early/late periods, train/test forward, and reserve a further independent calibration/evaluation window.
4. Conduct human privacy, pharmaceutical promotional-compliance, and formal fairness reviews. The region/specialty descriptive audit does not satisfy the last requirement.
5. Validate call-plan constraints with real rep/territory master data, contact restrictions and CRM reconciliation.

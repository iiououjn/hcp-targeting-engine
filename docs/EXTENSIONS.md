# Optional extensions — source and safety guide

## 1. How the time axis works

`hcp_engine` was originally designed for one 12-month lookback and one 3-month outcome period, with `month` values 1 through 15. `hcp_engine_plus.temporal` can instead consume a **chronologically consistent, multi-period** prescribing panel indexed by consecutive integers (`1, 2, ..., N`), preserving the source schema. The integer is an ordinal calendar month, not the `YYYYMM` integer format.

- `asof_snapshot(source, as_of_month=12, lookback_months=12)` reads Rx months **1..12** only.
- With `outcome_months=3`, it *also* extracts months 13..15 for supervised labels, but those months are **never supplied to `build_features`**.
- `temporal_holdout(train_data, test_data, 12, 16)` trains on labels from months 13..15 and evaluates on labels from months 17..19. Earlier labels must end strictly **before** the later feature cutoff. The 2 snapshots should refer to the same consistent upstream time axis.
- Two genuine `engagement.csv` files must be built with aggregates based only on activity **up to** their respective as-of months; the source schema has no event-level log from which to reconstruct history. Profile attributes must likewise reflect what was known at the cutoff.
- Both snapshots can contain overlapping HCP IDs. That is intentional for forward deployment evaluation, but two independent segmentation fits cannot be compared by segment ID.
- By default every HCP-month cell in the required period must be present. Treat absent records as **unknown** unless your organization's data contract explicitly says absence equals zero. If zero is verified, write explicit zero rows. Do not use the non-strict mode to mask genuine coverage gaps.

Use `python -m hcp_engine_plus.cli audit-data --input-dir DIR --out quality.json` first. The audit reports missing HCP-month cells, missing engagement IDs and duplicates without silently fixing or deduplicating anything.

## 2. Scoring later records

1. Train with the original `hcp-engine run` and save its `models/propensity.joblib` file. Never load a model downloaded from an untrusted source; joblib uses pickle.
2. Assemble `hcp_profile.csv`, `rx_monthly.csv`, `engagement.csv` representing a later as-of date, with ordinal month values. The later outcomes need not be present for `score-asof`.
3. Run:

```bash
python -m hcp_engine_plus.cli score-asof --input-dir snapshots/latest \
  --model outputs/models/propensity.joblib --as-of 16 --out outputs/new_scores.csv
```

Only prospects receive a probability; established writers are flagged separately. Model calibration must be applied explicitly if fit from a distinct validation period.

## 3. Forward holdout

```bash
python -m hcp_engine_plus.cli temporal-holdout \
  --train-dir snapshots/earlier --train-as-of 12 \
  --test-dir snapshots/later --test-as-of 16 \
  --out outputs/temporal.json
```

The JSON stores train/test counts and forward AUC, PR-AUC, Brier and top-decile lift. The paired `.csv` contains individual test scores and labels; protect it as controlled data. If classes are absent, the evaluator refuses to report misleading AUC. The time-based procedure is real code, but a *credible* performance claim requires properly prepared chronological data independent of the model author.

## 4. Calibration, tuning, audit and constraints (Python API)

```python
from hcp_engine_plus.decisions import ProbabilityCalibrator, constrained_call_plan
from hcp_engine_plus.audits import group_score_audit, population_stability_index
from hcp_engine_plus.tuning import tune_on_training

# Period A: train classifier. Period B: calibrate. Period C: final evaluation.
cal = ProbabilityCalibrator('sigmoid').fit(period_b_raw_probabilities, period_b_true_labels)
period_c_calibrated = cal.predict(period_c_raw_probabilities)

# Optional: tune ONLY on early training labels. Never use later holdout outcomes.
search = tune_on_training(train_features, train_labels, folds=3)

# Explicit territory, representative, budget and cooldown policies.
plan = constrained_call_plan(
    candidates, max_calls=100, territory_capacity={'north': 35, 'south': 65},
    rep_capacity={'rep_1': 20, 'rep_2': 25}, max_cost=8000,
    min_days_between_calls=30,
)
report = group_score_audit(candidates, 'region')
shift = population_stability_index(previous_scores, current_scores)
```

The `constrained_call_plan` API requires `days_since_last_call` so a cooldown can be enforced; if you request a budget, you must provide `cost_per_call`; if requesting capacities, provide `territory` and/or `rep_id`. Missing constraint columns cause errors instead of silent policy bypasses. All capacity limits use a greedy sort, **not** an optimization algorithm.

The `group_score_audit` API reports descriptive score differences by a supplied group; this alone is NOT demographic fairness validation, and there are no protected group attributes in the bundled schema. Handle any external protected attributes according to applicable policy and law.

`segment_migration(previous, current)` assumes both tables use the *same fitted segmenter*. Call `segmenter.predict(features)` on refreshed data rather than re-fitting and comparing arbitrary label numbers.

## 5. What remains unsupported

- Uplift / incremental response to sales calls needs randomized treatment/control and potential-outcome design. Propensity and SHAP cannot replace causal identification.
- Multi-brand modeling needs distinct brand/competitor Rx columns, identity resolution and product-specific labels; do not fake this by relabeling a single-brand score.
- Real-time inventory or sales CRM integrations have not been added. Do not assume data vendor feeds or consent exist.
- CI includes a Python 3.10 oldest-dependency job, but this was not locally executed without package access.

### Explicit expected history bounds for data audit

By default the data audit checks the continuous span between the smallest and largest Rx month values. To catch missing calendar months *at either end*, supply bounds:

```bash
hcp-engine-plus audit-data --input-dir snapshots/earlier \
  --expected-start 1 --expected-end 15 --out outputs/quality.json
```

## 6. Development checks

```bash
pip install -e '.[dev,types]'
pytest -q
ruff check src tests
mypy src/hcp_engine_plus
```

The mypy workflow is manually triggerable in GitHub Actions; the main CI continues to run pytest/ruff. Ruff, mypy and the oldest dependency matrix are provided for external verification but were not executable in the offline build environment. A failure in one of those jobs should be fixed before describing the project as fully lint/type/version verified.

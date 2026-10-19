# Input data schema

Run on your own data with `hcp-engine run --input-dir path/to/dir`. The directory must contain
three CSVs. Column names are fixed; extra columns are ignored. `hcp_id` is read as a string.

All activity columns must describe **only the lookback window** (months `1..lookback_months`).
Anything measured after that window leaks the outcome into the features.

## `hcp_profile.csv` (one row per HCP)

| column | type | notes |
|---|---|---|
| `hcp_id` | str | unique key |
| `specialty` | str | categorical |
| `practice_setting` | str | categorical |
| `region` | str | categorical |
| `years_in_practice` | number | may be blank. A proxy for age: consider `exclude_features` |
| `formulary_coverage` | 0-1 | share of the HCP's patient lives with favorable coverage; may be blank |

## `rx_monthly.csv` (one row per HCP-month)

| column | type | notes |
|---|---|---|
| `hcp_id` | str | must exist in the profile |
| `month` | int | `1..lookback+outcome` (default 1..15); month 1 is the oldest |
| `market_trx` | int | total class prescriptions that month |
| `brand_trx` | int | your brand's prescriptions; must be `<= market_trx` |

Months `1..12` become features. Months `13..15` define the label and are never visible to features.
Every HCP-month should be present (write `0`, not a missing row).

## `engagement.csv` (one row per HCP, lookback window)

| column | meaning |
|---|---|
| `rep_calls` | rep calls over the lookback window |
| `rep_calls_recent` | rep calls in the last 3 lookback months |
| `days_since_last_call` | days from last call to the end of lookback (365 if never) |
| `samples_requested` | sample requests |
| `emails_sent`, `emails_opened` | approved-email sends and opens |
| `digital_events` | webinars / portal sessions / virtual events |
| `speaker_programs` | peer programs attended |

## Label

`adopted = 1` if a **prospect** (fewer than `established_writer_trx` brand TRx in the last 3
lookback months) writes at least `adoption_trx` brand TRx during the outcome window.
Both thresholds are in `Config`.

## Outputs

| file | contents |
|---|---|
| `hcp_targeting_table.csv` | one row per HCP: segment, OOF propensity, expected value, tier, recommended play, channel hint, SHAP "why target" / "watch-outs" |
| `segment_profiles.csv` | segment names, sizes, raw-feature means |
| `segment_opportunity.csv` | per-segment prospects, mean propensity, expected value, call-plan share |
| `shap_global_importance.csv` | mean abs SHAP per feature (needs SHAP) |
| `metrics.json` | config, k selection, CV metrics (GBM and logistic baseline) |
| `lift_table.csv` | decile gains table |
| `models/*.joblib` | fitted segmenter and propensity model |
| `figures/*.png` | diagnostics |

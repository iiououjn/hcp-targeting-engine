# Model Card — HCP Propensity & Segmentation (demonstration)

## Intended use

**Prototype / educational** ranking and analysis of HCP behavior using appropriately governed prescribing and engagement aggregates. Outputs are decision-support artifacts, not automated outreach instructions.

## Data

The bundled example uses *entirely synthetic* HCP records. No personal health information or commercial data is shipped. The original single-brand schema includes an HCP profile, HCP-month market/brand TRx, and one engagement aggregate table. It has **no per-month engagement events** or treatment-assignment fields.

## Model

- Scikit-learn gradient boosting classifier trained on prospects. Random out-of-fold CV estimates performance on synthetic records from one quarter.
- Logistic regression baseline. SHAP log-odds contributions explain the booster prediction, not causal effects.
- Segmenter uses clustering and names segments using behavioral distinctions.

## Evaluation status

- `pytest` including real SHAP additivity was executed locally; see `VERIFICATION.md`.
- Illustrative plots in `docs/images` come from a synthetic demonstration and are not real-world performance evidence.
- The new temporal evaluator expects **separately prepared and chronologically consistent** earlier/later snapshots; it does not conjure valid time series from one quarter.
- Before deployments, evaluate on a representative external cohort, report uncertainty, temporal outcomes, calibration, missingness and changes in access/coverage.

## Risks

Potential demographic/protected-attribute proxies, misleading probabilities under distribution shift, prescriber profiling, inappropriate pharmaceutical promotion and data licensing restrictions. An audit by geography/specialty is NOT a protected-class fairness audit. Segment migration requires a common fitted segmenter; labels from independent refits aren't comparable.

## Not supported / nonclaims

No causal uplift estimate (requires randomized outreach/no-outreach data), no optimized multi-brand/competitor modeling, no clinical outcome or patient-level decision support, no compliance certification. Constrained scheduling is greedy, not globally optimal. The product should not be used to make important decisions without appropriate human review.

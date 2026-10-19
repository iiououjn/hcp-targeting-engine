"""Added features complement the original 32 regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helpers import data, prospect_xy
from hcp_engine.data import HCPData
from hcp_engine.propensity import PropensityModel
from hcp_engine_plus.audits import (group_score_audit, population_stability_index,
                                    quality_report, segment_migration)
from hcp_engine_plus.temporal import asof_snapshot, score_asof, temporal_holdout


def test_asof_snapshot_does_not_read_future_outcomes():
    source = data()
    snap = asof_snapshot(source, 12)
    assert snap.rx_monthly["month"].max() == 12
    assert len(snap.rx_monthly) == len(source.profile) * 12
    # Changing ANY future outcome rows cannot change as-of features or scores.
    mutated = HCPData(source.profile, source.rx_monthly.copy(), source.engagement)
    mutated.rx_monthly.loc[mutated.rx_monthly.month > 12, "brand_trx"] = 0
    assert snap.rx_monthly.equals(asof_snapshot(mutated, 12).rx_monthly)


def test_asof_score_model_only_scores_prospects():
    X, y = prospect_xy()
    model = PropensityModel(seed=2, gbm_params={"n_estimators": 45}).fit(X, y)
    result = score_asof(model, data(), 12)
    assert len(result) == len(data().profile)
    assert result.loc[result.status == "Established writer", "propensity"].isna().all()
    assert result.loc[result.status == "Prospect", "propensity"].between(0, 1).all()


def test_asof_rejects_missing_month_and_duplicate_rows():
    source = data()
    no_month = HCPData(source.profile, source.rx_monthly[source.rx_monthly.month != 6], source.engagement)
    with pytest.raises(ValueError, match="Missing whole calendar"):
        asof_snapshot(no_month, 12)
    no_row = HCPData(source.profile, source.rx_monthly.drop(source.rx_monthly.index[0]), source.engagement)
    with pytest.raises(ValueError, match="Incomplete HCP-month"):
        asof_snapshot(no_row, 12)
    duplicate = HCPData(source.profile, pd.concat([source.rx_monthly, source.rx_monthly.head(1)]),
                        source.engagement)
    with pytest.raises(ValueError, match="Duplicate HCP-month"):
        asof_snapshot(duplicate, 12)


def test_temporal_holdout_requires_forward_nonoverlap():
    source = data()
    later = HCPData(source.profile, source.rx_monthly.assign(month=lambda x: x.month + 4),
                    source.engagement)
    with pytest.raises(ValueError, match="AFTER"):
        temporal_holdout(source, later, train_as_of=12, test_as_of=15)
    result = temporal_holdout(source, later, train_as_of=12, test_as_of=16)
    assert result.train_as_of == 12 and result.test_as_of == 16
    assert result.n_train > 0 and result.n_test > 0
    assert 0 <= result.metrics["roc_auc"] <= 1
    assert result.predictions["as_of_month"].eq(16).all()


def test_quality_audit_catches_missing_cells_and_id_mismatch():
    source = data()
    mutated = HCPData(source.profile, source.rx_monthly.iloc[1:],
                      source.engagement.iloc[1:])
    report = quality_report(mutated, range(1, 16))
    assert report["missing_hcp_month_cells"] == 1
    assert report["missing_engagement_ids"] == 1


def test_group_audit_and_population_drift():
    X, _ = prospect_xy()
    df = X.assign(propensity=np.linspace(0.1, 0.8, len(X)))
    groups = group_score_audit(df, "region", min_group_size=10)
    assert len(groups) == 4
    assert groups["mean_ratio_to_overall"].gt(0).all()
    assert population_stability_index(pd.Series(np.linspace(0, 1, 200)),
                                      pd.Series(np.linspace(0, 1, 200))) < 1e-8
    assert population_stability_index(pd.Series(np.linspace(0, 1, 200)),
                                      pd.Series(np.linspace(.5, 1.5, 200))) > 0.1


def test_segment_migration_counts_overlap_only():
    a = pd.DataFrame({"hcp_id": ["1", "2", "3"], "segment_id": [1, 1, 2]})
    b = pd.DataFrame({"hcp_id": ["1", "2", "4"], "segment_id": [2, 1, 3]})
    report = segment_migration(a, b)
    assert report.n_hcps.sum() == 2

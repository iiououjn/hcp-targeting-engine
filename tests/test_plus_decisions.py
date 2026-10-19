from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcp_engine_plus.decisions import ProbabilityCalibrator, constrained_call_plan


def test_calibration_predicts_valid_probabilities():
    p = np.linspace(0.02, 0.98, 200)
    y = (np.arange(200) % 4 < 2).astype(int)
    for method in ("sigmoid", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(p, y)
        predicted = cal.predict([0.1, 0.5, 0.9])
        assert (predicted >= 0).all() and (predicted <= 1).all()
        assert len(predicted) == 3
    with pytest.raises(ValueError, match="held-out"):
        ProbabilityCalibrator().fit([0.2, 0.4], [0, 1])


def test_call_plan_enforces_budget_territory_rep_and_frequency():
    rows = pd.DataFrame({
        "hcp_id": ["a", "b", "c", "d", "e"],
        "propensity": [.9, .8, .7, .6, .5],
        "addressable_trx_3m": [100, 100, 100, 100, 100],
        "territory": ["north", "north", "south", "south", "south"],
        "rep_id": ["r1", "r1", "r2", "r2", "r2"],
        "cost_per_call": [15., 15., 25., 25., 5.],
        "days_since_last_call": [35, 35, 8, 40, 50],
    })
    plan = constrained_call_plan(rows, max_calls=4,
                                 territory_capacity={"north": 1, "south": 2},
                                 rep_capacity={"r1": 1, "r2": 2}, max_cost=45)
    assert list(plan.hcp_id) == ["a", "d", "e"]
    assert plan.cost_per_call.sum() == 45
    assert len(plan) == 3


def test_call_plan_fail_closed_if_required_constraint_data_missing():
    rows = pd.DataFrame({"hcp_id": ["a"], "propensity": [.5],
                         "addressable_trx_3m": [4], "days_since_last_call": [99]})
    with pytest.raises(ValueError, match="territory"):
        constrained_call_plan(rows, max_calls=1, territory_capacity={"a": 1})
    with pytest.raises(ValueError, match="cost_per_call"):
        constrained_call_plan(rows, max_calls=1, max_cost=50)

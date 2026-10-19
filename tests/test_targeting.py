import numpy as np
import pandas as pd

from helpers import features, prospect_xy
from hcp_engine.config import Config
from hcp_engine.propensity import PropensityModel
from hcp_engine.segmentation import HCPSegmenter
from hcp_engine.targeting import build_target_table, segment_opportunity

CFG = Config(call_capacity=100)


def _table():
    f = features()
    X, y = prospect_xy()
    oof = pd.Series(PropensityModel(seed=1).cross_validate(X, y, 3).oof, index=X.index)
    seg = HCPSegmenter(k=5, seed=1).fit(f)
    return build_target_table(f, seg.labels_, seg.name_of(seg.labels_), oof, CFG), seg


def test_established_writers_are_defended_not_scored():
    t, _ = _table()
    est = t[t["status"] == "Established writer"]
    assert len(est) > 0
    assert est["propensity"].isna().all()
    assert est["recommended_play"].str.startswith("Retain").all()
    assert (est["tier"] == "Defend").all()
    assert not est["in_call_plan"].any()


def test_prospect_ranking_tiers_and_call_plan():
    t, _ = _table()
    pr = t[t["status"] == "Prospect"]
    assert pr["propensity"].between(0, 1).all()
    assert np.allclose(pr["expected_value"], pr["propensity"] * pr["addressable_trx_3m"])
    assert pr["priority_rank"].is_unique
    assert pr["in_call_plan"].sum() == CFG.call_capacity
    # Tier 1 is the top slice, and the table is sorted by priority
    n = len(pr)
    assert abs((pr["tier"] == "Tier 1").sum() - CFG.tier1_pct * n) <= 1
    assert t["priority_rank"].dropna().is_monotonic_increasing
    assert pr.loc[pr["tier"] == "Tier 1", "expected_value"].min() >= \
        pr.loc[pr["tier"] == "Tier 3", "expected_value"].max()


def test_every_hcp_has_a_play_and_channel():
    t, _ = _table()
    assert t["recommended_play"].notna().all()
    assert set(t["channel_hint"]) <= {"Digital-first", "Rep-led", "Omnichannel"}
    assert len(t) == len(features())


def test_segment_rollup_adds_up():
    t, seg = _table()
    opp = segment_opportunity(t)
    assert opp["n_hcps"].sum() == len(t)
    assert (opp["n_prospects"] + opp["n_established"] == opp["n_hcps"]).all()
    assert opp["in_call_plan"].sum() == CFG.call_capacity

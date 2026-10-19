import numpy as np
from sklearn.metrics import adjusted_rand_score

from helpers import data, features
from hcp_engine.segmentation import SEGMENT_FEATURES, HCPSegmenter


def _fit(k=None):
    return HCPSegmenter(k=k, k_range=(4, 7), n_boot=4, seed=1).fit(features())


def test_recovers_latent_archetypes():
    seg = _fit()
    truth = data().truth.set_index("hcp_id")["archetype"].loc[features().index]
    assert adjusted_rand_score(truth, seg.labels_) > 0.55


def test_auto_k_in_range_and_selection_table():
    seg = _fit()
    assert 4 <= seg.k_ <= 7
    assert list(seg.selection_["k"]) == [4, 5, 6, 7]
    assert seg.selection_["stability_ari"].between(-1, 1).all()


def test_segments_ordered_by_size_and_named_uniquely():
    seg = _fit(k=5)
    sizes = seg.labels_.value_counts().sort_index()
    assert list(sizes.index) == [1, 2, 3, 4, 5]
    assert (np.diff(sizes.to_numpy()) <= 0).all()
    assert len(set(seg.names_.values())) == 5


def test_predict_matches_training_labels():
    seg = _fit(k=5)
    assert (seg.predict(features()) == seg.labels_).all()


def test_profile_shapes():
    seg = _fit(k=5)
    assert list(seg.z_profile_.columns) == SEGMENT_FEATURES
    assert abs(seg.profile_["pct_hcps"].sum() - 1) < 1e-9

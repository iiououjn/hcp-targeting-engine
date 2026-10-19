"""Shared, cached test data builders."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcp_engine.config import Config  # noqa: E402
from hcp_engine.data import generate_synthetic  # noqa: E402
from hcp_engine.features import build_features, build_labels  # noqa: E402

N = 1500


@lru_cache(maxsize=1)
def data():
    return generate_synthetic(N, seed=7)


@lru_cache(maxsize=1)
def features():
    return build_features(data(), 12)


@lru_cache(maxsize=1)
def labels():
    return build_labels(data(), 12, 3, Config().adoption_trx)


def prospect_xy():
    f, y = features(), labels()
    mask = f["brand_trx_3m"] < Config().established_writer_trx
    return f[mask], y.loc[f.index[mask], "adopted"]

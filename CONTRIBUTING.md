# Contributing

Thanks for contributing. Please open an issue explaining the proposed change before a major refactor.

1. Create a branch from `main`, install `python -m pip install -e '.[dev]'`.
2. Run `pytest -q` and `ruff check src tests` before opening a PR.
3. Include tests for new business logic and avoid changing existing behavior without discussing it.
4. For forward predictions, use only **point-in-time** inputs and keep all outcomes out of scoring features.
5. Never commit real HCP identifiers, private data, credentials, or regulated information.
6. Document the data source, missingness policy, privacy constraints and evaluated model limitations.

A successful unit test run is not a substitute for clinical, privacy, fairness or promotional-compliance reviews.

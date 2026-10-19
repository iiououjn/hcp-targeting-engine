"""Optional extensions CLI: python -m hcp_engine_plus.cli <command>."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from hcp_engine.data import load_csv_dir
from hcp_engine_plus.audits import group_score_audit, quality_report
from hcp_engine_plus.temporal import score_asof, temporal_holdout


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hcp-engine-plus")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("audit-data", help="Audit IDs, duplicates and missing month cells")
    q.add_argument("--input-dir", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--expected-start", type=int)
    q.add_argument("--expected-end", type=int)
    s = sub.add_parser("score-asof", help="Score an absolute month using a trained model")
    s.add_argument("--input-dir", required=True)
    s.add_argument("--model", required=True, help="Trusted local .joblib model from hcp-engine")
    s.add_argument("--as-of", required=True, type=int, help="Sequential integer calendar month index")
    s.add_argument("--lookback", type=int, default=12)
    s.add_argument("--out", required=True)
    t = sub.add_parser("temporal-holdout", help="Train early, evaluate later without lookahead")
    t.add_argument("--train-dir", required=True)
    t.add_argument("--test-dir", required=True)
    t.add_argument("--train-as-of", required=True, type=int)
    t.add_argument("--test-as-of", required=True, type=int)
    t.add_argument("--lookback", type=int, default=12)
    t.add_argument("--out", required=True)
    a = sub.add_parser("audit-scores", help="Group score distribution audit")
    a.add_argument("--targets", required=True)
    a.add_argument("--group", choices=["region", "specialty", "practice_setting"], default="region")
    a.add_argument("--min-group-size", type=int, default=20)
    a.add_argument("--out", required=True)
    d = sub.add_parser("dashboard", help="Produce a standalone HTML report from target CSV")
    d.add_argument("--targets", required=True)
    d.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "audit-data":
        if (args.expected_start is None) != (args.expected_end is None):
            raise ValueError("Provide both --expected-start and --expected-end")
        expected = (range(args.expected_start, args.expected_end + 1)
                    if args.expected_start is not None else None)
        result = quality_report(load_csv_dir(args.input_dir), expected)
        output.write_text(json.dumps(result, indent=2))
    elif args.command == "score-asof":
        # joblib uses pickle; only open YOUR OWN trusted model artifacts.
        model = joblib.load(args.model)
        score_asof(model, load_csv_dir(args.input_dir), args.as_of, args.lookback).to_csv(output, index=False)
    elif args.command == "temporal-holdout":
        result = temporal_holdout(
            load_csv_dir(args.train_dir), load_csv_dir(args.test_dir),
            args.train_as_of, args.test_as_of, lookback_months=args.lookback,
        )
        output.write_text(json.dumps({"train_as_of": result.train_as_of,
                                      "test_as_of": result.test_as_of,
                                      "n_train": result.n_train, "n_test": result.n_test,
                                      "metrics": result.metrics}, indent=2))
        result.predictions.to_csv(output.with_suffix(".csv"), index=False)
    elif args.command == "audit-scores":
        group_score_audit(pd.read_csv(args.targets), args.group,
                          args.min_group_size).to_csv(output, index=False)
    elif args.command == "dashboard":
        from hcp_engine_plus.dashboard import write_dashboard
        write_dashboard(pd.read_csv(args.targets), output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line interface: ``hcp-engine run`` and ``hcp-engine generate-data``."""

from __future__ import annotations

import argparse
import logging
import sys

from hcp_engine import __version__
from hcp_engine.config import Config
from hcp_engine.data import generate_synthetic, load_csv_dir, save_csv_dir
from hcp_engine.pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hcp-engine", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run segmentation + propensity + targeting")
    r.add_argument("--input-dir", help="Directory with hcp_profile.csv, rx_monthly.csv, "
                                       "engagement.csv (default: generate synthetic data)")
    r.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    r.add_argument("--n-hcps", type=int, default=4000, help="Synthetic HCP count")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--k", type=int, default=None, help="Fix the number of segments")
    r.add_argument("--capacity", type=int, default=300, help="Call-plan capacity")
    r.add_argument("--no-explain", action="store_true", help="Skip SHAP (faster)")
    r.add_argument("--no-plots", action="store_true")

    g = sub.add_parser("generate-data", help="Write a synthetic dataset as CSVs")
    g.add_argument("--out", default="data/synthetic")
    g.add_argument("--n-hcps", type=int, default=4000)
    g.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "generate-data":
        path = save_csv_dir(generate_synthetic(args.n_hcps, args.seed), args.out)
        print(f"Wrote synthetic data to {path}/")
        return 0

    cfg = Config(seed=args.seed, k=args.k, call_capacity=args.capacity,
                 explain=not args.no_explain)
    data = load_csv_dir(args.input_dir) if args.input_dir else None
    res = run_pipeline(cfg, data=data, out_dir=args.out, n_hcps=args.n_hcps,
                       make_plots=not args.no_plots)

    m = res.metrics
    gbm, lr = m["propensity_cv"]["gbm"], m["propensity_cv"]["logistic_baseline"]
    print(f"\nHCPs: {m['n_hcps']:,}  (prospects {m['n_prospects']:,}, "
          f"established writers {m['n_established_writers']:,})")
    print(f"Segments: k={m['segmentation']['k']}")
    for sid, name in m["segmentation"]["segment_names"].items():
        print(f"  {name}  (n={m['segmentation']['segment_sizes'][sid]:,})")
    print(f"Propensity CV  AUC={gbm['roc_auc']:.3f}  PR-AUC={gbm['pr_auc']:.3f}  "
          f"top-decile lift={gbm['lift_top_decile']:.2f}x   (LR baseline AUC={lr['roc_auc']:.3f})")
    print(f"Artifacts written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

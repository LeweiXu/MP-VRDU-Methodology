#!/usr/bin/env python3
"""Generate the analysis report, figures, and per-question comparison CSV.

    python scripts/report.py --results results/grid --out results/grid/report.md

Reads only the JSONL result files, so the dissertation's results section
regenerates from raw outputs with one command (reproducibility).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.analysis import build_report, write_comparisons_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/grid")
    ap.add_argument("--out", default=None, help="report markdown path")
    ap.add_argument("--figures", default=None,
                    help="figure dir (default: <results>/figures)")
    ap.add_argument("--comparisons", default=None,
                    help="per-question comparison CSV (default: beside report.md)")
    ap.add_argument("--suite", default=None,
                    help="grid suite YAML; enables the per-RQ hypothesis-verdict "
                         "section (e.g. experiments/grid_local_3b.yaml)")
    args = ap.parse_args()

    fig_dir = args.figures or str(Path(args.results) / "figures")
    md = build_report(args.results, fig_dir=fig_dir, suite=args.suite)
    out = Path(args.out or str(Path(args.results) / "report.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    comparisons = Path(args.comparisons) if args.comparisons else \
        out.with_name("comparisons.csv")
    n_comparisons = write_comparisons_csv(args.results, comparisons)
    print(md)
    print(f"\n-> wrote {out}  (figures in {fig_dir})")
    print(f"-> wrote {comparisons}  ({n_comparisons} question comparisons)")


if __name__ == "__main__":
    main()

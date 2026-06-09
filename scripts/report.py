#!/usr/bin/env python3
"""Generate the full analysis report (tables + figures) from a results dir.

    python scripts/report.py --results results/grid --out results/grid/report.md

Reads only the JSONL result files, so the dissertation's results section
regenerates from raw outputs with one command (reproducibility).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.analysis import build_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/grid")
    ap.add_argument("--out", default=None, help="report markdown path")
    ap.add_argument("--figures", default=None,
                    help="figure dir (default: <results>/figures)")
    args = ap.parse_args()

    fig_dir = args.figures or str(Path(args.results) / "figures")
    md = build_report(args.results, fig_dir=fig_dir)
    out = args.out or str(Path(args.results) / "report.md")
    Path(out).write_text(md, encoding="utf-8")
    print(md)
    print(f"\n-> wrote {out}  (figures in {fig_dir})")


if __name__ == "__main__":
    main()

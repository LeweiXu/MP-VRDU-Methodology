#!/usr/bin/env python3
"""Is condition A significantly better than condition B? (paired bootstrap)

    python scripts/compare.py results/grid/A_retrieval/colqwen...jsonl \
                              results/grid/A_retrieval/bm25...jsonl

Pairs the two runs' per-question correctness by qid and reports the accuracy
difference (A − B) with a 95% CI; "significant" means the CI excludes 0.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.analysis import paired_diff_test
from mpvrdu.results import read_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="result JSONL for condition A")
    ap.add_argument("b", help="result JSONL for condition B")
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()

    res = paired_diff_test(read_rows(args.a), read_rows(args.b),
                           n_resamples=args.resamples)
    lo, hi = res["ci"]
    print(f"paired on {res['n']} questions")
    print(f"accuracy(A) − accuracy(B) = {res['diff']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("SIGNIFICANT (CI excludes 0)" if res["significant"]
          else "not significant (CI includes 0)")


if __name__ == "__main__":
    main()

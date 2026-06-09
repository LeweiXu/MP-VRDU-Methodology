#!/usr/bin/env python3
"""Fingerprint the eval documents for a contamination audit (context.md §10).

    python scripts/contamination_audit.py --slice full --out results/audit.json
    # optionally diff against a list of known-training doc ids / sha256 / titles:
    python scripts/contamination_audit.py --slice full --against known_docs.txt

Writes a manifest (doc_id, sha256, num_pages, title) and prints a methodology
note documenting which generator + encoders were used so the check is on record.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.config import DataConfig
from mpvrdu.data.audit import build_doc_manifest, check_overlap
from mpvrdu.data.load import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mmlongbench-doc")
    ap.add_argument("--slice", default="full")
    ap.add_argument("--out", default="results/audit.json")
    ap.add_argument("--against", default=None,
                    help="file with known-training identifiers (one per line)")
    args = ap.parse_args()

    ds = load_dataset(DataConfig(name=args.name, slice=args.slice))
    manifest = build_doc_manifest(ds)
    present = [m for m in manifest if m.get("present")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"fingerprinted {len(present)}/{len(manifest)} documents -> {out}")
    if args.against:
        known = set(Path(args.against).read_text().splitlines())
        hits = check_overlap(manifest, known)
        print(f"OVERLAP with known-training set: {len(hits)} documents")
        for h in hits:
            print(f"  - {h['doc_id']}  {h.get('title','')}")
    print("\n--- methodology note (fill in + put in the dissertation) ---")
    print("Generator: Qwen2.5-VL-{7B|32B}-Instruct (Apache-2.0).")
    print("Visual retrievers: ColPali (vidore/colpali-v1.3), ColQwen2.5 "
          "(vidore/colqwen2.5-v0.2).")
    print("Eval docs fingerprinted above; MMLongBench-Doc sources DUDE/SlideVQA/"
          "ChartQA — note any encoder/generator trained on those pools.")


if __name__ == "__main__":
    main()

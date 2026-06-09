#!/usr/bin/env python3
"""Inspect what one experiment run actually produced, question by question.

Reads a result JSONL and shows, per question: the prediction vs the gold answer,
and — the point of this tool — the SELECTED pages vs the GOLD evidence pages, so
you can see whether a wrong answer was a retrieval miss (evidence not selected)
or a generation miss (evidence selected but answered wrong).

    # everything in a run:
    python scripts/inspect_run.py results/grid/A_retrieval/A_retrieval__bm25__image__4__<hash>.jsonl
    # only the questions it got wrong:
    python scripts/inspect_run.py <file> --errors-only
    # only answerable questions where retrieval missed a gold page:
    python scripts/inspect_run.py <file> --retrieval-misses
    # one question, full text:
    python scripts/inspect_run.py <file> --qid <qid> --full

With a DIRECTORY instead of a file, it lists the runs available to inspect.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.results import iter_results

MARK = {True: "✓", False: "✗"}


def _truncate(s, n, full):
    s = "" if s is None else str(s).replace("\n", " ")
    return s if full or len(s) <= n else s[: n - 1] + "…"


def _load(path):
    meta, rows = None, []
    for r in iter_results(path):
        if r.get("kind") == "meta":
            meta = r
        elif "correct" in r:
            rows.append(r)
    return meta, rows


def _gold_pages_1based(r):
    """Gold evidence pages are stored 1-based; return as a set for membership."""
    return set(r.get("evidence_pages") or [])


def _selected_1based(r):
    """Selected pages are stored 0-based; show 1-based to compare with gold."""
    return [p + 1 for p in (r.get("selected_pages_0based") or [])]


def _print_meta(meta, path, n_rows):
    print("=" * 78)
    if meta:
        cfg = meta.get("config", {})
        retr = cfg.get("retrieval", {})
        gen = cfg.get("generation", {})
        rep = cfg.get("representation", {})
        print(f"run:        {cfg.get('name')}   (hash {meta.get('config_hash')})")
        post = []
        if retr.get("k_strategy", "fixed") != "fixed":
            post.append(f"k_strategy={retr['k_strategy']}")
        if retr.get("rerank", "none") != "none":
            post.append(f"rerank={retr['rerank']}")
        if retr.get("expand", "none") != "none":
            post.append(f"expand={retr['expand']}(w{retr.get('expand_window')})")
        post_s = ("  [" + ", ".join(post) + "]") if post else ""
        print(f"retrieval:  {retr.get('method')} top_k={retr.get('top_k')}{post_s}")
        print(f"represent:  parser={rep.get('parser')} chunking={rep.get('chunking')}")
        print(f"generation: {gen.get('generator')} modality={gen.get('modality')} "
              f"model={gen.get('model_id')}")
    print(f"file:       {path}   ({n_rows} questions)")
    print("=" * 78)


def _keep(r, args):
    if args.qid and r.get("qid") != args.qid:
        return False
    if args.type and r.get("question_type") != args.type:
        return False
    if args.errors_only and bool(r.get("correct")):
        return False
    if args.retrieval_misses:
        # answerable question whose gold evidence was not fully selected
        if not r.get("gold_answerable"):
            return False
        gold = _gold_pages_1based(r)
        sel = set(_selected_1based(r))
        if not gold or gold <= sel:
            return False
    return True


def _print_row(r, full):
    correct = bool(r.get("correct"))
    gold = _gold_pages_1based(r)
    sel = _selected_1based(r)
    sel_set = set(sel)
    # annotate each gold page with a hit/miss marker
    if gold:
        gold_anno = " ".join(f"{p}{'✓' if p in sel_set else '✗'}"
                             for p in sorted(gold))
    else:
        gold_anno = "(none / unanswerable)"
    recall = r.get("recall_at_k")
    rec_s = f"{recall:.2f}" if isinstance(recall, (int, float)) else "–"

    print(f"\n[{MARK[correct]}] {r.get('qid')}  ({r.get('question_type')}, "
          f"fmt={r.get('answer_format')})")
    print(f"    Q:        {_truncate(r.get('question'), 100, full)}")
    print(f"    gold:     {_truncate(r.get('gold'), 100, full)}")
    print(f"    pred:     {_truncate(r.get('pred'), 100, full)}"
          + ("   [abstained]" if r.get("pred_abstained") else ""))
    print(f"    gold pgs: {gold_anno}   (1-based; ✓=selected ✗=missed)")
    print(f"    selected: {sel}   recall@k={rec_s}  n_selected={r.get('n_selected')}")
    if r.get("evidence_sources"):
        print(f"    src:      {r.get('evidence_sources')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="result JSONL file (or a directory to list runs)")
    ap.add_argument("--errors-only", action="store_true",
                    help="show only incorrectly answered questions")
    ap.add_argument("--retrieval-misses", action="store_true",
                    help="show only answerable Qs where a gold page wasn't selected")
    ap.add_argument("--qid", default=None, help="show only this question id")
    ap.add_argument("--type", default=None,
                    choices=["single", "cross", "unanswerable"])
    ap.add_argument("--limit", type=int, default=None, help="cap rows shown")
    ap.add_argument("--full", action="store_true", help="don't truncate text")
    args = ap.parse_args()

    path = Path(args.path)
    if path.is_dir():
        files = sorted(path.glob("**/*.jsonl"))
        print(f"{len(files)} result files under {path}:")
        for f in files:
            print(f"  {f}")
        print("\nPass one of these files to inspect it.")
        return

    meta, rows = _load(path)
    _print_meta(meta, path, len(rows))

    shown = correct = 0
    retr_miss = gen_miss = 0
    for r in rows:
        if bool(r.get("correct")):
            correct += 1
        # tally failure mode over ALL rows (not just shown)
        if not bool(r.get("correct")) and r.get("gold_answerable"):
            gold = _gold_pages_1based(r)
            sel = set(_selected_1based(r))
            if gold and not (gold <= sel):
                retr_miss += 1
            else:
                gen_miss += 1
        if not _keep(r, args):
            continue
        if args.limit is not None and shown >= args.limit:
            continue
        _print_row(r, args.full)
        shown += 1

    print("\n" + "-" * 78)
    n = len(rows)
    print(f"accuracy: {correct}/{n} = {correct / n:.3f}" if n else "no rows")
    print(f"wrong & answerable: retrieval-miss={retr_miss} (gold page not "
          f"selected), generation-miss={gen_miss} (evidence selected, answered "
          f"wrong)")
    print(f"shown: {shown}" + (f" (limited to {args.limit})"
                               if args.limit is not None else ""))


if __name__ == "__main__":
    main()

"""Aggregate result JSONL into summary tables + breakdowns (Stage 7).

All metrics are recomputed from the per-question rows (not trusted from a run's
stdout), so the tables are a pure function of the JSONL on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ..results import iter_results

# config fields surfaced as table columns (dotted paths into the meta config)
CONDITION_FIELDS = [
    "retrieval.method", "retrieval.top_k", "generation.modality",
    "generation.generator", "representation.parser", "representation.chunking",
]
# Tier-1 post-processing toggles — extracted into each summary's condition so the
# report can surface them, but kept OUT of CONDITION_FIELDS so the default tables
# stay compact (they only matter for the Tier-1 sections).
TIER1_FIELDS = [
    "retrieval.k_strategy", "retrieval.rerank", "retrieval.expand",
    "retrieval.expand_window",
]
# RQ-axis fields the per-RQ reporter matches conditions on (kept out of the
# default columns; the reasoning axis only matters for the RQ3 section).
RQ_FIELDS = ["generation.reasoning"]


def _get(d: dict, dotted: str, default=None):
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _split_meta_rows(path: str | Path):
    """Return (meta, scored_rows). Only per-question rows with a 'correct' field
    count as scored rows, so recall-only / summary files are ignored cleanly."""
    meta = None
    rows = []
    for r in iter_results(path):
        if r.get("kind") == "meta":
            meta = r
        elif "kind" in r:
            continue                  # recall_summary or other non-scored rows
        elif "correct" in r:
            rows.append(r)
    return meta, rows


def _binary_f1(gold: list[bool], pred: list[bool]) -> float:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(values: list[float], n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values` (e.g. per-question 0/1
    correctness). Returns (low, high). With 1091 questions this turns point
    estimates into defensible intervals for method comparisons."""
    import numpy as np

    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_resamples, len(arr)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def paired_diff_test(rows_a: list[dict], rows_b: list[dict],
                     n_resamples: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap on accuracy difference (A − B) over questions matched by
    qid. Returns the mean diff, its CI, and whether it excludes 0 (significant)."""
    import numpy as np

    a = {r["qid"]: int(bool(r.get("correct"))) for r in rows_a if "qid" in r}
    b = {r["qid"]: int(bool(r.get("correct"))) for r in rows_b if "qid" in r}
    common = sorted(set(a) & set(b))
    if not common:
        return {"n": 0, "diff": 0.0, "ci": (0.0, 0.0), "significant": False}
    d = np.array([a[q] - b[q] for q in common], dtype=float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_resamples, len(d)))].mean(axis=1)
    lo, hi = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    return {"n": len(common), "diff": float(d.mean()), "ci": (lo, hi),
            "significant": lo > 0 or hi < 0}


def summarize_run(path: str | Path) -> Optional[dict]:
    """Summarise one result file: overall + by question-type + by evidence-source.

    Returns None for files with no scored question rows (e.g. recall-only files).
    """
    meta, rows = _split_meta_rows(path)
    if not rows:
        return None
    n = len(rows)
    correct = [bool(r.get("correct")) for r in rows]
    accuracy = _mean([float(c) for c in correct])
    f1 = _binary_f1([bool(r.get("gold_answerable")) for r in rows],
                    [not bool(r.get("pred_abstained")) for r in rows])
    recall = _mean([float(r["recall_at_k"]) for r in rows if "recall_at_k" in r])

    by_type: dict[str, dict] = {}
    for r in rows:
        t = r.get("question_type", "?")
        by_type.setdefault(t, {"n": 0, "correct": 0})
        by_type[t]["n"] += 1
        by_type[t]["correct"] += int(bool(r.get("correct")))
    for t, d in by_type.items():
        d["accuracy"] = d["correct"] / d["n"] if d["n"] else 0.0

    by_source: dict[str, dict] = {}
    for r in rows:
        for src in (r.get("evidence_sources") or ["(none)"]):
            by_source.setdefault(src, {"n": 0, "correct": 0})
            by_source[src]["n"] += 1
            by_source[src]["correct"] += int(bool(r.get("correct")))
    for s, d in by_source.items():
        d["accuracy"] = d["correct"] / d["n"] if d["n"] else 0.0

    # abstention / hallucination: on unanswerable Qs, did the model wrongly
    # answer (hallucinate)? on answerable Qs, did it wrongly abstain?
    unans = [r for r in rows if not bool(r.get("gold_answerable"))]
    ans = [r for r in rows if bool(r.get("gold_answerable"))]
    hallucination_rate = _mean([0.0 if bool(r.get("pred_abstained")) else 1.0
                                for r in unans]) if unans else 0.0
    over_abstention_rate = _mean([1.0 if bool(r.get("pred_abstained")) else 0.0
                                  for r in ans]) if ans else 0.0

    # cost proxy (only if the generator logged usage)
    cost = {}
    if any("input_tokens" in r for r in rows):
        cost = {
            "mean_input_tokens": _mean([float(r.get("input_tokens", 0)) for r in rows]),
            "mean_output_tokens": _mean([float(r.get("output_tokens", 0)) for r in rows]),
            "total_gen_seconds": sum(float(r.get("gen_seconds", 0)) for r in rows),
            "mean_gen_seconds": _mean([float(r.get("gen_seconds", 0)) for r in rows]),
        }

    # mean pages actually fed to the generator (varies under adaptive-k/expansion)
    mean_n_selected = _mean([float(r["n_selected"]) for r in rows
                             if "n_selected" in r])

    lo, hi = bootstrap_ci([float(c) for c in correct])
    cfg = (meta or {}).get("config", {})
    condition = {f: _get(cfg, f) for f in CONDITION_FIELDS + TIER1_FIELDS + RQ_FIELDS}
    return {
        "path": str(path),
        "config_hash": (meta or {}).get("config_hash"),
        "name": cfg.get("name"),
        "condition": condition,
        "n": n,
        "accuracy": accuracy,
        "accuracy_ci": [lo, hi],
        "f1": f1,
        "mean_recall_at_k": recall,
        "mean_n_selected": mean_n_selected,
        "hallucination_rate": hallucination_rate,
        "over_abstention_rate": over_abstention_rate,
        "by_question_type": by_type,
        "by_evidence_source": by_source,
        "cost": cost,
    }


def aggregate_dir(results_dir: str | Path, pattern: str = "*.jsonl") -> list[dict]:
    """Summarise every result file in a directory, sorted by name then hash."""
    paths = sorted(Path(results_dir).glob(pattern))
    summaries = [s for s in (summarize_run(p) for p in paths) if s is not None]
    summaries.sort(key=lambda s: (s.get("name") or "", s.get("config_hash") or ""))
    return summaries


def to_markdown_table(summaries: Iterable[dict],
                      columns: Optional[list[str]] = None) -> str:
    """Render run summaries as a Markdown table (condition columns + metrics)."""
    summaries = list(summaries)
    cond_cols = columns or CONDITION_FIELDS
    headers = cond_cols + ["n", "accuracy", "95% CI", "f1", "recall@k", "halluc."]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for s in summaries:
        ci = s.get("accuracy_ci", [0, 0])
        cells = [str(s["condition"].get(c, "")) for c in cond_cols]
        cells += [str(s["n"]), f"{s['accuracy']:.3f}",
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]", f"{s['f1']:.3f}",
                  f"{s['mean_recall_at_k']:.3f}",
                  f"{s.get('hallucination_rate', 0):.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

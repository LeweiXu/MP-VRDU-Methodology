"""Turn a directory of grid results into the dissertation's analysis (Stage 7).

Everything is derived from the per-run summaries (themselves pure functions of
the JSONL), so the whole report regenerates from raw outputs with one command.

Key analyses:
- oracle-gap decomposition: per method, how much retrieval LIFTED over the
  no-retrieval floor, and how much GAP to the oracle ceiling remains.
- top-k curves: accuracy / recall vs k (top-k is non-monotonic — context.md §5).
- recall -> accuracy correlation: does better retrieval actually help the answer?
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .aggregate import (CONDITION_FIELDS, TIER1_FIELDS, aggregate_dir,
                        paired_diff_test, to_markdown_table)
from ..results import read_rows

BASELINE_METHODS = {"none", "oracle"}
QUESTION_TYPES = ["single", "cross", "unanswerable"]
# default values that mark a Tier-1 toggle as "off"
TIER1_DEFAULTS = {"retrieval.k_strategy": "fixed", "retrieval.rerank": "none",
                  "retrieval.expand": "none"}


def _cond(s: dict, key: str):
    return s.get("condition", {}).get(key)


def _substudy(s: dict) -> str:
    """The sub-study a run belongs to = its parent directory name under the
    results root (run_grid writes results/grid/<substudy>/<name>.jsonl)."""
    return Path(s["path"]).parent.name


def _name(s: dict) -> str:
    return s.get("name") or Path(s["path"]).stem


def oracle_gap(summaries: list[dict]) -> list[dict]:
    """Per retrieval run: lift over the no-retrieval floor + gap to oracle ceiling
    (matched within the same generation modality)."""
    floor, ceil = {}, {}
    for s in summaries:
        m, method = _cond(s, "generation.modality"), _cond(s, "retrieval.method")
        if method == "none":
            floor[m] = s["accuracy"]
        elif method == "oracle":
            ceil[m] = s["accuracy"]
    rows = []
    for s in summaries:
        method = _cond(s, "retrieval.method")
        if method in BASELINE_METHODS:
            continue
        m = _cond(s, "generation.modality")
        f, c = floor.get(m), ceil.get(m)
        rows.append({
            "method": method, "modality": m, "top_k": _cond(s, "retrieval.top_k"),
            "accuracy": s["accuracy"],
            "lift_over_floor": None if f is None else round(s["accuracy"] - f, 4),
            "gap_to_ceiling": None if c is None else round(c - s["accuracy"], 4),
            "pct_of_ceiling": None if not c else round(s["accuracy"] / c, 4),
        })
    return rows


def topk_series(summaries: list[dict], metric: str = "accuracy") -> dict:
    """{ 'method/modality' -> [(k, value), ...] } for retrieval runs."""
    series: dict[str, list] = {}
    for s in summaries:
        method, mod = _cond(s, "retrieval.method"), _cond(s, "generation.modality")
        k = _cond(s, "retrieval.top_k")
        if method in BASELINE_METHODS or k is None:
            continue
        val = s["accuracy"] if metric == "accuracy" else s["mean_recall_at_k"]
        series.setdefault(f"{method}/{mod}", []).append((int(k), val))
    return {key: sorted(v) for key, v in series.items()}


def pairwise_significance(summaries: list[dict]) -> list[dict]:
    """Paired bootstrap between every pair of retrieval methods sharing a
    (modality, top_k) block — the dissertation's 'is A really better than B'."""
    groups: dict[tuple, list[dict]] = {}
    for s in summaries:
        if _cond(s, "retrieval.method") in BASELINE_METHODS:
            continue
        key = (_cond(s, "generation.modality"), _cond(s, "retrieval.top_k"))
        groups.setdefault(key, []).append(s)
    out = []
    for (modality, k), runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        runs = sorted(runs, key=lambda s: _cond(s, "retrieval.method"))
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i], runs[j]
                res = paired_diff_test(read_rows(a["path"]), read_rows(b["path"]))
                out.append({
                    "modality": modality, "top_k": k,
                    "a": _cond(a, "retrieval.method"),
                    "b": _cond(b, "retrieval.method"),
                    "diff": round(res["diff"], 4), "ci": res["ci"],
                    "significant": res["significant"],
                })
    return out


def recall_accuracy_correlation(summaries: list[dict]) -> Optional[float]:
    """Pearson r between mean recall@k and accuracy across retrieval runs."""
    import numpy as np

    pts = [(s["mean_recall_at_k"], s["accuracy"]) for s in summaries
           if _cond(s, "retrieval.method") not in BASELINE_METHODS]
    if len(pts) < 3:
        return None
    x, y = zip(*pts)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def seed_variance(summaries: list[dict]) -> list[dict]:
    """Group runs that differ only by seed (same auto-name) and report accuracy
    mean ± std across seeds. Only groups with >1 run are returned."""
    import numpy as np

    groups: dict[str, list[float]] = {}
    for s in summaries:
        groups.setdefault(s.get("name") or "?", []).append(s["accuracy"])
    out = []
    for name, accs in sorted(groups.items()):
        if len(accs) > 1:
            out.append({"name": name, "n_seeds": len(accs),
                        "mean_accuracy": float(np.mean(accs)),
                        "std_accuracy": float(np.std(accs))})
    return out


def sanity_checks(summaries: list[dict]) -> list[str]:
    """Return human-readable warnings if results look broken (context.md §9)."""
    warns = []
    floor, ceil = {}, {}
    for s in summaries:
        m, method = _cond(s, "generation.modality"), _cond(s, "retrieval.method")
        if not (0.0 <= s["accuracy"] <= 1.0):
            warns.append(f"{s['name']}: accuracy {s['accuracy']} out of [0,1]")
        if method == "none":
            floor[m] = s["accuracy"]
        elif method == "oracle":
            ceil[m] = s["accuracy"]
    for m in ceil:
        if m in floor and ceil[m] < floor[m]:
            warns.append(f"modality={m}: oracle ({ceil[m]:.3f}) < no-retrieval "
                         f"({floor[m]:.3f}) — likely input-packing or judge bug")
    for s in summaries:
        method = _cond(s, "retrieval.method")
        if method not in BASELINE_METHODS and s["mean_recall_at_k"] < 0.02:
            warns.append(f"{s['name']}: recall@k≈0 — retriever likely broken")
    return warns


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def question_type_table(summaries: list[dict]) -> str:
    """Accuracy split by single / cross / unanswerable, per condition."""
    headers = ["run"] + [f"{t} acc (n)" for t in QUESTION_TYPES] + ["overall"]
    rows = []
    for s in sorted(summaries, key=_name):
        bt = s.get("by_question_type", {})
        cells = [_name(s)]
        for t in QUESTION_TYPES:
            d = bt.get(t)
            cells.append(f"{d['accuracy']:.3f} ({d['n']})" if d else "–")
        cells.append(f"{s['accuracy']:.3f}")
        rows.append(cells)
    return _md_table(headers, rows)


def abstention_table(summaries: list[dict]) -> str:
    """Hallucination (answered an unanswerable Q) vs over-abstention (abstained on
    an answerable Q) — the two failure modes on the 244 unanswerable items."""
    headers = ["run", "top_k", "hallucination↓", "over-abstention↓", "f1"]
    rows = []
    for s in sorted(summaries, key=_name):
        rows.append([_name(s), _cond(s, "retrieval.top_k"),
                     f"{s.get('hallucination_rate', 0):.3f}",
                     f"{s.get('over_abstention_rate', 0):.3f}",
                     f"{s['f1']:.3f}"])
    return _md_table(headers, rows)


def evidence_source_table(summaries: list[dict]) -> str:
    """Accuracy by gold evidence source (text / layout / chart / table / image).
    Shows which modalities of evidence a condition handles well."""
    sources: list[str] = []
    for s in summaries:
        for src in s.get("by_evidence_source", {}):
            if src not in sources:
                sources.append(src)
    sources = sorted(sources)
    headers = ["run"] + sources
    rows = []
    for s in sorted(summaries, key=_name):
        bs = s.get("by_evidence_source", {})
        cells = [_name(s)]
        for src in sources:
            d = bs.get(src)
            cells.append(f"{d['accuracy']:.3f} ({d['n']})" if d else "–")
        rows.append(cells)
    return _md_table(headers, rows), sources


def cost_table(summaries: list[dict]) -> Optional[str]:
    """Per-question cost proxy: tokens to the generator + wall-clock."""
    rows = []
    for s in sorted(summaries, key=_name):
        c = s.get("cost") or {}
        if not c:
            continue
        rows.append([_name(s), s.get("mean_n_selected"),
                     round(c.get("mean_input_tokens", 0)),
                     round(c.get("mean_output_tokens", 0)),
                     f"{c.get('mean_gen_seconds', 0):.2f}",
                     f"{c.get('total_gen_seconds', 0):.1f}"])
    if not rows:
        return None
    headers = ["run", "mean pages", "mean in-tok", "mean out-tok",
               "mean gen s", "total gen s"]
    return _md_table(headers, rows)


def tier1_table(summaries: list[dict]) -> Optional[str]:
    """Runs that use a Tier-1 toggle (adaptive-k / rerank / expansion), with the
    metrics those toggles move: accuracy, recall@k, and mean pages selected
    (which becomes variable under adaptive-k / expansion)."""
    def _on(s):
        return any(_cond(s, f) not in (None, TIER1_DEFAULTS[f])
                   for f in TIER1_DEFAULTS)

    rows = []
    for s in sorted(summaries, key=_name):
        if not _on(s):
            continue
        rows.append([_name(s), _cond(s, "retrieval.method"),
                     _cond(s, "retrieval.top_k"),
                     _cond(s, "retrieval.k_strategy"),
                     _cond(s, "retrieval.rerank"),
                     _cond(s, "retrieval.expand"),
                     f"{s['accuracy']:.3f}", f"{s['mean_recall_at_k']:.3f}",
                     f"{s.get('mean_n_selected', 0):.2f}"])
    if not rows:
        return None
    headers = ["run", "method", "top_k", "k_strategy", "rerank", "expand",
               "accuracy", "recall@k", "mean pages"]
    return _md_table(headers, rows)


def _gap_table(rows: list[dict]) -> str:
    headers = ["method", "modality", "top_k", "accuracy", "lift_over_floor",
               "gap_to_ceiling", "pct_of_ceiling"]
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def _by_substudy_section(summaries: list[dict]) -> list[str]:
    """One compact metrics table per sub-study (grouped by results subdir)."""
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        groups.setdefault(_substudy(s), []).append(s)
    parts = ["## Conditions by sub-study\n"]
    # render the no-grouping case (flat dir) as a single block
    for sub in sorted(groups):
        runs = sorted(groups[sub], key=_name)
        parts += [f"### {sub}  ({len(runs)} runs)\n",
                  to_markdown_table(runs), ""]
    return parts


def build_report(results_dir: str | Path, fig_dir: Optional[str | Path] = None) -> str:
    """Assemble the full markdown report; write figures if matplotlib is present."""
    summaries = aggregate_dir(results_dir, pattern="**/*.jsonl")
    if not summaries:
        return "# MP-VRDU results\n\n_No completed results found._\n"

    import datetime as _dt

    n_q = max((s["n"] for s in summaries), default=0)
    parts = ["# MP-VRDU results\n",
             f"_{len(summaries)} conditions · up to {n_q} questions each · "
             f"generated {_dt.date.today().isoformat()} from `{results_dir}`._\n",
             "> Columns: method · top_k · modality · generator · parser · chunking "
             "· n · accuracy (95% bootstrap CI) · answerability-F1 · recall@k · "
             "hallucination rate.\n"]

    warns = sanity_checks(summaries)
    if warns:
        parts += ["## ⚠ Sanity warnings\n"] + [f"- {w}" for w in warns] + [""]

    parts += ["## All conditions\n", to_markdown_table(summaries), ""]

    # per-sub-study tables (only adds value when results are grouped in subdirs)
    if len({_substudy(s) for s in summaries}) > 1:
        parts += _by_substudy_section(summaries)

    gap = oracle_gap(summaries)
    if gap:
        parts += ["## Oracle-gap decomposition\n",
                  "Lift = accuracy − no-retrieval floor; gap = oracle ceiling − "
                  "accuracy (matched per generation modality).\n",
                  _gap_table(gap), ""]

    sv = seed_variance(summaries)
    if sv:
        parts += ["## Seed variance (accuracy mean ± std)\n"]
        parts += [f"- {g['name']}: {g['mean_accuracy']:.3f} ± {g['std_accuracy']:.3f} "
                  f"(n={g['n_seeds']})" for g in sv]
        parts += [""]

    sig = pairwise_significance(summaries)
    if sig:
        parts += ["## Pairwise significance (retrieval methods)\n",
                  "Paired bootstrap within each (modality, top_k) block; "
                  "'significant' = 95% CI on the accuracy difference excludes 0.\n",
                  "| modality | top_k | A | B | acc(A)−acc(B) | 95% CI | sig? |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r_ in sig:
            ci = r_["ci"]
            parts.append(f"| {r_['modality']} | {r_['top_k']} | {r_['a']} | {r_['b']} "
                         f"| {r_['diff']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                         f"| {'**yes**' if r_['significant'] else 'no'} |")
        parts.append("")

    r = recall_accuracy_correlation(summaries)
    if r is not None:
        parts += [f"## Recall → accuracy\n\nPearson r(mean recall@k, accuracy) = "
                  f"**{r:.3f}** across {len([s for s in summaries if _cond(s,'retrieval.method') not in BASELINE_METHODS])} "
                  "retrieval runs. (Does better retrieval actually help the answer?)\n"]

    # accuracy split by question type (single / cross / unanswerable)
    parts += ["## Accuracy by question type\n",
              "Cross-page is the MP-distinctive type; unanswerable tests "
              "abstention.\n", question_type_table(summaries), ""]

    # abstention behaviour (hallucination vs over-abstention)
    parts += ["## Abstention behaviour\n",
              "hallucination = answered a gold-unanswerable question; "
              "over-abstention = abstained on a gold-answerable one (lower is "
              "better for both).\n", abstention_table(summaries), ""]

    # accuracy by gold evidence source
    src_tbl, srcs = evidence_source_table(summaries)
    if srcs:
        parts += ["## Accuracy by evidence source\n",
                  "Per gold evidence modality; cells are acc (n).\n", src_tbl, ""]

    # Tier-1 post-processing breakdown
    t1 = tier1_table(summaries)
    if t1:
        parts += ["## Retrieval post-processing (Tier-1)\n",
                  "Runs using adaptive-k (#1), rerank (#3) or expansion (#4/#5). "
                  "`mean pages` is how many pages were fed to the generator — it "
                  "becomes variable under adaptive-k / expansion.\n", t1, ""]

    # cost proxy
    ct = cost_table(summaries)
    if ct:
        parts += ["## Cost\n",
                  "Per-question tokens to the generator + wall-clock "
                  "(context.md §8 cost proxy).\n", ct, ""]

    # figures (best-effort)
    if fig_dir is not None:
        try:
            figs = _write_figures(summaries, fig_dir)
            if figs:
                parts += ["## Figures\n"] + [f"![{f.stem}]({f})" for f in figs] + [""]
        except Exception as e:  # matplotlib missing or backend issue
            parts += [f"_(figures skipped: {e})_\n"]

    return "\n".join(parts)


def _write_figures(summaries: list[dict], fig_dir: str | Path) -> list[Path]:
    from .figures import method_bars, scatter, topk_curve

    fig_dir = Path(fig_dir)
    out = []
    # accuracy vs cost (mean input tokens) — the efficiency frontier
    eff = {s["name"]: (s["cost"]["mean_input_tokens"], s["accuracy"])
           for s in summaries if s.get("cost", {}).get("mean_input_tokens")}
    if len(eff) >= 2:
        out.append(scatter(eff, "mean input tokens", "accuracy",
                           fig_dir / "efficiency_frontier.png",
                           title="Accuracy vs cost"))
    acc = topk_series(summaries, "accuracy")
    if acc:
        out.append(topk_curve(acc, "accuracy", fig_dir / "topk_accuracy.png",
                              title="Accuracy vs top-k"))
    rec = topk_series(summaries, "recall")
    if rec:
        out.append(topk_curve(rec, "recall@k", fig_dir / "topk_recall.png",
                              title="Recall vs top-k"))
    # method bars at the most common k, with floor/ceiling for that modality
    gap = oracle_gap(summaries)
    if gap:
        vals = {f"{g['method']}/{g['modality']}@{g['top_k']}": g["accuracy"]
                for g in gap}
        floors = [s["accuracy"] for s in summaries if _cond(s, "retrieval.method") == "none"]
        ceils = [s["accuracy"] for s in summaries if _cond(s, "retrieval.method") == "oracle"]
        out.append(method_bars(
            vals, "accuracy", fig_dir / "method_comparison.png",
            floor=min(floors) if floors else None,
            ceiling=max(ceils) if ceils else None, title="Method comparison"))
    return out

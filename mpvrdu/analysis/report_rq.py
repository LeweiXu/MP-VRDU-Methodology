"""Per-RQ hypothesis-verdict reporting (docs/pivot.md Step 3, RQ spec §4).

The analysis bar (research_questions.md §4): every RQ result must report the
DIRECTIONAL hypothesis, the PER-SUBSET breakdown on the discriminating subset
(not just a grand mean), the effect vs the declared control with a CI / paired
test, and a HELD / REFUTED / MIXED / NULL verdict. A predicted null (e.g. H3c:
ToT) is a first-class finding, not an empty cell.

This module joins the grid's per-study RQ metadata (suite YAML; see
``experiment.study_metadata``) to the result JSONL and renders one section per RQ.
It reuses the existing analysis backbone (paired bootstrap, subset slicing); it
adds the FRAMING and the verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .aggregate import (CONDITION_FIELDS, RQ_FIELDS, TIER1_FIELDS,
                        paired_diff_test)
from ..results import read_rows

# all condition fields a verdict matches on (so "everything but the tested key"
# is held equal between the two compared runs)
_MATCH_FIELDS = CONDITION_FIELDS + TIER1_FIELDS + RQ_FIELDS


def _substudy(s: dict) -> str:
    return Path(s["path"]).parent.name


def _cond(s: dict, key: str):
    return s.get("condition", {}).get(key)


# --------------------------------------------------------------------------- #
# subset axes (the "where the effect lives" dimension)
# --------------------------------------------------------------------------- #
def doc_length_bin(num_pages: Optional[int]) -> str:
    """Bin a document length for H4b (coarse-to-fine grows with length)."""
    if not num_pages:
        return "unknown"
    if num_pages <= 10:
        return "short(≤10)"
    if num_pages <= 30:
        return "medium(11-30)"
    return "long(>30)"


def _row_in_cell(row: dict, axis: str, value: str) -> bool:
    if axis == "question_type":
        return row.get("question_type") == value
    if axis == "evidence_source":
        return value in (row.get("evidence_sources") or [])
    if axis == "doc_length":
        return doc_length_bin(row.get("doc_num_pages")) == value
    return False


def _cell_values(rows: list[dict], axis: str) -> list[str]:
    """Distinct subset cells present in the rows, in a stable order."""
    seen: list[str] = []
    for r in rows:
        if axis == "evidence_source":
            vals = r.get("evidence_sources") or ["(none)"]
        elif axis == "question_type":
            vals = [r.get("question_type", "?")]
        elif axis == "doc_length":
            vals = [doc_length_bin(r.get("doc_num_pages"))]
        else:
            vals = []
        for v in vals:
            if v not in seen:
                seen.append(v)
    return sorted(seen)


def _subset_accuracy(rows: list[dict], axis: str) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for v in _cell_values(rows, axis):
        cell = [r for r in rows if _row_in_cell(r, axis, v)]
        n = len(cell)
        acc = sum(bool(r.get("correct")) for r in cell) / n if n else 0.0
        out[v] = (acc, n)
    return out


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def _verdict(diff: float, significant: bool, direction: str) -> str:
    """Map a paired effect (A − B) + its significance onto a hypothesis verdict."""
    if direction == "null":
        return "NULL-CONFIRMED" if not significant else "REFUTED"
    if direction == "a_gt_b":
        if significant and diff > 0:
            return "HELD"
        if significant and diff < 0:
            return "REFUTED"
        return "INCONCLUSIVE"
    if direction == "a_ge_b":
        if significant and diff < 0:
            return "REFUTED"
        return "HELD" if diff >= 0 else "INCONCLUSIVE"
    return "INCONCLUSIVE"


def _combine(verdicts: list[str]) -> str:
    if not verdicts:
        return "NO DATA"
    uniq = set(verdicts)
    if len(uniq) == 1:
        return verdicts[0]
    if "HELD" in uniq and ("REFUTED" in uniq or "NULL-CONFIRMED" in uniq):
        return "MIXED"
    if "REFUTED" in uniq:
        return "MIXED"
    # HELD + INCONCLUSIVE etc.
    return "HELD (partial)" if "HELD" in uniq else "INCONCLUSIVE"


def _matched_pairs(runs: list[dict], key: str) -> list[tuple[dict, dict]]:
    """Pairs (a_run, b_run) within a study that differ ONLY in condition[key]."""
    others = [f for f in _MATCH_FIELDS if f != key]
    pairs = []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if all(_cond(a, f) == _cond(b, f) for f in others) \
                    and _cond(a, key) != _cond(b, key):
                pairs.append((a, b))
    return pairs


def _hypothesis_block(runs: list[dict], hyp: dict) -> list[str]:
    """Render one hypothesis: statement, matched-effect table, verdict."""
    hid = hyp.get("id", "H?")
    text = " ".join(str(hyp.get("text", "")).split())
    parts = [f"**{hid}** — {text}"]

    expect = hyp.get("expect")
    if not expect:
        parts.append(f"_Verdict:_ qualitative — see the per-subset table below "
                     f"(`{hyp.get('subset', '?')}`).\n")
        return parts

    key, a, b = expect["key"], expect["a"], expect["b"]
    direction = str(expect.get("direction", "a_gt_b"))
    # optional single-cell subset, e.g. {question_type: cross}. Named `where` (not
    # `on`) because YAML 1.1 parses a bare `on:` key as the boolean True.
    where = expect.get("where")

    # orient each matched pair so the named 'a' value is column A
    rows_out = [
        "| compared (held equal) | cell | n | acc(A=%s) | acc(B=%s) | A−B | 95%% CI | sig | verdict |"
        % (a, b),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    verdicts: list[str] = []
    for ra, rb in _matched_pairs(runs, key):
        if _cond(ra, key) == b:           # orient so ra is the 'a' value
            ra, rb = rb, ra
        # this hypothesis compares exactly a vs b — skip pairs involving any other
        # level of the tested axis (e.g. a third reasoning strategy in the study)
        if _cond(ra, key) != a or _cond(rb, key) != b:
            continue
        rows_a, rows_b = read_rows(ra["path"]), read_rows(rb["path"])
        cell_label = "(all)"
        if where:
            (axis, value), = where.items()
            rows_a = [r for r in rows_a if _row_in_cell(r, axis, value)]
            rows_b = [r for r in rows_b if _row_in_cell(r, axis, value)]
            cell_label = f"{axis}={value}"
        res = paired_diff_test(rows_a, rows_b)
        v = _verdict(res["diff"], res["significant"], direction) if res["n"] else "NO DATA"
        if res["n"]:
            verdicts.append(v)
        held = {f: _cond(ra, f) for f in _MATCH_FIELDS
                if f != key and _cond(ra, f) not in (None, "")}
        held_label = ", ".join(f"{k.split('.')[-1]}={vv}" for k, vv in held.items()) or "—"
        acc_a = sum(bool(r.get("correct")) for r in rows_a) / len(rows_a) if rows_a else 0.0
        acc_b = sum(bool(r.get("correct")) for r in rows_b) / len(rows_b) if rows_b else 0.0
        ci = res["ci"]
        rows_out.append(
            f"| {held_label} | {cell_label} | {res['n']} | {acc_a:.3f} | {acc_b:.3f} "
            f"| {res['diff']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] "
            f"| {'**yes**' if res['significant'] else 'no'} | {v} |")

    overall = _combine(verdicts)
    pred = {"a_gt_b": f"{a} > {b}", "a_ge_b": f"{a} ≥ {b}",
            "null": f"{a} ≈ {b} (predicted null)"}.get(direction, direction)
    parts.append(f"_Prediction:_ {pred}.  _Verdict:_ **{overall}**.\n")
    if len(rows_out) > 2:
        parts += rows_out + [""]
    else:
        parts.append("_(no matched run pair found for this comparison yet.)_\n")
    return parts


def _study_subset_table(runs: list[dict], axis: str) -> list[str]:
    """Per-condition accuracy on the discriminating subset (the 'where')."""
    from .report import _name

    cells: list[str] = []
    per_run: dict[str, dict] = {}
    for s in sorted(runs, key=_name):
        sub = _subset_accuracy(read_rows(s["path"]), axis)
        per_run[_name(s)] = sub
        for c in sub:
            if c not in cells:
                cells.append(c)
    cells = sorted(cells)
    if not cells:
        return []
    header = ["condition (acc · n)"] + cells
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for name, sub in per_run.items():
        row = [name]
        for c in cells:
            if c in sub:
                acc, n = sub[c]
                row.append(f"{acc:.3f} ({n})")
            else:
                row.append("–")
        out.append("| " + " | ".join(row) + " |")
    return [f"Per-subset accuracy by `{axis}` (the discriminating subset):", "",
            *out, ""]


def rq_sections(summaries: list[dict], suite_meta: dict) -> str:
    """Assemble the per-RQ verdict sections, grouped by research question."""
    if not suite_meta:
        return ""
    # group studies by RQ id; a study maps to summaries in its results subdir
    by_rq: dict[str, list[tuple[str, dict]]] = {}
    for study, meta in suite_meta.items():
        by_rq.setdefault(meta.get("rq", "other"), []).append((study, meta))

    parts = ["# Per-RQ analysis (hypothesis verdicts)\n",
             "_Each research question: the directional hypothesis, its effect vs the "
             "declared control with a paired bootstrap test, and a verdict. Predicted "
             "nulls are findings. See docs/research_questions.md §4._\n"]

    for rq in sorted(by_rq):
        if rq == "crosscutting":
            continue
        parts.append(f"## {rq}\n")
        for study, meta in by_rq[rq]:
            runs = [s for s in summaries if _substudy(s) == study]
            parts.append(f"### {study}"
                         + (f"  ({len(runs)} runs)" if runs else "  _(no results yet)_")
                         + "\n")
            ctrl = meta.get("control", {})
            if ctrl:
                parts.append(f"_Control: {ctrl}._\n")
            if not runs:
                # still surface the hypotheses so the framing is visible pre-run
                for h in meta.get("hypotheses", []):
                    parts.append(f"**{h.get('id')}** — "
                                 + " ".join(str(h.get("text", "")).split()) + "\n")
                continue
            for h in meta.get("hypotheses", []):
                parts += _hypothesis_block(runs, h)
            axis = meta.get("subset")
            if axis:
                parts += _study_subset_table(runs, axis)
    return "\n".join(parts)

"""Per-RQ hypothesis-verdict reporting (docs/pivot.md Step 3, RQ spec §4)."""

from mpvrdu.analysis.report_rq import (_combine, _verdict, doc_length_bin,
                                       rq_sections)
from mpvrdu.analysis.report import build_report
from mpvrdu.analysis.aggregate import aggregate_dir
from mpvrdu.config import dict_to_config
from mpvrdu.pipeline import run


def test_verdict_mapping():
    # a_gt_b
    assert _verdict(+0.2, True, "a_gt_b") == "HELD"
    assert _verdict(-0.2, True, "a_gt_b") == "REFUTED"
    assert _verdict(+0.2, False, "a_gt_b") == "INCONCLUSIVE"
    # a_ge_b
    assert _verdict(+0.1, False, "a_ge_b") == "HELD"
    assert _verdict(-0.1, True, "a_ge_b") == "REFUTED"
    # predicted null is a first-class finding
    assert _verdict(0.0, False, "null") == "NULL-CONFIRMED"
    assert _verdict(0.3, True, "null") == "REFUTED"


def test_combine_verdicts():
    assert _combine(["HELD", "HELD"]) == "HELD"
    assert _combine(["HELD", "REFUTED"]) == "MIXED"
    assert _combine(["NULL-CONFIRMED", "NULL-CONFIRMED"]) == "NULL-CONFIRMED"
    assert _combine([]) == "NO DATA"


def test_doc_length_bin():
    assert doc_length_bin(5) == "short(≤10)"
    assert doc_length_bin(20) == "medium(11-30)"
    assert doc_length_bin(99) == "long(>30)"
    assert doc_length_bin(None) == "unknown"


def _run_into(study_dir, name, reasoning, mode, ds):
    cfg = dict_to_config({
        "name": name,
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "dpi": 72},
        "retrieval": {"method": "oracle", "top_k": 4},
        "generation": {"generator": "mock", "mock_mode": mode, "modality": "text",
                       "reasoning": reasoning},
        "judge": {"type": "rule"},
    })
    out = study_dir / f"{name}__{cfg.hash()}.jsonl"
    run(cfg, dataset=ds, out_path=out)
    return out


def _suite_meta():
    return {"RQ3_reasoning": {
        "rq": "RQ3",
        "subset": "question_type",
        "control": {"kind": "condition", "key": "generation.reasoning", "value": "direct"},
        "hypotheses": [{
            "id": "H3a",
            "text": "CoT helps on cross-page but ~zero on single-hop.",
            "subset": "question_type",
            "expect": {"key": "generation.reasoning", "a": "cot", "b": "direct",
                       "where": {"question_type": "cross"}, "direction": "a_gt_b"},
        }],
    }}


def test_rq_section_renders_hypothesis_and_subset_table(synthetic_ds, tmp_path):
    root = tmp_path / "results"
    study = root / "RQ3_reasoning"
    study.mkdir(parents=True)
    _run_into(study, "direct", "direct", "gold", synthetic_ds)
    _run_into(study, "cot", "cot", "gold", synthetic_ds)

    summaries = aggregate_dir(root, pattern="**/*.jsonl")
    md = rq_sections(summaries, _suite_meta())

    # the RQ framing, the hypothesis, a verdict, and the per-subset breakdown
    assert "## RQ3" in md
    assert "H3a" in md
    assert "Verdict:" in md
    # matched pair found (cot vs direct) -> a comparison row with both columns
    assert "acc(A=cot)" in md
    # the discriminating subset table is populated with the cross-page cell
    assert "question_type" in md and "cross" in md


def test_build_report_includes_rq_section_when_suite_given(synthetic_ds, tmp_path):
    root = tmp_path / "results"
    study = root / "RQ3_reasoning"
    study.mkdir(parents=True)
    _run_into(study, "direct", "direct", "gold", synthetic_ds)
    _run_into(study, "cot", "cot", "gold", synthetic_ds)
    suite = tmp_path / "suite.yaml"
    import yaml
    yaml.safe_dump({"defaults": {}, "substudies": _suite_meta()},
                   suite.open("w"))

    with_suite = build_report(root, suite=suite)
    without = build_report(root)
    assert "Per-RQ analysis" in with_suite
    assert "Per-RQ analysis" not in without      # opt-in only

"""Bootstrap CIs, paired significance test, and richer summary fields."""

from mpvrdu.analysis import (aggregate_dir, bootstrap_ci, paired_diff_test,
                             summarize_run)
from mpvrdu.config import dict_to_config
from mpvrdu.pipeline import run


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50, seed=0)
    assert lo <= 0.5 <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_all_correct_is_tight():
    lo, hi = bootstrap_ci([1.0] * 30)
    assert lo == 1.0 and hi == 1.0


def test_paired_diff_significant_when_a_dominates():
    rows_a = [{"qid": str(i), "correct": True} for i in range(40)]
    rows_b = [{"qid": str(i), "correct": False} for i in range(40)]
    res = paired_diff_test(rows_a, rows_b)
    assert res["n"] == 40
    assert res["diff"] == 1.0
    assert res["significant"]


def test_paired_diff_not_significant_when_equal():
    rows = [{"qid": str(i), "correct": i % 2 == 0} for i in range(40)]
    res = paired_diff_test(rows, list(rows))
    assert res["diff"] == 0.0
    assert not res["significant"]


def test_summary_has_ci_cost_and_abstention(synthetic_ds, chdir_tmp):
    cfg = dict_to_config({
        "name": "rich", "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "dpi": 72},
        "retrieval": {"method": "oracle", "top_k": 4},
        "generation": {"generator": "mock", "mock_mode": "gold", "modality": "both"},
    })
    run(cfg, dataset=synthetic_ds, out_path="out.jsonl")
    s = summarize_run("out.jsonl")
    assert len(s["accuracy_ci"]) == 2 and s["accuracy_ci"][0] <= s["accuracy"] <= s["accuracy_ci"][1]
    assert "mean_input_tokens" in s["cost"]            # usage logged
    assert 0.0 <= s["hallucination_rate"] <= 1.0
    # mock(gold) abstains correctly on unanswerable -> no hallucination
    assert s["hallucination_rate"] == 0.0

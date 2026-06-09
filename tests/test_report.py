"""Stage 7 analysis: oracle-gap, top-k series, recall correlation, full report."""

from mpvrdu.analysis import (build_report, oracle_gap, pairwise_significance,
                             recall_accuracy_correlation, sanity_checks,
                             seed_variance, topk_series)
from mpvrdu.config import dict_to_config
from mpvrdu.pipeline import run


def _s(method, modality, k, acc, recall=0.5):
    return {"condition": {"retrieval.method": method, "generation.modality": modality,
                          "retrieval.top_k": k},
            "accuracy": acc, "mean_recall_at_k": recall, "name": f"{method}-{k}"}


def test_oracle_gap_lift_and_ceiling():
    summaries = [
        _s("none", "image", 10, 0.20),
        _s("oracle", "image", 8, 0.80),
        _s("bm25", "image", 4, 0.50),
    ]
    rows = oracle_gap(summaries)
    assert len(rows) == 1
    r = rows[0]
    assert r["method"] == "bm25"
    assert abs(r["lift_over_floor"] - 0.30) < 1e-9      # 0.50 - 0.20
    assert abs(r["gap_to_ceiling"] - 0.30) < 1e-9       # 0.80 - 0.50
    assert abs(r["pct_of_ceiling"] - 0.625) < 1e-9      # 0.50 / 0.80


def test_topk_series_grouping_sorted():
    summaries = [_s("bm25", "image", 4, 0.5), _s("bm25", "image", 1, 0.3),
                 _s("bm25", "image", 2, 0.4), _s("dense", "text", 4, 0.6)]
    series = topk_series(summaries, "accuracy")
    assert series["bm25/image"] == [(1, 0.3), (2, 0.4), (4, 0.5)]
    assert series["dense/text"] == [(4, 0.6)]


def test_recall_accuracy_correlation_positive():
    summaries = [_s("bm25", "image", k, acc, recall=rec) for k, acc, rec in
                 [(1, 0.2, 0.2), (2, 0.4, 0.4), (4, 0.6, 0.6), (8, 0.8, 0.8)]]
    r = recall_accuracy_correlation(summaries)
    assert r is not None and r > 0.9                    # perfectly correlated here


def test_sanity_flags_oracle_below_floor():
    summaries = [_s("none", "image", 10, 0.50), _s("oracle", "image", 8, 0.30)]
    warns = sanity_checks(summaries)
    assert any("oracle" in w and "no-retrieval" in w for w in warns)


def test_sanity_flags_broken_retriever():
    summaries = [_s("bm25", "image", 4, 0.5, recall=0.0)]
    warns = sanity_checks(summaries)
    assert any("recall@k" in w for w in warns)


def test_seed_variance_groups_by_name():
    summaries = [
        {"name": "A__bm25", "accuracy": 0.4, "condition": {}, "mean_recall_at_k": 0.5},
        {"name": "A__bm25", "accuracy": 0.6, "condition": {}, "mean_recall_at_k": 0.5},
        {"name": "A__dense", "accuracy": 0.5, "condition": {}, "mean_recall_at_k": 0.5},
    ]
    sv = seed_variance(summaries)
    assert len(sv) == 1                             # only bm25 has >1 seed
    assert sv[0]["name"] == "A__bm25"
    assert abs(sv[0]["mean_accuracy"] - 0.5) < 1e-9
    assert sv[0]["std_accuracy"] > 0


def test_sanity_clean_when_ok():
    summaries = [_s("none", "image", 10, 0.2), _s("oracle", "image", 8, 0.8),
                 _s("bm25", "image", 4, 0.5, recall=0.6)]
    assert sanity_checks(summaries) == []


def _write_run(path, method, correct_flags):
    from mpvrdu.config import dict_to_config
    from mpvrdu.results import ResultsWriter
    cfg = dict_to_config({"name": f"A__{method}",
                          "retrieval": {"method": method, "top_k": 4},
                          "generation": {"modality": "image", "generator": "mock"}})
    with ResultsWriter(path, config=cfg) as w:
        for i, c in enumerate(correct_flags):
            w.write({"qid": str(i), "correct": c, "gold_answerable": True,
                     "pred_abstained": False, "recall_at_k": 0.5,
                     "question_type": "single", "evidence_sources": ["Text"]})


def test_pairwise_significance(chdir_tmp):
    from mpvrdu.analysis import aggregate_dir
    _write_run("results/grid/A/bm25.jsonl", "bm25", [True] * 8)
    _write_run("results/grid/A/dense.jsonl", "dense", [False] * 8)
    summaries = aggregate_dir("results/grid", pattern="**/*.jsonl")
    sig = pairwise_significance(summaries)
    assert len(sig) == 1
    assert abs(abs(sig[0]["diff"]) - 1.0) < 1e-9
    assert sig[0]["significant"]


def test_build_report_end_to_end(synthetic_ds, chdir_tmp):
    base = {"data": {"name": "synthetic"},
            "representation": {"parser": "pymupdf", "dpi": 72},
            "generation": {"generator": "mock", "mock_mode": "gold", "modality": "image"}}
    for sub, method in [("baselines", "none"), ("baselines", "oracle"),
                        ("A_retrieval", "bm25")]:
        cfg = dict_to_config({**base, "name": f"{sub}-{method}",
                              "retrieval": {"method": method, "top_k": 4}})
        run(cfg, dataset=synthetic_ds, out_path=f"results/grid/{sub}/{method}.jsonl")

    md = build_report("results/grid", fig_dir="results/grid/figures")
    assert "# MP-VRDU results" in md
    assert "All conditions" in md
    assert "Oracle-gap decomposition" in md             # baselines present
    # comprehensive sections
    assert "Accuracy by question type" in md
    assert "Abstention behaviour" in md
    assert "Accuracy by evidence source" in md
    assert "Conditions by sub-study" in md              # grouped subdirs
    assert "### baselines" in md and "### A_retrieval" in md


# --------------------------------------------------------------------------- #
# comprehensive-report section builders
# --------------------------------------------------------------------------- #
def _full_summary(name, **over):
    s = {"name": name, "path": f"results/grid/X/{name}.jsonl",
         "accuracy": 0.5, "accuracy_ci": [0.3, 0.7], "f1": 0.8,
         "mean_recall_at_k": 0.6, "mean_n_selected": 4.0, "n": 10,
         "hallucination_rate": 0.2, "over_abstention_rate": 0.1,
         "by_question_type": {"single": {"n": 6, "correct": 4, "accuracy": 0.667},
                              "cross": {"n": 4, "correct": 1, "accuracy": 0.25}},
         "by_evidence_source": {"Table": {"n": 5, "correct": 3, "accuracy": 0.6}},
         "cost": {"mean_input_tokens": 1000, "mean_output_tokens": 20,
                  "mean_gen_seconds": 1.5, "total_gen_seconds": 15.0},
         "condition": {"retrieval.method": "bm25", "retrieval.top_k": 4,
                       "retrieval.k_strategy": "fixed", "retrieval.rerank": "none",
                       "retrieval.expand": "none"}}
    s.update(over)
    return s


def test_question_type_table_has_columns():
    from mpvrdu.analysis import question_type_table
    md = question_type_table([_full_summary("r1")])
    assert "single acc (n)" in md and "cross acc (n)" in md
    assert "0.667 (6)" in md and "0.250 (4)" in md
    assert "unanswerable" in md                          # column present even if absent in data


def test_abstention_table():
    from mpvrdu.analysis import abstention_table
    md = abstention_table([_full_summary("r1")])
    assert "hallucination" in md and "0.200" in md and "0.100" in md


def test_evidence_source_table():
    from mpvrdu.analysis import evidence_source_table
    md, srcs = evidence_source_table([_full_summary("r1")])
    assert srcs == ["Table"]
    assert "0.600 (5)" in md


def test_cost_table_and_none_when_absent():
    from mpvrdu.analysis import cost_table
    assert "mean in-tok" in cost_table([_full_summary("r1")])
    assert cost_table([_full_summary("r2", cost={})]) is None


def test_tier1_table_only_lists_active_toggles():
    from mpvrdu.analysis import tier1_table
    plain = _full_summary("plain")
    assert tier1_table([plain]) is None                  # all defaults -> nothing
    adaptive = _full_summary("adapt")
    adaptive["condition"]["retrieval.k_strategy"] = "gmm"
    adaptive["mean_n_selected"] = 2.3
    md = tier1_table([plain, adaptive])
    assert md is not None
    assert "adapt" in md and "plain" not in md           # only the active one
    assert "gmm" in md and "2.30" in md

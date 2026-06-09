"""scripts/inspect_run.py — run-output inspection (selected vs gold pages)."""

import argparse
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inspect_run", Path(__file__).resolve().parent.parent / "scripts" / "inspect_run.py")
inspect_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inspect_run)


def _args(**over):
    base = dict(errors_only=False, retrieval_misses=False, qid=None, type=None,
                limit=None, full=False)
    base.update(over)
    return argparse.Namespace(**base)


def _row(qid, correct, gold_pages, selected_0b, answerable=True, qtype="single"):
    return {"qid": qid, "correct": correct, "gold_answerable": answerable,
            "question_type": qtype, "evidence_pages": gold_pages,
            "selected_pages_0based": selected_0b, "recall_at_k": 1.0,
            "n_selected": len(selected_0b), "pred_abstained": False}


def test_page_conversion_helpers():
    r = _row("q1", True, [2, 3], [1, 2])      # gold 1-based, selected 0-based
    assert inspect_run._gold_pages_1based(r) == {2, 3}
    assert inspect_run._selected_1based(r) == [2, 3]   # 0-based [1,2] -> 1-based


def test_errors_only_filter():
    good = _row("q1", True, [1], [0])
    bad = _row("q2", False, [1], [0])
    a = _args(errors_only=True)
    assert inspect_run._keep(bad, a) and not inspect_run._keep(good, a)


def test_retrieval_misses_filter():
    # gold page 3 (1-based) selected -> not a miss; gold page 5 not selected -> miss
    hit = _row("q1", False, [3], [2])         # selected 0-based 2 == 1-based 3
    miss = _row("q2", False, [5], [0, 1])     # gold 5 not in selected {1,2}
    unans = _row("q3", False, [], [0], answerable=False)
    a = _args(retrieval_misses=True)
    assert inspect_run._keep(miss, a)
    assert not inspect_run._keep(hit, a)
    assert not inspect_run._keep(unans, a)    # unanswerable excluded


def test_qid_and_type_filters():
    r = _row("q1", True, [1], [0], qtype="cross")
    assert inspect_run._keep(r, _args(qid="q1"))
    assert not inspect_run._keep(r, _args(qid="other"))
    assert inspect_run._keep(r, _args(type="cross"))
    assert not inspect_run._keep(r, _args(type="single"))


def test_load_reads_meta_and_rows(tmp_path):
    from mpvrdu.config import dict_to_config
    from mpvrdu.results import ResultsWriter

    cfg = dict_to_config({"name": "r", "retrieval": {"method": "bm25"},
                          "generation": {"generator": "mock", "modality": "image"}})
    path = tmp_path / "r.jsonl"
    with ResultsWriter(path, config=cfg) as w:
        w.write(_row("q1", True, [1], [0]))
        w.write(_row("q2", False, [2], [9]))
    meta, rows = inspect_run._load(path)
    assert meta is not None and meta.get("config", {}).get("name") == "r"
    assert len(rows) == 2

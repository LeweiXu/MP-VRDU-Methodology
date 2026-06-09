"""Stage 4 — retrievers, the retriever->selector adapter, hybrid, recall eval."""

import pytest

from mpvrdu.config import dict_to_config
from mpvrdu.retrieve import build_selector, build_units, evaluate_retrieval
from mpvrdu.retrieve.retrievers import BM25Retriever, TFIDFRetriever, tokenize


def _cfg(method, **retr):
    r = {"method": method, "top_k": 4, **retr}
    return dict_to_config({
        "name": method,
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "chunking": "page", "dpi": 72},
        "retrieval": r,
        "generation": {"generator": "mock", "mock_mode": "gold", "modality": "image"},
    })


def test_tokenize():
    assert tokenize("Hello, World! 42%") == ["hello", "world", "42"]


def test_build_units_text(synthetic_ds):
    doc = synthetic_ds.get_document("beta.pdf")
    units = build_units(doc, "text", "pymupdf", dpi=72, chunking="page")
    assert len(units) == 4                       # 4 pages
    assert all(u.text and u.image_path is None for u in units)
    assert [u.page_index for u in units] == [0, 1, 2, 3]


def test_build_units_visual_renders_images(synthetic_ds, chdir_tmp):
    # visual modality ignores the parser and produces image units
    doc = synthetic_ds.get_document("beta.pdf")
    units = build_units(doc, "visual", "pymupdf", dpi=72)
    assert len(units) == 4
    assert all(u.image_path and u.text is None for u in units)


def test_bm25_finds_evidence_page(synthetic_ds):
    doc = synthetic_ds.get_document("beta.pdf")
    units = build_units(doc, "text", "pymupdf", dpi=72)
    r = BM25Retriever()
    r.index(units, doc_id="beta.pdf")
    # "accuracy" lives on page index 2 (Results page)
    top = r.retrieve("What accuracy was reached?", k=1)
    assert top[0][0] == 2


def test_grep_floor_finds_evidence_page(synthetic_ds):
    from mpvrdu.retrieve.retrievers import GrepRetriever
    doc = synthetic_ds.get_document("beta.pdf")
    units = build_units(doc, "text", "pymupdf", dpi=72)
    r = GrepRetriever()
    r.index(units, doc_id="beta.pdf")
    top = r.retrieve("accuracy reached", k=1)
    assert top[0][0] == 2                          # Results page has "accuracy"


def test_tfidf_finds_evidence_page(synthetic_ds):
    doc = synthetic_ds.get_document("beta.pdf")
    units = build_units(doc, "text", "pymupdf", dpi=72)
    r = TFIDFRetriever()
    r.index(units, doc_id="beta.pdf")
    top = r.retrieve("how many layers were used", k=1)
    assert top[0][0] == 1                          # Methods page (7 layers)


def test_bm25_selector_via_factory(synthetic_ds):
    sel = build_selector(_cfg("bm25"))
    q = next(q for q in synthetic_ds.questions if q.qid == "s2")
    doc = synthetic_ds.get_document(q.doc_id)
    selection = sel.select(q, doc)
    assert 2 in selection.page_indices            # evidence page (0-based) present
    assert len(selection.page_indices) <= 4


def test_recall_eval_bm25(synthetic_ds):
    sel = build_selector(_cfg("bm25"))
    res = evaluate_retrieval(sel, synthetic_ds, ks=[1, 2, 4, 8])
    table = res["recall_at_k"]
    # monotonic non-decreasing in k
    vals = [table[k] for k in (1, 2, 4, 8)]
    assert vals == sorted(vals)
    assert table[8] == 1.0                         # k >= pages-per-doc -> all found
    assert table[1] > 0.0
    assert res["n_answerable"] == 5                # 3 single + 2 cross


def test_hybrid_runs(synthetic_ds):
    sel = build_selector(_cfg("hybrid", hybrid_methods=["bm25", "tfidf"]))
    q = next(q for q in synthetic_ds.questions if q.qid == "s2")
    doc = synthetic_ds.get_document(q.doc_id)
    selection = sel.select(q, doc)
    assert selection.meta["components"] == ["bm25", "tfidf"]
    assert 2 in selection.page_indices


def test_adjacent_expansion_adds_neighbours(synthetic_ds):
    # top_k=1 isolates the evidence page (0-based 2), then ±1 expansion adds 1,3
    sel = build_selector(_cfg("bm25", top_k=1, expand="adjacent", expand_window=1))
    q = next(q for q in synthetic_ds.questions if q.qid == "s2")
    doc = synthetic_ds.get_document(q.doc_id)
    selection = sel.select(q, doc)
    assert selection.page_indices[0] == 2              # the hit stays ranked first
    assert {1, 3} <= set(selection.page_indices)       # neighbours brought along
    assert selection.meta["expand"] == "adjacent"
    # full candidate ranking is preserved for recall sweeps
    assert selection.meta["ranked_pages"] == [2]


def test_adaptive_k_runs_and_records_cut(synthetic_ds):
    pytest.importorskip("sklearn")
    sel = build_selector(_cfg("bm25", top_k=4, k_strategy="kmeans"))
    q = next(q for q in synthetic_ds.questions if q.qid == "s2")
    doc = synthetic_ds.get_document(q.doc_id)
    selection = sel.select(q, doc)
    assert 1 <= selection.meta["n_kept"] <= len(selection.meta["ranked_pages"])
    assert len(selection.page_indices) == selection.meta["n_kept"]


def test_recall_eval_uses_candidate_ranking_under_expansion(synthetic_ds):
    # expansion reshapes the final pages, but the recall sweep must still see the
    # raw ranking (via meta ranked_pages) so recall@k stays monotone.
    sel = build_selector(_cfg("bm25", top_k=1, expand="adjacent"))
    res = evaluate_retrieval(sel, synthetic_ds, ks=[1, 2, 4, 8])
    vals = [res["recall_at_k"][k] for k in (1, 2, 4, 8)]
    assert vals == sorted(vals)


def test_dense_retriever_optional(synthetic_ds):
    st = pytest.importorskip("sentence_transformers")  # skip if not installed
    from mpvrdu.retrieve.retrievers import DenseRetriever

    doc = synthetic_ds.get_document("beta.pdf")
    units = build_units(doc, "text", "pymupdf", dpi=72)
    r = DenseRetriever()
    r.index(units, doc_id="beta.pdf")
    top = r.retrieve("What accuracy was reached?", k=2)
    assert any(uid == 2 for uid, _ in top)

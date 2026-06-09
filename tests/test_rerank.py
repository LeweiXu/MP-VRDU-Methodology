"""Tier-1 #3 — retrieve -> rerank -> read. Offline via an injected complete_fn."""

from mpvrdu.config import dict_to_config
from mpvrdu.retrieve import build_selector
from mpvrdu.retrieve.rerank import LLMReranker


def _accuracy_scorer(prompt: str) -> str:
    """Fake LLM: a page is relevant iff its CONTENT mentions accuracy.

    Keys on the page-content portion only, so the question text (which may also
    mention accuracy) doesn't make every page look relevant.
    """
    page = prompt.split("Page content:")[-1]
    return "9" if "accuracy" in page.lower() else "1"


def test_reranker_reorders_by_relevance():
    rr = LLMReranker(complete_fn=_accuracy_scorer)
    cands = [(0, "intro text"), (1, "methods text"),
             (2, "accuracy reached 88 percent")]
    ranked = rr.rerank("What accuracy?", cands)
    assert ranked[0][0] == 2                       # the accuracy page floats up
    assert [p for p, _ in ranked] == [2, 0, 1] or ranked[0][0] == 2


def test_reranker_parses_garbage_as_zero():
    rr = LLMReranker(complete_fn=lambda p: "no idea")
    ranked = rr.rerank("q", [(0, "a"), (1, "b")])
    assert all(score == 0.0 for _, score in ranked)
    assert [p for p, _ in ranked] == [0, 1]        # stable on ties


def _cfg(method, **retr):
    r = {"method": method, "top_k": 1, **retr}
    return dict_to_config({
        "name": method,
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "chunking": "page", "dpi": 72},
        "retrieval": r,
        "generation": {"generator": "mock", "mock_mode": "gold", "modality": "image"},
    })


def test_rerank_changes_top_selection(synthetic_ds):
    # bm25 alone already finds page 2; here we confirm rerank drives the cut by
    # injecting a reranker and asking for top_k=1.
    cfg = _cfg("bm25")
    sel = build_selector(cfg, reranker=LLMReranker(complete_fn=_accuracy_scorer))
    q = next(q for q in synthetic_ds.questions if q.qid == "s2")
    doc = synthetic_ds.get_document(q.doc_id)
    selection = sel.select(q, doc)
    assert selection.meta.get("reranked") is True
    assert selection.page_indices == [2]           # accuracy page, cut to top_k=1


def test_rerank_hybrid_wires_through(synthetic_ds):
    cfg = _cfg("hybrid", hybrid_methods=["bm25", "tfidf"], top_k=1)
    sel = build_selector(cfg, reranker=LLMReranker(complete_fn=_accuracy_scorer))
    q = next(q for q in synthetic_ds.questions if q.qid == "s2")
    doc = synthetic_ds.get_document(q.doc_id)
    selection = sel.select(q, doc)
    assert selection.meta.get("reranked") is True
    assert selection.page_indices == [2]

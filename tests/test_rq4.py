"""RQ4 coarse-to-fine / evidence-narrowing (docs/pivot.md Step 6, RQ spec RQ4).

The mechanism (retrieve -> rerank -> read) is the existing `rerank: llm` stage.
RQ4 frames it as single-pass vs coarse-to-fine at a MATCHED FINAL BUDGET (the same
number of pages reach the reader, so the contrast is narrowing-vs-not, not more
compute — H4c), and surfaces the extra stage's per-page LLM calls (the cost
co-axis). Tested with an injected reranker so no model is needed.
"""

from mpvrdu.config import dict_to_config
from mpvrdu.retrieve import build_selector
from mpvrdu.retrieve.rerank import LLMReranker


def _cfg(rerank):
    return dict_to_config({
        "representation": {"parser": "pymupdf", "chunking": "page"},
        "retrieval": {"method": "dense", "top_k": 3, "rerank": rerank,
                      "rerank_candidates": 4},
    })


def _select(cfg, doc, q, reranker=None):
    sel = build_selector(cfg, reranker=reranker)
    out = sel.select(q, doc)
    sel.unload()
    return out


def test_matched_final_budget(synthetic_ds):
    """single_pass and retrieve->rerank->read feed the SAME #pages to the reader."""
    doc = synthetic_ds.get_document("beta.pdf")        # 4 pages
    q = next(x for x in synthetic_ds.questions if x.doc_id == "beta.pdf")

    single = _select(_cfg("none"), doc, q)
    # rerank: a frozen scorer (no model) just returns a constant relevance
    rr = LLMReranker(complete_fn=lambda prompt: "5")
    coarse = _select(_cfg("llm"), doc, q, reranker=rr)

    # H4c control: identical final budget (top_k=3) into the reader
    assert len(single.page_indices) == 3
    assert len(coarse.page_indices) == 3


def test_extra_stage_cost_is_logged(synthetic_ds):
    """The rerank stage's per-page LLM calls are recorded (RQ4 cost co-axis)."""
    doc = synthetic_ds.get_document("beta.pdf")
    q = next(x for x in synthetic_ds.questions if x.doc_id == "beta.pdf")

    single = _select(_cfg("none"), doc, q)
    assert "rerank_calls" not in single.meta            # single-pass: no extra stage

    rr = LLMReranker(complete_fn=lambda prompt: "5")
    coarse = _select(_cfg("llm"), doc, q, reranker=rr)
    assert coarse.meta.get("reranked") is True
    # one LLM call per candidate page reranked (the narrowing stage's cost)
    assert coarse.meta.get("rerank_calls", 0) >= 3


def test_rerank_does_not_exceed_budget_but_reranks_more(synthetic_ds):
    """Coarse-to-fine considers a deeper candidate pool, then narrows to top_k."""
    doc = synthetic_ds.get_document("beta.pdf")
    q = next(x for x in synthetic_ds.questions if x.doc_id == "beta.pdf")
    rr = LLMReranker(complete_fn=lambda prompt: "5")
    coarse = _select(_cfg("llm"), doc, q, reranker=rr)
    # narrowed to the budget, but the candidate pool reranked was >= the budget
    assert len(coarse.page_indices) == 3
    assert coarse.meta["candidates_reranked"] >= len(coarse.page_indices)

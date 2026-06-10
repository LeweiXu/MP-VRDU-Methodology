"""RQ1 relation-aware structural traversal (docs/pivot.md Step 4).

The defining test (H1b): on a multi-hop item whose answer page is LOW similarity
to the query but structurally adjacent to a high-similarity anchor, relation-aware
traversal recovers that page while pure-similarity retrieval (BM25) misses it — at
the SAME page budget.
"""

import pytest

from mpvrdu.config import dict_to_config
from mpvrdu.data.dataset import Document, Question
from mpvrdu.retrieve import build_selector
from mpvrdu.retrieve.traverse import TraverseSelector


def _make_doc(tmp_path):
    """4-page PDF. Pages 0+1 share the 'Setup' section; page 1 is the answer page
    (no query keywords). Page 2 is a keyword-rich distractor in another section."""
    import fitz

    pages = [
        ("Setup", "experiment uses widget alpha keyword beta configuration"),  # anchor
        ("Setup", "the result value equals eighty eight here"),                 # answer (low sim)
        ("Other", "widget alpha keyword beta appears again as a distractor"),   # keyword distractor
        ("More",  "unrelated closing remarks and acknowledgements"),
    ]
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for heading, body in pages:
        pg = doc.new_page()
        pg.insert_text((72, 72), heading, fontsize=20)
        pg.insert_text((72, 120), body, fontsize=12)
    doc.save(str(path))
    doc.close()
    return Document(doc_id="doc.pdf", pdf_path=path)


_Q = "Which widget alpha keyword beta configuration was used?"


def _select(method, doc, top_k=2, expand="none"):
    cfg = dict_to_config({
        "representation": {"parser": "pymupdf", "chunking": "page"},
        "retrieval": {"method": method, "top_k": top_k, "expand": expand},
    })
    sel = build_selector(cfg)
    q = Question(qid="q", doc_id=doc.doc_id, question=_Q, answer="x",
                 evidence_pages=[2])   # answer page is 1-based page 2 (0-based 1)
    out = sel.select(q, doc)
    sel.unload()
    return out.page_indices


def test_traverse_recovers_page_similarity_misses(tmp_path):
    doc = _make_doc(tmp_path)
    traverse_pages = _select("traverse", doc, top_k=2)
    bm25_pages = _select("bm25", doc, top_k=2)
    # relation-aware: anchor (page 0) + its section-mate (page 1, the answer)
    assert 1 in traverse_pages
    # pure similarity ranks the keyword distractor (page 2) over the low-sim
    # answer page, so at the same budget it misses page 1
    assert 1 not in bm25_pages
    assert len(traverse_pages) == 2


def test_traverse_respects_fixed_budget(tmp_path):
    doc = _make_doc(tmp_path)
    assert len(_select("traverse", doc, top_k=1)) == 1
    assert len(_select("traverse", doc, top_k=3)) == 3


def test_traverse_expansion_adds_neighbours(tmp_path):
    doc = _make_doc(tmp_path)
    base = set(_select("traverse", doc, top_k=2, expand="none"))
    expanded = set(_select("traverse", doc, top_k=2, expand="adjacent"))
    # expansion only ever ADDS pages on top of the traversal
    assert base <= expanded


def test_traverse_runs_in_pipeline(synthetic_ds, chdir_tmp):
    from mpvrdu.pipeline import run

    cfg = dict_to_config({
        "name": "traverse-smoke",
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "dpi": 72},
        "retrieval": {"method": "traverse", "top_k": 2},
        "generation": {"generator": "mock", "mock_mode": "echo", "modality": "text"},
    })
    m = run(cfg, dataset=synthetic_ds, out_path="out.jsonl")
    assert m["n"] == 7

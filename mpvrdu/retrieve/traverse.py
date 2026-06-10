"""Relation-aware retrieval: structural section/tree traversal (RQ1, H1b).

`docs/research_questions.md` RQ1 splits retrieval into SIMILARITY-based (score
pages independently by query similarity — BM25/dense/ColPali) vs RELATION-AWARE
(traverse explicit document structure). This is the richer relation-aware level:
find a high-similarity ANCHOR page, then NAVIGATE the document's structure
(section membership, then page adjacency) to bring related pages along, within a
FIXED page budget (top_k) for fair comparison.

The mechanism it tests: on multi-hop / cross-page questions whose individual
evidence pages are LOW similarity to the query, a pure-similarity retriever ranks
them below keyword-matching distractors and misses them; traversal recovers them
from a single high-similarity entry point. On single-page questions the anchor is
already the answer, so traversal adds little — the predicted ~zero gain (H1b).

Training-free and deterministic (no model, no GPU): the anchor score is lexical
token overlap, structure comes from the parser headings (shared with RQ2).
"""

from __future__ import annotations

from collections import defaultdict

from ..data.dataset import Document, Question
from ..eval.extract import normalize_text
from ..represent.base import get_parser
from .base import EvidenceSelector, Selection
from .postprocess import build_page_sections, expand_pages


def _overlap_score(query_tokens: set[str], text: str) -> float:
    """Lexical anchor score: fraction of query tokens present on the page."""
    if not query_tokens:
        return 0.0
    page_tokens = set(normalize_text(text).split())
    return len(query_tokens & page_tokens) / len(query_tokens)


class TraverseSelector(EvidenceSelector):
    """Similarity-anchored structural traversal under a fixed page budget."""

    name = "traverse"

    def __init__(self, top_k: int, parser_name: str, dpi: int,
                 chunking: str = "page", expand: str = "none",
                 expand_window: int = 1):
        self.top_k = top_k
        self.parser_name = parser_name
        self.dpi = dpi
        self.chunking = chunking
        self.expand = expand
        self.expand_window = expand_window
        self._doc_id = None
        self._page_text: dict[int, str] = {}
        self._page_sections: dict[int, object] = {}
        self._n_pages = 0

    def _ensure_parsed(self, document: Document) -> None:
        if self._doc_id == document.doc_id:
            return
        parsed = get_parser(self.parser_name).parse_document(document.pdf_path)
        self._page_text = {p.page_index: (p.text or "") for p in parsed}
        self._page_sections = build_page_sections(parsed)
        self._n_pages = document.ensure_pages()
        self._doc_id = document.doc_id

    def select(self, question: Question, document: Document) -> Selection:
        self._ensure_parsed(document)
        qtokens = set(normalize_text(question.question).split())

        # rank pages by anchor similarity (the entry points for traversal)
        anchors = sorted(self._page_text,
                         key=lambda p: (-_overlap_score(qtokens, self._page_text[p]), p))

        sec_to_pages: dict[object, list[int]] = defaultdict(list)
        for pg, sec in self._page_sections.items():
            sec_to_pages[sec].append(pg)

        budget = self.top_k
        pages: list[int] = []
        scores: list[float] = []
        seen: set[int] = set()

        def add(p: int, score: float) -> bool:
            if 0 <= p < self._n_pages and p not in seen and len(pages) < budget:
                seen.add(p)
                pages.append(p)
                scores.append(score)
            return len(pages) >= budget

        for anchor in anchors:
            if len(pages) >= budget:
                break
            anchor_score = _overlap_score(qtokens, self._page_text[anchor])
            if add(anchor, anchor_score):
                break
            # relation-aware step 1: traverse the anchor's SECTION (structural)
            sec = self._page_sections.get(anchor)
            for sp in sorted(sec_to_pages.get(sec, [])):
                if add(sp, anchor_score * 0.9):
                    break
            if len(pages) >= budget:
                break
            # relation-aware step 2: traverse to immediate NEIGHBOURS (adjacency)
            for d in (1, -1):
                if add(anchor + d, anchor_score * 0.8):
                    break

        meta = {"selector": self.name, "ranked_pages": list(pages)}
        # #4/#5 expansion wraps the traversal too (RQ1 relation-aware cheap end),
        # reusing the section structure already built above.
        if self.expand != "none" and pages:
            pages, scores = expand_pages(
                pages, scores, mode=self.expand, window=self.expand_window,
                n_pages=self._n_pages, page_sections=self._page_sections)
            meta["expand"] = self.expand

        return Selection(page_indices=pages, scores=scores, meta=meta)

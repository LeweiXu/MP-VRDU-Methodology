"""Hybrid retrieval via Reciprocal Rank Fusion (Stage 4, built LAST).

Fusion happens at the PAGE level in selector space, not at the unit level. This
makes hybrid work uniformly whether the fused methods share a modality
(sparse+dense, both text) or cross it (text+visual) — each sub-selector builds
its own units and we fuse the resulting ranked page lists. RRF:

    score(page) = sum_over_selectors 1 / (rrf_k + rank_in_that_selector)
"""

from __future__ import annotations

from ..data.dataset import Document, Question
from .base import EvidenceSelector, Selection


class HybridSelector(EvidenceSelector):
    name = "hybrid"

    def __init__(self, selectors: list[EvidenceSelector], top_k: int,
                 rrf_k: int = 60, component_k: int = 20, *,
                 k_strategy: str = "fixed", expand: str = "none",
                 expand_window: int = 1, parser_name: str = "pymupdf4llm",
                 reranker=None):
        if len(selectors) < 2:
            raise ValueError("hybrid needs >= 2 selectors")
        self.selectors = selectors
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.component_k = component_k  # how deep each component contributes
        # Tier-1 post-processing applied once on the fused page list.
        self.k_strategy = k_strategy
        self.expand = expand
        self.expand_window = expand_window
        self.parser_name = parser_name
        self.reranker = reranker
        self._page_text: dict[int, str] = {}
        self._page_sections: dict[int, object] = {}
        self._parsed_doc = None

    def _parsed_pages(self, document: Document):
        from ..represent.base import get_parser
        from .postprocess import build_page_sections

        if self._parsed_doc != document.doc_id:
            parser = get_parser(self.parser_name)
            parsed = parser.parse_document(document.pdf_path)
            self._page_text = {p.page_index: (p.text or "") for p in parsed}
            self._page_sections = build_page_sections(parsed)
            self._parsed_doc = document.doc_id
        return self._page_text, self._page_sections

    def select(self, question: Question, document: Document) -> Selection:
        from .postprocess import apply_k_strategy, expand_pages

        fused: dict[int, float] = {}
        for sel in self.selectors:
            pages = sel.select(question, document).page_indices[: self.component_k]
            for rank, page in enumerate(pages):
                fused[page] = fused.get(page, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        pages = [p for p, _ in ranked]
        scores = [s for _, s in ranked]
        meta: dict = {"selector": self.name,
                      "components": [s.name for s in self.selectors],
                      "ranked_pages": list(pages)}

        if self.reranker is not None and pages:
            page_text, _ = self._parsed_pages(document)
            reranked = self.reranker.rerank(
                question.question, [(p, page_text.get(p, "")) for p in pages])
            pages = [p for p, _ in reranked]
            scores = [float(s) for _, s in reranked]
            meta["reranked"] = True

        keep = apply_k_strategy(scores, self.k_strategy, self.top_k)
        pages, scores = pages[:keep], scores[:keep]

        if self.expand != "none" and pages:
            psec = None
            if self.expand == "parent_section":
                _, psec = self._parsed_pages(document)
            pages, scores = expand_pages(
                pages, scores, mode=self.expand, window=self.expand_window,
                n_pages=document.ensure_pages(), page_sections=psec)
            meta["expand"] = self.expand

        return Selection(page_indices=pages, scores=scores, meta=meta)

    def unload(self) -> None:
        for sel in self.selectors:
            sel.unload()
        if self.reranker is not None:
            self.reranker.unload()

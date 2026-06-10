"""Evidence-selector + retriever interfaces, baselines, and unit building.

Two layers:
- `EvidenceSelector` (select pages for a question) — the seam the pipeline uses.
  NoRetrieval (floor) and Oracle (ceiling) are trivial selectors.
- `Retriever` (index units, retrieve top-k) — the Stage-4 retrieval methods.
  `RetrieverSelector` adapts a Retriever into an EvidenceSelector: it builds the
  per-document units, indexes them (cached per doc), retrieves top-k, and maps
  retrieved units back to 0-based PAGE indices so recall stays page-based even
  when chunking is sub-page.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import RunConfig
from ..data.dataset import Document, Question
from ..data.render import render_page
from ..represent.base import get_parser
from ..represent.chunking import chunk_pages


# --------------------------------------------------------------------------- #
# Units + selection
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    unit_id: int                 # sequential id within a document
    page_index: int              # 0-based source page (for recall)
    text: Optional[str] = None
    image_path: Optional[str] = None


@dataclass
class Selection:
    page_indices: list[int] = field(default_factory=list)   # 0-based, ranked, deduped
    scores: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def build_units(document: Document, modality: str, parser_name: str,
                dpi: int, chunking: str = "page") -> list[Unit]:
    """Build indexable units for one document.

    modality="visual" -> one image unit per page (parser is irrelevant).
    modality="text"    -> chunked text units via the parser + chunking strategy.
    """
    n = document.ensure_pages()
    if modality == "visual":
        units = []
        for p in range(n):
            rp = render_page(document.pdf_path, p, dpi=dpi, doc_id=document.doc_id)
            units.append(Unit(unit_id=p, page_index=p, image_path=str(rp.path)))
        return units

    parser = get_parser(parser_name)
    parsed = parser.parse_document(document.pdf_path)
    chunks = chunk_pages(parsed, chunking)
    return [Unit(unit_id=i, page_index=c.page_index, text=c.text)
            for i, c in enumerate(chunks)]


# --------------------------------------------------------------------------- #
# Selector interface + baselines
# --------------------------------------------------------------------------- #
class EvidenceSelector(ABC):
    name: str = "base"

    @abstractmethod
    def select(self, question: Question, document: Document) -> Selection:
        ...

    def unload(self) -> None:
        """Release any GPU model held by this selector. Default: nothing."""
        return None


class NoRetrieval(EvidenceSelector):
    """No-retrieval lower bound: feed the first N pages (context.md §7)."""

    name = "none"

    def __init__(self, n_pages: int = 10):
        self.n_pages = n_pages

    def select(self, question: Question, document: Document) -> Selection:
        total = document.ensure_pages()
        pages = list(range(min(self.n_pages, total)))
        return Selection(page_indices=pages, scores=[0.0] * len(pages),
                         meta={"selector": self.name, "n_pages": self.n_pages})


class Oracle(EvidenceSelector):
    """Oracle upper bound: feed exactly the gold evidence pages (context.md §7).

    Unanswerable questions whose annotation lists no pages get NO evidence; those
    that DO list a page (the page checked and found lacking) get it — either way
    the generator must abstain, which the scorer enforces. Pages outside the doc
    range (e.g. the rare evidence_pages=[0]) are clamped out.
    """

    name = "oracle"

    def select(self, question: Question, document: Document) -> Selection:
        total = document.ensure_pages()
        pages = [p for p in question.evidence_pages_zero_based if 0 <= p < total]
        return Selection(page_indices=pages, scores=[1.0] * len(pages),
                         meta={"selector": self.name,
                               "unanswerable": question.is_unanswerable})


# --------------------------------------------------------------------------- #
# Retriever interface + adapter
# --------------------------------------------------------------------------- #
class Retriever(ABC):
    name: str = "base"
    modality: str = "text"       # "text" | "visual"

    @abstractmethod
    def index(self, units: list[Unit], doc_id: Optional[str] = None) -> None:
        """Build the index for one document's units (current active index)."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k (unit_id, score) pairs, best first."""

    def unload(self) -> None:
        """Release any GPU model + cached embeddings. Default: nothing."""
        return None


class RetrieverSelector(EvidenceSelector):
    """Adapt a Retriever into an EvidenceSelector with per-document indexing.

    Beyond the fixed top_k cut, this is the seam for the Tier-1 post-processing
    that wraps any retriever (docs/corpus_techniques.md): an optional LLM
    ``reranker`` (#3) re-orders the candidate pages, ``k_strategy`` (#1) chooses
    the cut adaptively from the score distribution, and ``expand`` (#4/#5) brings
    parent/neighbour pages along. With the defaults (no reranker, fixed,
    none) the behaviour is identical to the original page-collapse.
    """

    def __init__(self, retriever: Retriever, top_k: int, parser_name: str,
                 dpi: int, chunking: str = "page", *,
                 k_strategy: str = "fixed", candidate_k: int = 0,
                 reranker: Optional["object"] = None, rerank_candidates: int = 0,
                 expand: str = "none", expand_window: int = 1):
        self.retriever = retriever
        self.top_k = top_k
        self.parser_name = parser_name
        self.dpi = dpi
        self.chunking = chunking
        self.k_strategy = k_strategy
        self.candidate_k = candidate_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.expand = expand
        self.expand_window = expand_window
        self.name = retriever.name
        self._indexed_doc: Optional[str] = None
        self._units: list[Unit] = []
        self._page_text: dict[int, str] = {}      # cached per indexed doc
        self._page_sections: dict[int, object] = {}

    def unload(self) -> None:
        self.retriever.unload()
        if self.reranker is not None:
            self.reranker.unload()
        self._indexed_doc = None
        self._units = []
        self._page_text = {}
        self._page_sections = {}

    def _ensure_indexed(self, document: Document) -> None:
        if self._indexed_doc == document.doc_id:
            return
        self._units = build_units(document, self.retriever.modality,
                                  self.parser_name, self.dpi, self.chunking)
        self.retriever.index(self._units, doc_id=document.doc_id)
        self._indexed_doc = document.doc_id
        self._page_text = {}
        self._page_sections = {}

    def _candidate_depth(self) -> int:
        """How many distinct pages to surface before rerank/cut/expand."""
        depth = self.top_k
        auto = max(self.top_k * 4, 8)
        if self.k_strategy != "fixed":
            depth = max(depth, self.candidate_k or auto)
        if self.reranker is not None:
            depth = max(depth, self.rerank_candidates or auto)
        return depth

    def _parsed_pages(self, document: Document):
        # Rerank and parent_section need page TEXT/structure even for visual
        # retrievers (which index images) — re-invoke the parser once per doc.
        from ..represent.base import get_parser

        if not self._page_text and not self._page_sections:
            parser = get_parser(self.parser_name)
            parsed = parser.parse_document(document.pdf_path)
            self._page_text = {p.page_index: (p.text or "") for p in parsed}
            from .postprocess import build_page_sections
            self._page_sections = build_page_sections(parsed)
        return self._page_text, self._page_sections

    def select(self, question: Question, document: Document) -> Selection:
        from .postprocess import apply_k_strategy, expand_pages

        self._ensure_indexed(document)
        cand_pages = self._candidate_depth()
        # retrieve extra units so that, after collapsing chunks to pages, we can
        # still surface `cand_pages` distinct pages.
        unit_k = min(len(self._units), max(cand_pages * 4, cand_pages))
        ranked = self.retriever.retrieve(question.question, k=unit_k)
        id_to_page = {u.unit_id: u.page_index for u in self._units}

        pages: list[int] = []
        scores: list[float] = []
        for uid, score in ranked:
            page = id_to_page.get(uid)
            if page is None or page in pages:
                continue
            pages.append(page)
            scores.append(float(score))
            if len(pages) >= cand_pages:
                break

        meta: dict[str, Any] = {"selector": self.name, "unit_k": unit_k,
                                # full candidate ranking (for recall sweeps)
                                "ranked_pages": list(pages)}

        # #3 rerank the candidate pages over their text, then cut keeps top_k.
        # This is RQ4's retrieve->rerank->read coarse-to-fine stage; the candidate
        # pages reranked here are the extra stage's per-page LLM calls (logged so
        # the cost of narrowing is visible — RQ spec RQ4 cost co-axis).
        if self.reranker is not None and pages:
            page_text, _ = self._parsed_pages(document)
            candidates = [(p, page_text.get(p, "")) for p in pages]
            reranked = self.reranker.rerank(question.question, candidates)
            pages = [p for p, _ in reranked]
            scores = [float(s) for _, s in reranked]
            meta["reranked"] = True
            meta["rerank_calls"] = len(candidates)   # extra-stage LLM calls
            meta["candidates_reranked"] = len(candidates)

        # #1 adaptive vs fixed cut.
        keep = apply_k_strategy(scores, self.k_strategy, self.top_k)
        pages, scores = pages[:keep], scores[:keep]
        meta["n_kept"] = keep

        # #4/#5 expansion (only ever adds pages).
        if self.expand != "none" and pages:
            psec = None
            if self.expand == "parent_section":
                _, psec = self._parsed_pages(document)
            pages, scores = expand_pages(
                pages, scores, mode=self.expand, window=self.expand_window,
                n_pages=document.ensure_pages(), page_sections=psec)
            meta["expand"] = self.expand

        return Selection(page_indices=pages, scores=scores, meta=meta)

"""Selector factory: RunConfig -> EvidenceSelector (Stage 1-4 dispatch)."""

from __future__ import annotations

from typing import Optional

from ..config import RunConfig
from .base import (EvidenceSelector, NoRetrieval, Oracle, RetrieverSelector)
from .hybrid import HybridSelector


def _retriever_selector(method: str, cfg: RunConfig, *, reranker=None,
                        base_only: bool = False) -> RetrieverSelector:
    """Build a single retriever's selector.

    ``base_only`` strips the Tier-1 post-processing (used for hybrid components,
    where rerank/cut/expand are applied once on the fused list instead).
    """
    from .retrievers import build_retriever

    retriever = build_retriever(method, cfg.retrieval)
    if base_only:
        return RetrieverSelector(
            retriever=retriever, top_k=cfg.retrieval.top_k,
            parser_name=cfg.representation.parser, dpi=cfg.representation.dpi,
            chunking=cfg.representation.chunking)
    return RetrieverSelector(
        retriever=retriever,
        top_k=cfg.retrieval.top_k,
        parser_name=cfg.representation.parser,
        dpi=cfg.representation.dpi,
        chunking=cfg.representation.chunking,
        k_strategy=cfg.retrieval.k_strategy,
        candidate_k=cfg.retrieval.candidate_k,
        reranker=reranker,
        rerank_candidates=cfg.retrieval.rerank_candidates,
        expand=cfg.retrieval.expand,
        expand_window=cfg.retrieval.expand_window,
    )


def build_selector(cfg: RunConfig, reranker=None) -> EvidenceSelector:
    """Config -> EvidenceSelector. ``reranker`` may be injected (tests); else a
    reranker is constructed from the config when ``retrieval.rerank == 'llm'``."""
    method = cfg.retrieval.method

    if reranker is None and cfg.retrieval.rerank == "llm":
        from .rerank import build_reranker

        reranker = build_reranker(cfg.retrieval)

    if method == "none":
        return NoRetrieval(n_pages=cfg.retrieval.no_retrieval_pages)
    if method == "oracle":
        return Oracle()
    if method in {"grep", "bm25", "tfidf", "dense", "colpali", "colqwen"}:
        return _retriever_selector(method, cfg, reranker=reranker)
    if method == "hybrid":
        # components stay plain; rerank/adaptive-k/expand apply on the fused list.
        subs = [_retriever_selector(m, cfg, base_only=True)
                for m in cfg.retrieval.hybrid_methods]
        return HybridSelector(
            subs, top_k=cfg.retrieval.top_k, rrf_k=cfg.retrieval.rrf_k,
            k_strategy=cfg.retrieval.k_strategy, expand=cfg.retrieval.expand,
            expand_window=cfg.retrieval.expand_window,
            parser_name=cfg.representation.parser, reranker=reranker)
    raise ValueError(f"unknown retrieval method {method!r}")

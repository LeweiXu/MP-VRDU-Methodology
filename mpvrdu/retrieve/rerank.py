"""LLM/VLM reranking — the "retrieve -> rerank -> read" second stage (Tier 1 #3).

The coarse-to-fine pattern shared by most strong systems (SimpleDoc re-ranks an
embedding shortlist with an LLM over page summaries; CREAM/MHier-RAG re-rank in
groups). Here a frozen LLM re-scores the retriever's candidate PAGES using their
parser text, then the selector's cut keeps top_k of the reranked order. This is a
pure post-retrieval pass — it wraps any retriever and changes only ordering.

Backends mirror eval/judge.LLMJudge:
1. an injected ``complete_fn(prompt) -> str`` (tests / any API client),
2. a local HuggingFace causal-LM (lazy, offline-capable) named by ``model_id``.

GPU note (README §3 two-phase): the reranker holds a model during Phase 1
alongside the retriever. Reuse the retriever's slot or run it as a small
Phase-1.5 with explicit load/unload to respect the 12 GB ceiling; ``unload``
frees it before the generator loads.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..logging_utils import get_logger

log = get_logger("rerank")

RERANK_PROMPT = """You are judging whether a document page helps answer a \
question. Rate the page's relevance on an integer scale from 0 (irrelevant) to \
10 (directly contains the answer). Output ONLY the number.

Question: {question}
Page content:
{page}

Relevance (0-10):"""

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# keep prompts bounded: a few thousand chars of page text is plenty for relevance
_MAX_PAGE_CHARS = 4000


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, question: str,
               candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
        """Re-score (page_index, text) candidates; return them best-first."""

    def unload(self) -> None:
        return None


class LLMReranker(Reranker):
    """Scores each candidate page independently 0-10 and sorts descending.

    Per-page scoring (rather than one big list prompt) keeps the context small
    and the order stable — the DREAM / MM-R5 single-page relevance recipe.
    """

    name = "llm"

    def __init__(self, model_id: Optional[str] = None,
                 complete_fn: Optional[Callable[[str], str]] = None,
                 max_new_tokens: int = 8):
        self.model_id = model_id
        self._complete_fn = complete_fn
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    # ---- backend ----
    def _hf_complete(self, prompt: str) -> str:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("loading LLM reranker %s", self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        import torch

        tok, model = self._tokenizer, self._model
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _complete(self, prompt: str) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(prompt)
        if not self.model_id:
            raise ValueError("LLMReranker needs a model_id or a complete_fn")
        return self._hf_complete(prompt)

    def _score_one(self, question: str, page_text: str) -> float:
        prompt = RERANK_PROMPT.format(question=question,
                                      page=(page_text or "")[:_MAX_PAGE_CHARS])
        raw = self._complete(prompt)
        m = _NUM_RE.search(raw or "")
        return float(m.group()) if m else 0.0

    def rerank(self, question: str,
               candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
        scored = [(page, self._score_one(question, text))
                  for page, text in candidates]
        # stable sort by score desc; ties keep the original retrieval order.
        order = sorted(range(len(scored)), key=lambda i: -scored[i][1])
        return [scored[i] for i in order]

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def build_reranker(cfg) -> Reranker:
    """cfg is a RetrievalConfig. Construct the configured reranker."""
    if cfg.rerank == "none":
        raise ValueError("build_reranker called with rerank=none")
    if cfg.rerank == "llm":
        return LLMReranker(model_id=cfg.rerank_model)
    raise ValueError(f"unknown rerank {cfg.rerank!r}")

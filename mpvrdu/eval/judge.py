"""Judge interface: rule-based (local, deterministic) + LLM (faithful scorer).

The judge decides per-question correctness. Local plumbing uses the
deterministic RuleBasedJudge (no model, reproducible). For headline numbers, use
LLMJudge — which MUST be fixed and declared in the methodology (context.md
§6,§8): it is a hidden experimental variable otherwise.

LLMJudge follows MMLongBench-Doc's official scoring shape: an LLM EXTRACTS the
concise final answer from the (possibly verbose) model response, then the SAME
`answer_format`-aware rule comparison used everywhere scores it. So the LLM only
does the fuzzy extraction step; the comparison stays deterministic and tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..config import JudgeConfig
from ..logging_utils import get_logger
from .extract import is_abstention
from .metrics import AnswerScore, score_answer

log = get_logger("judge")


class Judge(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, question: str, pred: str, gold: str, answer_format: str,
              gold_is_unanswerable: Optional[bool] = None) -> AnswerScore:
        ...


class RuleBasedJudge(Judge):
    """Deterministic, model-free judge via the rule comparison in eval/metrics."""

    name = "rule"

    def score(self, question: str, pred: str, gold: str, answer_format: str,
              gold_is_unanswerable: Optional[bool] = None) -> AnswerScore:
        return score_answer(pred, gold, answer_format,
                            gold_is_unanswerable=gold_is_unanswerable)


EXTRACTION_PROMPT = """You extract the concise final answer from a model's \
response to a document question. Output ONLY the short answer, nothing else.
Rules:
- If the response does not answer the question, or says it cannot find/determine \
the answer, output exactly: Not answerable
- For a numeric answer, output just the number (with unit if essential).
- For a list, output the items separated by commas.
- Do not explain.

Question: {question}
Expected answer type: {answer_format}
Model response: {response}

Extracted answer:"""


class LLMJudge(Judge):
    """LLM-extraction + rule-comparison judge (fixed, declared model).

    Backends, in priority order:
    1. an injected ``complete_fn(prompt) -> str`` (tests, or any API client),
    2. a local HuggingFace causal-LM (lazy, offline-capable) named by model_id.
    """

    name = "llm"

    def __init__(self, model_id: str,
                 complete_fn: Optional[Callable[[str], str]] = None,
                 max_new_tokens: int = 64):
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

            log.info("loading LLM judge %s", self.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        tok, model = self._tokenizer, self._model
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        import torch

        inputs = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _complete(self, prompt: str) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(prompt)
        return self._hf_complete(prompt)

    # ---- judging ----
    def extract(self, question: str, response: str, answer_format: str) -> str:
        prompt = EXTRACTION_PROMPT.format(
            question=question, answer_format=answer_format, response=response)
        return self._complete(prompt).strip()

    def score(self, question: str, pred: str, gold: str, answer_format: str,
              gold_is_unanswerable: Optional[bool] = None) -> AnswerScore:
        extracted = self.extract(question, pred, answer_format)
        # The answerability bookkeeping (for F1) uses the EXTRACTED answer, so a
        # verbose hedge that the extractor resolves to "Not answerable" counts as
        # an abstention.
        return score_answer(extracted, gold, answer_format,
                            gold_is_unanswerable=gold_is_unanswerable)


def build_judge(cfg: JudgeConfig, complete_fn: Optional[Callable] = None) -> Judge:
    if cfg.type == "rule":
        return RuleBasedJudge()
    if cfg.type == "llm":
        if not cfg.model_id:
            raise ValueError("judge.type=llm requires judge.model_id")
        return LLMJudge(cfg.model_id, complete_fn=complete_fn)
    raise ValueError(f"unknown judge type {cfg.type!r}")

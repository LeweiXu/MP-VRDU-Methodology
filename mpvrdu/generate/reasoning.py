"""RQ3 reasoning axis — the reasoning STRUCTURE over a fixed evidence buffer.

`docs/research_questions.md` RQ3: hold the assembled evidence identical and vary
only the reasoning structure, so any accuracy delta is attributable to reasoning
alone (not retrieval). This wrapper sits between the pipeline and the raw
generator: `InputBuilder` assembles the (images, text) ONCE per question, and the
strategy here orchestrates one-or-more generator calls over that SAME buffer.

Levels (config `generation.reasoning`):
- ``direct``           — one call, no explicit reasoning (the control).
- ``cot``              — single-pass chain-of-thought; parse the marked final answer.
- ``self_reflection``  — CoT draft + one critic-driven revision pass (sequential).
- ``self_consistency`` — sample N at temp>0, majority-vote the final answers (parallel).
- ``tot``              — bounded tree-of-thoughts: N branch proposals + one select
                         pass (parallel/search). Predicted NULL in MP-VRDU (H3c).

Cost is the co-axis for the parallel methods: every call's tokens roll up into
``last_usage`` and ``n_calls`` reflects the true call count (logged per question).

The wrapper is generator-agnostic — it drives the real Qwen VLM or the MockGenerator
identically, so the whole axis is exercised offline. NEVER report mock numbers.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

from ..eval.extract import normalize_text
from ..logging_utils import get_logger
from .base import Generator, GeneratorInput, Usage

log = get_logger("reasoning")

# The base generator wraps the question in its evidence prompt and appends
# "Question: {question}"; these instructions ride along on the question string.
ANSWER_MARKER = "Final answer:"

_COT_SUFFIX = (
    f" Reason step by step using ONLY the evidence, then end your reply with a "
    f"line '{ANSWER_MARKER} <answer>' giving the shortest exact answer "
    f"(or 'Not answerable')."
)
_REVISE_SUFFIX = (
    " A draft answer was proposed: {draft!r}. Re-check it against the evidence; "
    "if it is wrong or unsupported, correct it. Reason briefly, then end with a "
    f"line '{ANSWER_MARKER} <answer>'."
)
_SELECT_SUFFIX = (
    " Several candidate answers were proposed:\n{candidates}\n"
    "Choose the one best supported by the evidence (or 'Not answerable' if none "
    f"is). End with a line '{ANSWER_MARKER} <answer>'."
)


def extract_final_answer(text: str) -> str:
    """Pull the answer after the last ANSWER_MARKER; fall back to the raw text.

    Models that don't follow the format (and the MockGenerator, which returns the
    gold string verbatim) fall through to the stripped reply, so a missing marker
    never loses an otherwise-correct answer.
    """
    if not text:
        return ""
    idx = text.lower().rfind(ANSWER_MARKER.lower())
    if idx == -1:
        return text.strip()
    tail = text[idx + len(ANSWER_MARKER):].strip()
    if not tail:
        return text.strip()
    return tail.splitlines()[0].strip()


def _vote(answers: list[str]) -> str:
    """Majority vote over normalised answers; return a representative raw form.

    Ties break toward the earliest-sampled answer (stable), matching the
    self-consistency recipe (most frequent reasoning outcome wins)."""
    answers = [a for a in answers if a is not None]
    if not answers:
        return ""
    norm = [normalize_text(a) for a in answers]
    counts = Counter(norm)
    best_norm, _ = max(counts.items(), key=lambda kv: (kv[1], -norm.index(kv[0])))
    # representative: first raw answer whose normalisation is the winner
    for raw, n in zip(answers, norm):
        if n == best_norm:
            return raw
    return answers[0]


class ReasoningGenerator(Generator):
    """Wrap a base Generator with a multi-call reasoning strategy (RQ3)."""

    def __init__(self, base: Generator, strategy: str, n_samples: int = 5,
                 sampling_temperature: float = 0.7):
        super().__init__()
        self.base = base
        self.strategy = strategy
        self.n_samples = max(1, n_samples)
        self.sampling_temperature = sampling_temperature
        self.name = f"{base.name}+{strategy}"

    # ---- mock plumbing: the pipeline sets the qid on the underlying generator ----
    def set_qid(self, qid: str) -> None:
        if hasattr(self.base, "set_qid"):
            self.base.set_qid(qid)

    def unload(self) -> None:
        if hasattr(self.base, "unload"):
            self.base.unload()

    # ---- one underlying call; accumulates into the running usage tally ----
    def _call(self, gi: GeneratorInput, question: str, acc: Usage) -> str:
        out = self.base.answer(question, images=gi.image_paths, text=gi.text)
        u = getattr(self.base, "last_usage", None) or Usage()
        acc.input_tokens += u.input_tokens
        acc.output_tokens += u.output_tokens
        acc.n_images = u.n_images          # per-call image count is constant
        acc.seconds += u.seconds
        acc.n_calls += 1
        return out

    def _sampled_calls(self, gi: GeneratorInput, question: str, acc: Usage,
                       n: int) -> list[str]:
        """n calls at the sampling temperature (restored afterwards)."""
        prev = getattr(self.base, "temperature", None)
        if prev is not None:
            self.base.temperature = max(prev, self.sampling_temperature)
        try:
            return [self._call(gi, question, acc) for _ in range(n)]
        finally:
            if prev is not None:
                self.base.temperature = prev

    # required abstract method (single-shot passthrough); strategies use _call
    def answer(self, question: str, images: Optional[list[str]] = None,
               text: Optional[str] = None) -> str:
        return self.base.answer(question, images=images, text=text)

    def answer_input(self, gi: GeneratorInput) -> str:
        t0 = time.time()
        acc = Usage(n_calls=0)
        q = gi.question

        if self.strategy == "direct":
            pred = self._call(gi, q, acc)

        elif self.strategy == "cot":
            pred = extract_final_answer(self._call(gi, q + _COT_SUFFIX, acc))

        elif self.strategy == "self_reflection":
            draft = extract_final_answer(self._call(gi, q + _COT_SUFFIX, acc))
            revised = self._call(gi, q + _REVISE_SUFFIX.format(draft=draft), acc)
            pred = extract_final_answer(revised)

        elif self.strategy == "self_consistency":
            outs = self._sampled_calls(gi, q + _COT_SUFFIX, acc, self.n_samples)
            pred = _vote([extract_final_answer(o) for o in outs])

        elif self.strategy == "tot":
            # bounded breadth: N branch proposals, then one select pass.
            branches = [extract_final_answer(o) for o in
                        self._sampled_calls(gi, q + _COT_SUFFIX, acc, self.n_samples)]
            listing = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(branches))
            chosen = extract_final_answer(
                self._call(gi, q + _SELECT_SUFFIX.format(candidates=listing), acc))
            # if the selector didn't land on a proposed branch, fall back to vote
            norm_branches = {normalize_text(b) for b in branches}
            pred = chosen if normalize_text(chosen) in norm_branches else _vote(branches)

        else:
            raise ValueError(f"unknown reasoning strategy {self.strategy!r}")

        acc.seconds = round(time.time() - t0, 4)
        self.last_usage = acc
        return pred


def wrap_reasoning(base: Generator, gen_cfg) -> Generator:
    """Wrap ``base`` per ``gen_cfg.reasoning``. ``direct`` returns base unchanged
    (zero overhead, identical to the pre-RQ3 behaviour — the control)."""
    if getattr(gen_cfg, "reasoning", "direct") == "direct":
        return base
    return ReasoningGenerator(
        base, gen_cfg.reasoning,
        n_samples=getattr(gen_cfg, "self_consistency_n", 5),
        sampling_temperature=getattr(gen_cfg, "reasoning_temperature", 0.7))

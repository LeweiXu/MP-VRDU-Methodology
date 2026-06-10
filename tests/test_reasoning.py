"""RQ3 reasoning axis (docs/research_questions.md RQ3, docs/pivot.md Step 5).

Every reasoning level must:
- run end-to-end over the SAME assembled evidence buffer (reasoning is the only
  variable — the retrieved pages are frozen across the sweep), and
- log a call count that reflects the strategy (parallel methods multiply calls).
Driven with the MockGenerator so the whole axis is exercised offline.
"""

import pytest

from mpvrdu.config import dict_to_config
from mpvrdu.generate.base import GeneratorInput, Generator, Usage
from mpvrdu.generate.factory import build_generator
from mpvrdu.generate.reasoning import (ReasoningGenerator, extract_final_answer,
                                       _vote, wrap_reasoning)
from mpvrdu.pipeline import run

REASONINGS = ["direct", "cot", "self_reflection", "self_consistency", "tot"]


def _cfg(reasoning, n=3):
    return dict_to_config({
        "name": f"reason-{reasoning}",
        "data": {"name": "synthetic"},
        "representation": {"parser": "pymupdf", "dpi": 72},
        "retrieval": {"method": "oracle", "top_k": 4},
        "generation": {"generator": "mock", "mock_mode": "gold", "modality": "text",
                       "reasoning": reasoning, "self_consistency_n": n},
        "judge": {"type": "rule"},
    })


def test_extract_final_answer():
    assert extract_final_answer("blah blah\nFinal answer: 42 million") == "42 million"
    assert extract_final_answer("reasoning... FINAL ANSWER: yes") == "yes"
    # no marker -> the raw reply (mock returns gold verbatim)
    assert extract_final_answer("88 percent") == "88 percent"
    assert extract_final_answer("") == ""


def test_vote_picks_majority():
    assert _vote(["7 layers", "7 layers", "nine"]) == "7 layers"
    # normalisation: punctuation/case ignored when counting
    assert _vote(["Yes.", "yes", "no"]) in {"Yes.", "yes"}


@pytest.mark.parametrize("reasoning", REASONINGS)
def test_every_reasoning_level_runs_and_scores(reasoning, synthetic_ds, chdir_tmp):
    # mock(gold) returns the gold answer; every reasoning path must surface it
    m = run(_cfg(reasoning), dataset=synthetic_ds, out_path="out.jsonl")
    assert m["n"] == 7
    assert m["accuracy"] == 1.0          # reasoning plumbing preserves the answer


def test_direct_is_zero_overhead_passthrough():
    base = build_generator(dict_to_config(
        {"generation": {"generator": "mock"}}).generation)
    # direct -> the raw generator, not a wrapper (the control has no overhead)
    assert wrap_reasoning(base, dict_to_config(
        {"generation": {"reasoning": "direct"}}).generation) is base


class _CountingGen(Generator):
    """Counts calls and returns a fixed answer with the marker."""
    name = "counting"

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.temperature = 0.0

    def answer(self, question, images=None, text=None):
        self.calls += 1
        self.last_usage = Usage(input_tokens=10, output_tokens=2, n_calls=1)
        # plain answer (no marker) so every strategy surfaces it identically and
        # the test isolates call-counting from answer-extraction.
        return "42"


def _gi():
    return GeneratorInput(question="q?", image_paths=[], text="evidence", n_pages=1)


@pytest.mark.parametrize("strategy,n,expected_calls", [
    ("direct", 3, 1),
    ("cot", 3, 1),
    ("self_reflection", 3, 2),       # draft + revision
    ("self_consistency", 4, 4),      # N samples
    ("tot", 3, 4),                   # N branches + 1 select
])
def test_call_counts_match_strategy(strategy, n, expected_calls):
    base = _CountingGen()
    g = ReasoningGenerator(base, strategy, n_samples=n)
    pred = g.answer_input(_gi())
    assert pred == "42"
    assert base.calls == expected_calls
    # cost axis: usage reflects the true number of LLM calls
    assert g.last_usage.n_calls == expected_calls
    assert g.last_usage.input_tokens == 10 * expected_calls


def test_sampling_restores_base_temperature():
    base = _CountingGen()
    base.temperature = 0.0
    ReasoningGenerator(base, "self_consistency", n_samples=3,
                       sampling_temperature=0.7).answer_input(_gi())
    assert base.temperature == 0.0       # restored after the sampled calls


def test_reasoning_sweep_freezes_retrieved_pages(synthetic_ds, chdir_tmp):
    """RQ3 control guard: changing ONLY generation.reasoning must not change the
    retrieved pages (else the comparison confounds reasoning with retrieval)."""
    from mpvrdu.retrieve import build_selector

    base_cfg = _cfg("direct")
    sel_pages = {}
    for reasoning in REASONINGS:
        cfg = _cfg(reasoning)
        # retrieval block is identical across the sweep -> identical selections
        assert cfg.retrieval == base_cfg.retrieval
        selector = build_selector(cfg)
        pages = {q.qid: selector.select(q, synthetic_ds.get_document(q.doc_id)).page_indices
                 for q in synthetic_ds.questions}
        selector.unload()
        sel_pages[reasoning] = pages
    # all reasoning conditions saw exactly the same pages per question
    assert all(sel_pages[r] == sel_pages["direct"] for r in REASONINGS)

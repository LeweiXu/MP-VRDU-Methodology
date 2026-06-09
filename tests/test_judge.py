"""LLM judge: extraction + rule comparison, with a fake backend (no model)."""

from mpvrdu.config import JudgeConfig
from mpvrdu.eval.judge import LLMJudge, RuleBasedJudge, build_judge


def _judge(extraction):
    """LLMJudge whose 'LLM' always returns `extraction` (or a function of prompt)."""
    fn = extraction if callable(extraction) else (lambda prompt: extraction)
    return LLMJudge("fake-model", complete_fn=fn)


def test_llm_judge_correct_extraction():
    j = _judge("42")
    s = j.score("How much?", "The total is 42 dollars, roughly.", "42", "Int")
    assert s.correct


def test_llm_judge_wrong_extraction():
    j = _judge("99")
    s = j.score("How much?", "...", "42", "Int")
    assert not s.correct


def test_llm_judge_extracts_from_verbose_response():
    # the LLM resolves a hedge to the concise answer; rule compare then matches
    j = _judge("Paris")
    s = j.score("Capital?", "Well, considering everything, it's Paris.", "Paris", "Str")
    assert s.correct


def test_llm_judge_unanswerable_correct_when_abstains():
    j = _judge("Not answerable")
    s = j.score("CEO home address?", "I cannot find that.", "Not answerable", "None",
                gold_is_unanswerable=True)
    assert s.correct and s.pred_abstained


def test_llm_judge_unanswerable_wrong_when_hallucinates():
    j = _judge("123 Main St")
    s = j.score("CEO home address?", "It's 123 Main St.", "Not answerable", "None",
                gold_is_unanswerable=True)
    assert not s.correct


def test_llm_judge_abstain_on_answerable_is_wrong():
    j = _judge("Not answerable")
    s = j.score("How much?", "unclear", "42", "Int", gold_is_unanswerable=False)
    assert not s.correct


def test_extraction_prompt_includes_question_and_response():
    seen = {}
    j = _judge(lambda prompt: seen.setdefault("p", prompt) or "x")
    j.score("What year?", "It was 2007.", "2007", "Int")
    assert "What year?" in seen["p"] and "It was 2007." in seen["p"]


def test_build_judge_dispatch():
    assert isinstance(build_judge(JudgeConfig(type="rule")), RuleBasedJudge)
    assert isinstance(build_judge(JudgeConfig(type="llm", model_id="m")), LLMJudge)

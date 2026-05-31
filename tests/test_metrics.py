import pytest
import math
from microeval.metrics import (
    eval_classification,
    eval_translation,
    eval_summarization,
)
from microeval.metrics.fluency import flesch_kincaid


class TestClassification:
    def test_perfect(self):
        result = eval_classification(["a", "b"], ["a", "b"])
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_half_wrong(self):
        result = eval_classification(["a", "a"], ["a", "b"])
        assert result["accuracy"] == 0.5

    def test_all_wrong(self):
        result = eval_classification(["b", "b"], ["a", "a"])
        assert result["accuracy"] == 0.0

    def test_empty(self):
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result = eval_classification([], [])
        assert math.isnan(result["accuracy"])
        assert math.isnan(result["f1"])


class TestTranslation:
    def test_perfect_match(self):
        result = eval_translation(
            ["hello world foo bar"], ["hello world foo bar"]
        )
        assert result["bleu"] == pytest.approx(100.0, abs=1e-9)

    def test_no_match(self):
        result = eval_translation(["xyz"], ["hello"])
        assert result["bleu"] == 0.0

    def test_partial(self):
        result = eval_translation(
            ["hello world baz qux"], ["hello world foo bar"]
        )
        assert result["bleu"] > 0.0

    def test_multiple(self):
        result = eval_translation(
            ["hello world foo bar", "goodbye world foo bar"],
            ["hello world foo bar", "goodbye world foo bar"],
        )
        assert result["bleu"] == pytest.approx(100.0, abs=1e-9)


class TestSummarization:
    def test_exact_match(self):
        result = eval_summarization(["hello world"], ["hello world"])
        assert result["rouge1"] > 0.9
        assert result["rouge2"] > 0.9
        assert result["rougeL"] > 0.9

    def test_no_match(self):
        result = eval_summarization(["abc"], ["xyz"])
        assert result["rouge1"] == 0.0
        assert result["rouge2"] == 0.0
        assert result["rougeL"] == 0.0

    def test_partial_overlap(self):
        result = eval_summarization(["hello world"], ["hello"])
        assert result["rouge1"] > 0.0


class TestFluency:
    def test_flesch_kincaid_easy(self):
        score = flesch_kincaid("The cat sat on the mat.")
        assert score >= 80.0

    def test_flesch_kincaid_hard(self):
        score = flesch_kincaid(
            "The fundamental cosmological principle posits that "
            "the distribution of matter in the universe is "
            "homogeneous and isotropic when viewed at sufficiently "
            "large scales."
        )
        assert score <= 40.0

    def test_flesch_kincaid_vowels(self):
        score = flesch_kincaid("I am.")
        assert isinstance(score, float)
        assert score > 0

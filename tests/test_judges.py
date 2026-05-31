"""Tests for judges module.

These tests use monkeypatching to avoid real API calls.
"""

import pytest
from microeval.judges.pairwise import PairwiseJudge, DEFAULT_PAIRWISE_PROMPT
from microeval.judges.pointwise import PointwiseJudge
from microeval.judges.ensemble import MultiJudgeEnsemble
from microeval.judges.calibration import JudgeCalibration


class TestPairwiseJudge:
    def test_parse_winner_a(self):
        judge = PairwiseJudge("test-model")
        result = judge._parse_result("Winner: A", "answer a", "answer b")
        assert result["winner"] == "A"
        assert result["score"] == 1.0

    def test_parse_winner_b(self):
        judge = PairwiseJudge("test-model")
        result = judge._parse_result("Winner: B", "answer a", "answer b")
        assert result["winner"] == "B"
        assert result["score"] == 0.0

    def test_parse_tie(self):
        judge = PairwiseJudge("test-model")
        result = judge._parse_result("Both are equally good", "answer a", "answer b")
        assert result["winner"] == "tie"
        assert result["score"] == 0.5

    def test_all_retries_fail(self):
        judge = PairwiseJudge("test-model", max_retries=1)

        def failing_call(*args, **kwargs):
            raise ConnectionError("API down")

        judge._call_judge = failing_call
        result = judge.compare("A", "B", "question")
        assert result["winner"] == "tie"
        assert result["reason"] == "all_retries_exhausted"


class TestPointwiseJudge:
    def test_parse_score(self):
        judge = PointwiseJudge("test-model")
        result = judge._parse_result("8/10")
        assert result["score"] == 0.8

    def test_parse_decimal(self):
        judge = PointwiseJudge("test-model")
        result = judge._parse_result("7.5/10")
        assert result["score"] == 0.75

    def test_parse_no_score(self):
        judge = PointwiseJudge("test-model")
        result = judge._parse_result("No numeric score found")
        assert result["score"] == 0.0


class TestMultiJudgeEnsemble:
    def test_majority_vote(self):
        class MockJudge:
            def __init__(self, winner):
                self._winner = winner

            def compare(self, a, b, q=""):
                score = 1.0 if self._winner == "A" else 0.0
                return {"winner": self._winner, "score": score}

        ensemble = MultiJudgeEnsemble([
            MockJudge("A"),
            MockJudge("A"),
            MockJudge("B"),
        ])
        result = ensemble.compare("ans_a", "ans_b", "question")
        assert result["majority_winner"] == "A"
        assert result["agreement"] == 2.0 / 3.0
        assert result["num_judges"] == 3

    def test_handles_failures_gracefully(self):
        class WorkingJudge:
            def compare(self, a, b, q=""):
                return {"winner": "A", "score": 1.0}

        class BrokenJudge:
            def compare(self, a, b, q=""):
                raise ValueError("broken")

        ensemble = MultiJudgeEnsemble([WorkingJudge(), BrokenJudge()])
        result = ensemble.compare("a", "b")
        assert result["majority_winner"] == "A"
        assert result["num_valid_votes"] == 1


class TestJudgeCalibration:
    def test_returns_report_keys(self):
        class MockJudge:
            def compare(self, a, b, q=""):
                return {"winner": "A", "score": 1.0}

        cal = JudgeCalibration()
        report = cal.evaluate(MockJudge(), num_samples=5)
        assert "position_bias" in report
        assert "consistency" in report
        assert "length_bias" in report
        assert "self_enhancement_bias" in report

    def test_position_bias_returns_float(self):
        class MockJudge:
            def compare(self, a, b, q=""):
                return {"winner": "A", "score": 1.0}

        cal = JudgeCalibration()
        bias = cal._measure_position_bias(MockJudge(), n=5)
        assert isinstance(bias, float)
        assert 0.0 <= bias <= 1.0

    def test_consistency_returns_float(self):
        class MockJudge:
            def compare(self, a, b, q=""):
                return {"winner": "A", "score": 1.0}

        cal = JudgeCalibration()
        c = cal._measure_consistency(MockJudge(), n=5)
        assert isinstance(c, float)

    def test_non_pairwise_judge_position_bias_zero(self):
        """Non-PairwiseJudge should get 0.0 position bias."""

        class OtherJudge:
            pass

        cal = JudgeCalibration()
        bias = cal._measure_position_bias(OtherJudge(), n=5)
        assert bias == 0.0

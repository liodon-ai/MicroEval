import logging
from typing import Any, Dict
from statistics import mean

logger = logging.getLogger(__name__)


class JudgeCalibration:
    """Measure biases and reliability of an LLM judge.

    Runs a battery of controlled tests to quantify:
    - Position bias: does the judge prefer answer A or B regardless of content?
    - Consistency: does the judge give the same answer repeatedly?
    - Length bias: does the judge prefer longer answers?
    - Self-enhancement bias: does the judge prefer answers that praise it?
    """

    def evaluate(self, judge: Any, num_samples: int = 20) -> Dict[str, float]:
        """Run all calibration tests and return a report.

        Args:
            judge: A judge object with a ``compare()`` method.
            num_samples: Number of trials per test (default: 20).

        Returns:
            dict with keys:
                - "position_bias": 0.0 = no bias, 1.0 = always prefers position A
                - "consistency": 1.0 = perfectly consistent, 0.0 = random
                - "length_bias": >0 prefers long, <0 prefers short
                - "self_enhancement_bias": >0 prefers self-praise
        """
        report = {}
        report["position_bias"] = self._measure_position_bias(judge, num_samples)
        report["consistency"] = self._measure_consistency(judge, num_samples)
        report["length_bias"] = self._measure_length_bias(judge, num_samples)
        report["self_enhancement_bias"] = self._measure_self_enhancement(
            judge, num_samples
        )
        return report

    def _measure_position_bias(self, judge: Any, n: int) -> float:
        """Check if judge consistently prefers A or B when answers are identical."""
        from microeval.judges.pairwise import PairwiseJudge

        if not isinstance(judge, PairwiseJudge):
            return 0.0

        test_a = "The answer is 42."
        test_b = "The answer is 42."

        results = []
        for _ in range(n):
            r1 = judge.compare(test_a, test_b, "What is 6 times 7?")
            results.append(r1["winner"])

        a_count = results.count("A")
        b_count = results.count("B")
        total = a_count + b_count
        if total == 0:
            return 0.0
        return abs(a_count - b_count) / total

    def _measure_consistency(self, judge: Any, n: int) -> float:
        """Check if judge gives the same answer when repeated."""
        question = "What is the capital of France?"
        answer_a = "Paris"
        answer_b = "London"

        results = []
        for _ in range(n):
            r = judge.compare(answer_a, answer_b, question)
            results.append(r["winner"])

        if not results:
            return 0.0
        majority = max(set(results), key=results.count)
        return results.count(majority) / len(results)

    def _measure_length_bias(self, judge: Any, n: int) -> float:
        """Check if judge prefers longer answers by swapping position."""
        base_answer = "The answer is 42."
        long_answer = (
            "The answer is 42. Let me explain why. First, "
            "we need to consider the fundamental principles. "
            "The universe is vast and complex, and many factors "
            "contribute to the final result. After careful analysis, "
            "it becomes clear that 42 is indeed the correct answer."
        )

        results_a = []
        results_b = []
        for _ in range(n):
            r1 = judge.compare(base_answer, long_answer, "What is 6 times 7?")
            r2 = judge.compare(long_answer, base_answer, "What is 6 times 7?")
            results_a.append(r1["winner"])
            results_b.append(r2["winner"])

        prefer_long = sum(
            1
            for w1, w2 in zip(results_a, results_b)
            if w1 == "B" and w2 == "A"
        )
        prefer_short = sum(
            1
            for w1, w2 in zip(results_a, results_b)
            if w1 == "A" and w2 == "B"
        )
        total = prefer_long + prefer_short
        if total == 0:
            return 0.0
        return (prefer_long - prefer_short) / total

    def _measure_self_enhancement(self, judge: Any, n: int) -> float:
        """Check if judge prefers answers that praise it."""
        judge_name = getattr(judge, "model", "unknown")
        self_answer = f"I am {judge_name}, a highly capable AI assistant."
        other_answer = "I am a simple rule-based system."

        results = []
        for _ in range(n):
            r = judge.compare(self_answer, other_answer, "Describe yourself.")
            results.append(r["winner"])

        self_wins = results.count("A")
        total = len(results)
        if total == 0:
            return 0.0
        return self_wins / total

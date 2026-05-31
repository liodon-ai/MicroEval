import logging
from typing import List, Dict, Any
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class MultiJudgeEnsemble:
    def __init__(self, judges: List[Any]):
        self.judges = judges

    def compare(
        self, answer_a: str, answer_b: str, question: str = ""
    ) -> Dict[str, Any]:
        votes = []
        for judge in self.judges:
            try:
                result = judge.compare(answer_a, answer_b, question)
                votes.append(result)
            except Exception as e:
                logger.warning("Judge in ensemble failed: %s", e)

        winners = [v["winner"] for v in votes]
        majority = max(set(winners), key=winners.count) if winners else "tie"
        agreement = winners.count(majority) / len(winners) if winners else 0.0

        scores = [v["score"] for v in votes]
        return {
            "majority_winner": majority,
            "agreement": agreement,
            "vote_distribution": {w: winners.count(w) for w in set(winners)},
            "mean_score": mean(scores) if scores else 0.5,
            "std_score": stdev(scores) if len(scores) > 1 else 0.0,
            "num_judges": len(self.judges),
            "num_valid_votes": len(votes),
        }

    def score(self, answer: str, question: str = "", rubric: str = "") -> Dict[str, Any]:
        scores = []
        for judge in self.judges:
            try:
                result = judge.score(answer, question, rubric)
                scores.append(result["score"])
            except Exception as e:
                logger.warning("Judge in ensemble failed: %s", e)

        return {
            "mean_score": mean(scores) if scores else 0.0,
            "std_score": stdev(scores) if len(scores) > 1 else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "num_judges": len(self.judges),
            "num_valid_votes": len(scores),
        }

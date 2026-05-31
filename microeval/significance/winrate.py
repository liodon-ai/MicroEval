from statistics import mean
from typing import Dict, List, Optional

from microeval.significance.bootstrap import bootstrap_ci, bootstrap_pvalue


def win_rate(
    scores_a: List[float],
    scores_b: List[float],
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """Compute pairwise win/tie/loss rates between two sets of scores.

    Each position i in scores_a is compared against scores_b[i].

    Args:
        scores_a: Scores from model A (paired with scores_b).
        scores_b: Scores from model B.
        higher_is_better: If True, higher score = win (default: True).

    Returns:
        dict with keys "win_rate", "tie_rate", "loss_rate" (0-1 each),
        and "num_comparisons".
    """
    wins = 0
    ties = 0
    losses = 0

    for sa, sb in zip(scores_a, scores_b):
        if higher_is_better:
            if sa > sb:
                wins += 1
            elif sa < sb:
                losses += 1
            else:
                ties += 1
        else:
            if sa < sb:
                wins += 1
            elif sa > sb:
                losses += 1
            else:
                ties += 1

    total = len(scores_a)
    return {
        "win_rate": wins / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
        "num_comparisons": total,
    }


def compare_models(
    scores_a: List[float],
    scores_b: List[float],
    higher_is_better: bool = True,
    num_resamples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Full head-to-head comparison of two models.

    Combines win-rate analysis with bootstrap confidence intervals
    and permutation p-value.

    Args:
        scores_a: Scores from model A.
        scores_b: Scores from model B.
        higher_is_better: If True, higher score = win.
        num_resamples: Bootstrap/permutation resamples (default: 1000).
        seed: Optional random seed.

    Returns:
        dict with keys: "win_rate", "tie_rate", "loss_rate", "mean_a",
        "mean_b", "ci_lower", "ci_upper", "p_value", "num_resamples".
    """
    wr = win_rate(scores_a, scores_b, higher_is_better)

    all_scores = scores_a + scores_b
    ci = bootstrap_ci(all_scores, num_resamples=num_resamples, seed=seed)

    pval = bootstrap_pvalue(
        scores_a, scores_b, num_resamples=num_resamples, seed=seed
    )

    return {
        "win_rate": wr["win_rate"],
        "tie_rate": wr["tie_rate"],
        "loss_rate": wr["loss_rate"],
        "mean_a": mean(scores_a) if scores_a else 0.0,
        "mean_b": mean(scores_b) if scores_b else 0.0,
        "ci_lower": ci["lower"],
        "ci_upper": ci["upper"],
        "p_value": pval["p_value"],
        "num_resamples": num_resamples,
    }

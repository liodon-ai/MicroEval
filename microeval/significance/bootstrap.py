import random
from typing import Callable, List, Optional
from statistics import mean, stdev


def bootstrap_ci(
    samples: List[float],
    metric: Callable[[List[float]], float] = mean,
    num_resamples: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> dict:
    """Compute bootstrap confidence interval for a metric over samples.

    Uses the percentile bootstrap method: resamples with replacement,
    computes the metric on each resample, and takes percentiles.

    Args:
        samples: List of observed values.
        metric: Function that computes a statistic from a list of floats.
            Defaults to the arithmetic mean.
        num_resamples: Number of bootstrap resamples (default: 1000).
        ci: Confidence level (default: 0.95 for 95% CI).
        seed: Optional random seed for reproducibility.

    Returns:
        dict with keys:
            - "estimate": metric on original samples
            - "lower": lower bound of CI
            - "upper": upper bound of CI
            - "std_err": standard error (std of bootstrap distribution)
            - "ci_level": requested confidence level
            - "num_resamples": number of resamples
    """
    if seed is not None:
        random.seed(seed)

    if not samples:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0, "std_err": 0.0}

    estimates = []
    n = len(samples)
    for _ in range(num_resamples):
        resample = [random.choice(samples) for _ in range(n)]
        estimates.append(metric(resample))

    estimates.sort()
    lower_idx = int((1 - ci) / 2 * num_resamples)
    upper_idx = int((1 + ci) / 2 * num_resamples)

    return {
        "estimate": metric(samples),
        "lower": estimates[lower_idx],
        "upper": estimates[upper_idx],
        "std_err": stdev(estimates),
        "ci_level": ci,
        "num_resamples": num_resamples,
    }


def bootstrap_pvalue(
    group_a: List[float],
    group_b: List[float],
    num_resamples: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """Compute a permutation p-value for the difference of means.

    Shuffles group labels and counts how often the permuted difference
    exceeds the observed difference.

    Args:
        group_a: Scores from model A.
        group_b: Scores from model B.
        num_resamples: Number of permutations (default: 1000).
        seed: Optional random seed for reproducibility.

    Returns:
        dict with keys:
            - "observed_diff": mean(group_a) - mean(group_b)
            - "p_value": two-sided permutation p-value
            - "num_resamples": number of permutations
    """
    if seed is not None:
        random.seed(seed)

    observed_diff = mean(group_a) - mean(group_b)

    combined = group_a + group_b
    n_a = len(group_a)

    count_extreme = 0
    for _ in range(num_resamples):
        random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = mean(perm_a) - mean(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    return {
        "observed_diff": observed_diff,
        "p_value": (count_extreme + 1) / (num_resamples + 1),
        "num_resamples": num_resamples,
    }

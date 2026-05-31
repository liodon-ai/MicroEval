"""lm-eval-harness integration layer.

Registers MicroEval metrics as lm-eval compatible metrics via
the ``@register_metric`` decorator.

Usage:
    .. code-block:: bash

        pip install microeval[lmeval]
        python -c "from microeval.lmeval import register_all; register_all()"
        lm-eval --model hf --model_args pretrained=gpt2 --tasks my_task
"""

import logging

logger = logging.getLogger(__name__)

_registered = False


def register_all():
    """Register all MicroEval metrics with the lm-eval-harness registry.

    Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return

    try:
        from lm_eval.api.registry import (
            register_metric,
            register_aggregation,
            metric_registry,
        )
        from lm_eval.api import metrics as _  # noqa: F401
    except ImportError:
        logger.warning("lm_eval is not installed. Skipping metric registration.")
        return

    try:
        register_aggregation("mean")(_mean)
    except ValueError:
        pass

    if "pairwise_judge" not in metric_registry._objs:

        @register_metric(
            metric="pairwise_judge",
            higher_is_better=True,
            output_type="generate_until",
            aggregation="mean",
        )
        def pairwise_judge_score(items, judge_model="gpt-4", **kwargs):
            """LLM pairwise judge score. Requires ``microeval[judge]``."""
            from microeval.judges import PairwiseJudge

            judge = PairwiseJudge(model=judge_model)
            scores = []
            for ref, pred in items:
                result = judge.compare(pred, ref)
                scores.append(result["score"])
            return scores

    if "pointwise_judge" not in metric_registry._objs:

        @register_metric(
            metric="pointwise_judge",
            higher_is_better=True,
            output_type="generate_until",
            aggregation="mean",
        )
        def pointwise_judge_score(items, judge_model="gpt-4", **kwargs):
            """LLM pointwise judge score. Requires ``microeval[judge]``."""
            from microeval.judges import PointwiseJudge

            judge = PointwiseJudge(model=judge_model)
            scores = []
            for ref, pred in items:
                result = judge.score(pred)
                scores.append(result["score"])
            return scores

    _registered = True
    logger.info("Registered microeval metrics: pairwise_judge, pointwise_judge")


def _mean(items):
    from statistics import mean

    return mean(items)

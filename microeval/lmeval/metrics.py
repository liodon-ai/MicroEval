try:
    from lm_eval.api.registry import register_metric

    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False

    def register_metric(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


if _LM_EVAL_AVAILABLE:

    @register_metric(
        metric="pairwise_judge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="mean",
    )
    def pairwise_judge_score(items, judge_model="gpt-4", **kwargs):
        from microeval.judges import PairwiseJudge

        judge = PairwiseJudge(model=judge_model)
        scores = []
        for ref, pred in items:
            result = judge.compare(pred, ref)
            scores.append(result["score"])
        return scores

    @register_metric(
        metric="pointwise_judge",
        higher_is_better=True,
        output_type="generate_until",
        aggregation="mean",
    )
    def pointwise_judge_score(items, judge_model="gpt-4", **kwargs):
        from microeval.judges import PointwiseJudge

        judge = PointwiseJudge(model=judge_model)
        scores = []
        for ref, pred in items:
            result = judge.score(pred)
            scores.append(result["score"])
        return scores


else:

    def pairwise_judge_score(items, judge_model="gpt-4", **kwargs):
        raise ImportError(
            "lm_eval is not installed. Install with: pip install lm-eval"
        )

    def pointwise_judge_score(items, judge_model="gpt-4", **kwargs):
        raise ImportError(
            "lm_eval is not installed. Install with: pip install lm-eval"
        )

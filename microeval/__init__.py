"""Lightweight NLP/LLM evaluation toolkit.

MicroEval provides metrics, LLM-as-Judge infrastructure, statistical
significance testing, and prompt robustness analysis — designed as a
shared dependency for evaluation frameworks like LM Eval Harness,
OpenCompass, and Lighteval.

Install:
    ``pip install microeval``

    Optional feature sets:
    - ``microeval[judge]`` — LLM-as-Judge (openai)
    - ``microeval[semantic]`` — embedding similarity (sentence-transformers)
    - ``microeval[lmeval]`` — lm-eval-harness integration
    - ``microeval[full]`` — all metrics (sklearn, sacrebleu, transformers, etc.)
    - ``microeval[all]`` — everything

Quick Start:

    **Metrics:**
    >>> from microeval import eval_classification, perplexity, bert_score
    >>> eval_classification(["pos", "neg"], ["pos", "pos"])
    {'accuracy': 0.5, 'f1': 0.5}
    >>> perplexity("This is a sentence.")
    102.5  # (approximate)

    **LLM-as-Judge:**
    >>> from microeval import PairwiseJudge, JudgeCalibration
    >>> judge = PairwiseJudge("gpt-4")
    >>> judge.compare("Paris is capital.", "It's Paris.", "What is the capital of France?")
    {'winner': 'A', 'score': 1.0, 'reason': '...'}
    >>> JudgeCalibration().evaluate(judge)
    {'position_bias': 0.05, 'consistency': 0.95, ...}

    **Statistical Significance:**
    >>> from microeval import bootstrap_ci, compare_models
    >>> bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], num_resamples=1000)
    {'estimate': 3.0, 'lower': 2.0, 'upper': 4.0, ...}

    **Prompt Robustness:**
    >>> from microeval import PromptVariance
    >>> pv = PromptVariance(model_fn=lambda p: "output")
    >>> pv.analyze("What is 2+2?", num_variants=3)
    {'num_variants': 3, 'output_length_mean': 6.0, ...}

    **LM Eval Harness Integration:**
    >>> from microeval.lmeval import register_all
    >>> register_all()  # registers pairwise_judge, pointwise_judge metrics

Source: https://github.com/liodon-ai/MicroEval
Documentation: https://liodon-ai.github.io/MicroEval/
"""

import microeval.lmeval.metrics as _lmeval_metrics

# Hide implementation details from docs
__pdoc__ = {
    "metrics.fluency._load_gpt2": False,
    "metrics.fluency._tokenizer": False,
    "metrics.fluency._model": False,
    "metrics.fluency._count_syllables": False,
    "metrics.semantic._load_encoder": False,
    "metrics.semantic._encoder": False,
    "metrics.semantic._SENTENCE_TRANSFORMERS_AVAILABLE": False,
    "judges.pairwise._call_judge": False,
    "judges.pairwise._parse_result": False,
    "judges.pairwise.DEFAULT_PAIRWISE_PROMPT": False,
    "judges.pointwise._call_judge": False,
    "judges.pointwise._parse_result": False,
    "judges.pointwise.DEFAULT_POINTWISE_PROMPT": False,
    "judges.calibration._measure_position_bias": False,
    "judges.calibration._measure_consistency": False,
    "judges.calibration._measure_length_bias": False,
    "judges.calibration._measure_self_enhancement": False,
    "lmeval.metrics._mean": False,
    "lmeval.metrics._registered": False,
    "prompt.sensitivity.DEFAULT_TEMPLATES": False,
    "prompt.sensitivity._compute_stats": False,
}

from microeval.metrics import (
    eval_classification,
    eval_translation,
    eval_summarization,
    perplexity,
    flesch_kincaid,
    bert_score,
    semantic_similarity,
)
from microeval.judges import PairwiseJudge, PointwiseJudge, MultiJudgeEnsemble, JudgeCalibration
from microeval.significance import bootstrap_ci, bootstrap_pvalue, win_rate, compare_models
from microeval.prompt import PromptVariance

__all__ = [
    "eval_classification",
    "eval_translation",
    "eval_summarization",
    "perplexity",
    "flesch_kincaid",
    "bert_score",
    "semantic_similarity",
    "PairwiseJudge",
    "PointwiseJudge",
    "MultiJudgeEnsemble",
    "JudgeCalibration",
    "bootstrap_ci",
    "bootstrap_pvalue",
    "win_rate",
    "compare_models",
    "PromptVariance",
]

"""Lightweight NLP/LLM evaluation toolkit for metrics, LLM-as-Judge,
statistical significance testing, prompt robustness analysis, and CLI.

Designed for fast smoke-tests, demos, and as a drop-in dependency for
frameworks like LM Eval Harness, DeepEval, Lighteval, and LangChain.

## Install

```bash
pip install juryeval
```

Optional extras:

| Extra | What you get |
|-------|-------------|
| `[judge]` | LLM-as-Judge (openai) |
| `[semantic]` | Embedding similarity (sentence-transformers) |
| `[lmeval]` | lm-eval-harness integration |
| `[full]` | All metrics (sklearn, sacrebleu, transformers, torch, etc.) |
| `[all]` | Everything |

## Python API

### Metrics

```python
from juryeval import (
    eval_classification, eval_translation, eval_summarization,
    perplexity, flesch_kincaid, bert_score,
)

acc_f1 = eval_classification(preds=["pos", "neg"], refs=["pos", "pos"])
bleu   = eval_translation(preds=["hello world"], refs=["bonjour le monde"])
rouge  = eval_summarization(preds=["summary"], refs=["reference"])
ppl    = perplexity("This is a sentence.")
fk     = flesch_kincaid("This is easy to read.")
bs     = bert_score(preds=["answer"], refs=["reference"])
```

### LLM-as-Judge

```python
from juryeval import PairwiseJudge, PointwiseJudge, MultiJudgeEnsemble, JudgeCalibration

# Pairwise comparison
judge = PairwiseJudge("gpt-4")
result = judge.compare(
    answer_a="Paris is the capital of France.",
    answer_b="It's Paris.",
    question="What is the capital of France?",
)
# {"winner": "A", "score": 1.0, "reason": "..."}

# Pointwise scoring
scorer = PointwiseJudge("gpt-4")
result = scorer.score("Paris is the capital.", question="What is the capital of France?")
# {"score": 0.9, "reason": "..."}

# Multi-judge ensemble
from juryeval import MultiJudgeEnsemble
ensemble = MultiJudgeEnsemble([
    PairwiseJudge("gpt-4"),
    PairwiseJudge("claude-3-opus"),
])
result = ensemble.compare(answer_a, answer_b, question)
# {"majority_winner": "A", "agreement": 0.67, ...}

# Judge calibration
cal = JudgeCalibration()
report = cal.evaluate(judge)
# {"position_bias": 0.05, "consistency": 0.95, ...}
```

### Statistical Significance

```python
from juryeval import bootstrap_ci, compare_models

ci = bootstrap_ci(scores, num_resamples=2000)
# {"estimate": 0.72, "lower": 0.68, "upper": 0.76, "std_err": 0.02}

result = compare_models(model_a_scores, model_b_scores)
# {"win_rate": 0.65, "p_value": 0.003, "mean_a": 0.72, "mean_b": 0.68, ...}
```

### Prompt Robustness

```python
from juryeval import PromptVariance

pv = PromptVariance(model_fn=lambda prompt: "output")
report = pv.analyze("What is 2+2?")
# {"num_variants": 7, "output_length_mean": 5.0, "outputs": [...], ...}
```

## CLI

```bash
# Score a single output
juryeval score --question "What is 2+2?" --output "4"

# Compare two outputs
juryeval compare --question "Capital of France?" --output-a "Paris" --output-b "London"

# Evaluate a dataset
juryeval evaluate --metric classification --predictions preds.json --references refs.json

# Judge calibration
juryeval calibrate --model gpt-4

# Prompt sensitivity analysis
juryeval prompt --question "Explain AI" --num-variants 5
```

## Framework Integrations

| Framework | Setup |
|-----------|-------|
| **lm-eval-harness** | `pip install juryeval[lmeval]` then `from juryeval.lmeval import register_all; register_all()` |
| **DeepEval** | `pip install deepeval[juryeval]` then `from deepeval.metrics.juryeval import JuryEvalMetric` |
| **Lighteval** | `pip install lighteval[juryeval]` then use `JuryEvalPointwiseJudge` / `JuryEvalPairwiseJudge` metrics |
| **LangChain** | `pip install langchain[juryeval]` |

## Development

```bash
pip install pytest
pytest tests/ -v
```

Source: https://github.com/pydev42/juryeval
Documentation: https://pydev42.github.io/juryeval/
License: MIT
"""

import juryeval.lmeval.metrics as _lmeval_metrics

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

from juryeval.metrics import (
    eval_classification,
    eval_translation,
    eval_summarization,
    perplexity,
    flesch_kincaid,
    bert_score,
    semantic_similarity,
)
from juryeval.judges import PairwiseJudge, PointwiseJudge, MultiJudgeEnsemble, JudgeCalibration
from juryeval.significance import bootstrap_ci, bootstrap_pvalue, win_rate, compare_models
from juryeval.prompt import PromptVariance

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

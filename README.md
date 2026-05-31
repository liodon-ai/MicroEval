# MicroEval

Lightweight NLP/LLM evaluation toolkit — metrics, LLM-as-Judge infrastructure, statistical significance testing, and prompt robustness analysis.

Designed for fast smoke-tests, demos, and as a shared dependency for evaluation frameworks like LM Eval Harness, OpenCompass, and Lighteval.

## Install

```bash
pip install microeval

# Optional feature sets:
pip install microeval[full]    # all metrics (sklearn, sacrebleu, transformers, etc.)
pip install microeval[judge]   # LLM-as-Judge (openai)
pip install microeval[semantic]  # embedding similarity (sentence-transformers)
pip install microeval[lmeval]  # lm-eval-harness integration
pip install microeval[all]     # everything
```

## Usage

### Metrics

```python
from microeval import (
    eval_classification, eval_translation, eval_summarization,
    perplexity, flesch_kincaid, bert_score,
)

acc_f1 = eval_classification(preds=["pos", "neg"], refs=["pos", "pos"])
bleu   = eval_translation(preds=["hello world"], refs=["bonjour le monde"])
rouge  = eval_summarization(preds=["summary here"], refs=["reference here"])
ppl    = perplexity("This is a sentence.")
fk     = flesch_kincaid("This is easy to read.")
bs     = bert_score(preds=["answer"], refs=["reference"])
```

### LLM-as-Judge

```python
from microeval import PairwiseJudge, PointwiseJudge, MultiJudgeEnsemble, JudgeCalibration

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
ensemble = MultiJudgeEnsemble([
    PairwiseJudge("gpt-4"),
    PairwiseJudge("claude-3-opus"),
    PairwiseJudge("gemini-pro"),
])
result = ensemble.compare(answer_a, answer_b, question)
# {"majority_winner": "A", "agreement": 0.67, "vote_distribution": {...}, ...}

# Judge calibration
cal = JudgeCalibration()
report = cal.evaluate(judge)
# {"position_bias": 0.05, "consistency": 0.95, "length_bias": 0.1, ...}
```

### Statistical Significance

```python
from microeval import bootstrap_ci, compare_models

ci = bootstrap_ci(scores, num_resamples=2000)
# {"estimate": 0.72, "lower": 0.68, "upper": 0.76, "std_err": 0.02}

result = compare_models(model_a_scores, model_b_scores)
# {"win_rate": 0.65, "p_value": 0.003, "mean_a": 0.72, "mean_b": 0.68, ...}
```

### Prompt Robustness

```python
from microeval import PromptVariance

pv = PromptVariance(model_fn=lambda prompt: "output")
report = pv.analyze("What is 2+2?")
# {"num_variants": 7, "output_length_mean": 5.0, "outputs": [...], ...}
```

### LM Eval Harness Integration

```bash
pip install microeval[lmeval]
python -c "from microeval.lmeval import register_all; register_all()"

# Then register pairwise_judge / pointwise_judge metrics in your task YAML:
# metric_list:
#   - metric: pairwise_judge
#     aggregation: mean
#     higher_is_better: true
```

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT

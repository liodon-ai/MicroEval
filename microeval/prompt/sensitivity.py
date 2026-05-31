import itertools
import logging
from typing import Callable, Dict, List, Optional, Any
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class PromptVariance:
    def __init__(self, model_fn: Callable[[str], str]):
        self.model_fn = model_fn

    def analyze(
        self,
        task_prompt: str,
        templates: Optional[List[str]] = None,
        num_variants: int = 5,
    ) -> Dict[str, Any]:
        templates = templates or DEFAULT_TEMPLATES[:num_variants]
        results = []

        for template in templates:
            prompt = template.format(task=task_prompt)
            try:
                output = self.model_fn(prompt)
                results.append(
                    {"template": template, "output": output, "prompt": prompt}
                )
            except Exception as e:
                logger.warning("PromptVariant failed for template: %s", e)

        return self._compute_stats(results)

    def _compute_stats(self, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {"num_variants": 0, "error": "all_variants_failed"}

        output_lengths = [len(r["output"]) for r in results]
        return {
            "num_variants": len(results),
            "output_length_mean": mean(output_lengths),
            "output_length_std": stdev(output_lengths) if len(output_lengths) > 1 else 0.0,
            "output_length_min": min(output_lengths),
            "output_length_max": max(output_lengths),
            "outputs": [r["output"] for r in results],
            "prompts": [r["prompt"] for r in results],
        }


DEFAULT_TEMPLATES = [
    "{task}",
    "Please answer: {task}",
    "Question: {task}\nAnswer:",
    "Solve the following: {task}",
    "I need help with: {task}",
    "{task}\nLet's think step by step.",
    "Q: {task}\nA:",
]

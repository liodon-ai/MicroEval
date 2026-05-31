import logging
from typing import Callable, Dict, List, Optional, Any
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class PromptVariance:
    """Measure how sensitive a model's outputs are to prompt template changes.

    Runs the same task prompt through multiple template variants and
    reports statistics on output length and variability.

    Args:
        model_fn: A callable that takes a prompt string and returns
            a response string.
    """

    def __init__(self, model_fn: Callable[[str], str]):
        self.model_fn = model_fn

    def analyze(
        self,
        task_prompt: str,
        templates: Optional[List[str]] = None,
        num_variants: int = 5,
    ) -> Dict[str, Any]:
        """Analyze prompt sensitivity across template variants.

        Args:
            task_prompt: The core task instruction to wrap in templates.
            templates: Custom list of template strings. Each must have a
                {task} placeholder. Defaults to 7 built-in templates.
            num_variants: Number of default templates to use (ignored
                if custom templates are provided).

        Returns:
            dict with keys:
                - "num_variants": number of templates that succeeded
                - "output_length_mean": mean output length
                - "output_length_std": std dev of output length
                - "output_length_min": minimum output length
                - "output_length_max": maximum output length
                - "outputs": list of model responses
                - "prompts": list of prompt templates used
        """
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
            "output_length_std": (
                stdev(output_lengths) if len(output_lengths) > 1 else 0.0
            ),
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

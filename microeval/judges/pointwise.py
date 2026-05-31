import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


class PointwiseJudge:
    """LLM-as-a-Judge for scoring a single answer on a 0-10 scale.

    Args:
        model: LLM model name (default: "gpt-4").
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        temperature: Sampling temperature (default: 0.0).
        max_retries: Number of retry attempts on API failure (default: 3).
        prompt_template: Custom prompt template. Must have {question},
            {answer}, {rubric} placeholders.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        prompt_template: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self._prompt_template = prompt_template or DEFAULT_POINTWISE_PROMPT

    def score(self, answer: str, question: str = "", rubric: str = "") -> dict:
        """Score a single answer on a 0-1 scale (extracted from X/10 output).

        Args:
            answer: The answer string to evaluate.
            question: Optional question that produced the answer.
            rubric: Optional scoring rubric / criteria.

        Returns:
            dict with keys:
                - "score": float between 0.0 and 1.0
                - "reason": Raw LLM response text
        """
        for attempt in range(self.max_retries):
            try:
                result = self._call_judge(question, answer, rubric)
                return self._parse_result(result)
            except Exception as e:
                logger.warning(
                    "PointwiseJudge attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
        return {"score": 0.0, "reason": "all_retries_exhausted"}

    def _call_judge(self, question: str, answer: str, rubric: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        prompt = self._prompt_template.format(
            question=question, answer=answer, rubric=rubric
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _parse_result(self, raw: str) -> dict:
        scores = re.findall(r"(\d+(?:\.\d+)?)\s*/\s*10", raw)
        if scores:
            return {"score": float(scores[0]) / 10.0, "reason": raw}
        return {"score": 0.0, "reason": raw}


DEFAULT_POINTWISE_PROMPT = """You are a helpful judge. Score the following answer on a scale of 0-10.

Question: {question}

Answer: {answer}

{rubric}

Output your score as X/10 on a new line."""

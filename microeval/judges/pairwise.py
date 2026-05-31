import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class PairwiseJudge:
    """LLM-as-a-Judge for pairwise comparison of two answers.

    Uses an LLM (e.g. GPT-4) to compare two answers to the same question
    and decide which is better. Supports retry with exponential backoff.

    Args:
        model: LLM model name (default: "gpt-4").
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        temperature: Sampling temperature (default: 0.0).
        max_retries: Number of retry attempts on API failure (default: 3).
        prompt_template: Custom prompt template. Must have {question},
            {answer_a}, {answer_b} placeholders.
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
        self._prompt_template = prompt_template or DEFAULT_PAIRWISE_PROMPT

    def compare(self, answer_a: str, answer_b: str, question: str = "") -> dict:
        """Compare two answers and determine the better one.

        Args:
            answer_a: First answer string.
            answer_b: Second answer string.
            question: Optional question/prompt that produced the answers.

        Returns:
            dict with keys:
                - "winner": "A", "B", or "tie"
                - "score": 1.0 (A wins), 0.0 (B wins), or 0.5 (tie)
                - "reason": Raw LLM response text
        """
        for attempt in range(self.max_retries):
            try:
                result = self._call_judge(question, answer_a, answer_b)
                return self._parse_result(result, answer_a, answer_b)
            except Exception as e:
                logger.warning(
                    "PairwiseJudge attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
        return {"winner": "tie", "reason": "all_retries_exhausted", "score": 0.5}

    def _call_judge(self, question: str, answer_a: str, answer_b: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        prompt = self._prompt_template.format(
            question=question, answer_a=answer_a, answer_b=answer_b
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _parse_result(self, raw: str, answer_a: str, answer_b: str) -> dict:
        text = raw.lower()
        if "winner: a" in text or "answer a" in text:
            return {"winner": "A", "reason": raw, "score": 1.0}
        elif "winner: b" in text or "answer b" in text:
            return {"winner": "B", "reason": raw, "score": 0.0}
        else:
            return {"winner": "tie", "reason": raw, "score": 0.5}


DEFAULT_PAIRWISE_PROMPT = """You are a helpful judge. Given a question and two answers, decide which answer is better.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

First, reason step by step. Then output your decision on a new line as:
Winner: A
or
Winner: B
or
Winner: Tie"""

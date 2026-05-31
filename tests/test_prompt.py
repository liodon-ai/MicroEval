import pytest
from microeval.prompt import PromptVariance


class TestPromptVariance:
    def test_basic_analysis(self):
        def model_fn(prompt):
            return f"Response to: {prompt[:10]}"

        pv = PromptVariance(model_fn=model_fn)
        result = pv.analyze("What is 2+2?", num_variants=3)
        assert result["num_variants"] == 3
        assert result["output_length_min"] > 0
        assert len(result["outputs"]) == 3
        assert len(result["prompts"]) == 3

    def test_custom_templates(self):
        def model_fn(prompt):
            return "42"

        pv = PromptVariance(model_fn=model_fn)
        templates = ["Q: {task}", "A: {task}"]
        result = pv.analyze("test", templates=templates)
        assert result["num_variants"] == 2

    def test_all_failures(self):
        def model_fn(prompt):
            raise ValueError("always fails")

        pv = PromptVariance(model_fn=model_fn)
        result = pv.analyze("test", num_variants=3)
        assert result["num_variants"] == 0
        assert "error" in result

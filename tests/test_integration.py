"""Lightweight integration tests — end-to-end flows with no external API calls."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
from juryeval import (
    eval_classification,
    eval_translation,
    eval_summarization,
    flesch_kincaid,
    PairwiseJudge,
    PointwiseJudge,
    MultiJudgeEnsemble,
    JudgeCalibration,
    PromptVariance,
    bootstrap_ci,
    bootstrap_pvalue,
    win_rate,
    compare_models,
)


JURYEVAL = [sys.executable, "-m", "juryeval.cli"]


class TestPackageImports:
    """Verify all public API symbols import correctly."""

    def test_all_exports_available(self):
        from juryeval import (
            eval_classification,
            eval_translation,
            eval_summarization,
            perplexity,
            flesch_kincaid,
            bert_score,
            semantic_similarity,
            PairwiseJudge,
            PointwiseJudge,
            MultiJudgeEnsemble,
            JudgeCalibration,
            bootstrap_ci,
            bootstrap_pvalue,
            win_rate,
            compare_models,
            PromptVariance,
        )

    def test_submodules_import(self):
        import juryeval.judges
        import juryeval.metrics
        import juryeval.significance
        import juryeval.prompt
        import juryeval.utils
        import juryeval.lmeval

    def test_cli_module_imports(self):
        from juryeval.cli import main, _build_parser, _load_file


class TestCLIIntegration:
    """Run the CLI as a subprocess with sample data files."""

    def test_cli_help(self):
        result = subprocess.run([*JURYEVAL, "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage:" in result.stdout

    def test_cli_evaluate_classification(self):
        preds = json.dumps(["pos", "neg", "pos", "neg", "pos"])
        refs = json.dumps(["pos", "neg", "pos", "pos", "pos"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as pf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as rf:
            pf.write(preds)
            rf.write(refs)
            pred_path = pf.name
            ref_path = rf.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "classification",
                 "--predictions", pred_path, "--references", ref_path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert "accuracy" in data
            assert "f1" in data
            assert data["accuracy"] == 0.8
        finally:
            os.unlink(pred_path)
            os.unlink(ref_path)

    def test_cli_evaluate_classification_jsonl(self):
        preds = "\n".join([
            json.dumps({"prediction": "cat", "reference": "cat"}),
            json.dumps({"prediction": "dog", "reference": "cat"}),
        ])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write(preds)
            path = f.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "classification",
                 "--predictions", path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert data["accuracy"] == 0.5
        finally:
            os.unlink(path)

    def test_cli_evaluate_translation(self):
        preds = json.dumps(["hello world foo bar", "goodbye world foo bar"])
        refs = json.dumps(["hello world foo bar", "goodbye world foo bar"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as pf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as rf:
            pf.write(preds)
            rf.write(refs)
            pred_path = pf.name
            ref_path = rf.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "translation",
                 "--predictions", pred_path, "--references", ref_path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert data["bleu"] == pytest.approx(100.0, abs=1e-6)
        finally:
            os.unlink(pred_path)
            os.unlink(ref_path)

    def test_cli_evaluate_summarization(self):
        preds = json.dumps(["the cat sat on the mat"])
        refs = json.dumps(["the cat sat on the mat"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as pf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as rf:
            pf.write(preds)
            rf.write(refs)
            pred_path = pf.name
            ref_path = rf.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "summarization",
                 "--predictions", pred_path, "--references", ref_path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert data["rouge1"] > 0.9
            assert data["rouge2"] > 0.9
            assert data["rougeL"] > 0.9
        finally:
            os.unlink(pred_path)
            os.unlink(ref_path)

    def test_cli_evaluate_fluency(self):
        preds = json.dumps(["The cat sat on the mat."])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(preds)
            path = f.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "fluency",
                 "--predictions", path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            data = json.loads(result.stdout)
            assert "perplexity" in data
            assert "flesch_kincaid" in data
        finally:
            os.unlink(path)

    def test_cli_evaluate_fluency_text_output(self):
        preds = json.dumps(["The cat sat on the mat."])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(preds)
            path = f.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "fluency",
                 "--predictions", path, "--format", "text"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert "perplexity" in result.stdout or "Perplexity" in result.stdout
        finally:
            os.unlink(path)

    def test_cli_evaluate_output_file(self):
        preds = json.dumps(["pos", "neg"])
        refs = json.dumps(["pos", "pos"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as pf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as rf, tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as of:
            pf.write(preds)
            rf.write(refs)
            pred_path = pf.name
            ref_path = rf.name
            out_path = of.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "classification",
                 "--predictions", pred_path, "--references", ref_path,
                 "--output", out_path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert "Results written to" in result.stdout
            with open(out_path) as f:
                data = json.load(f)
            assert data["accuracy"] == 0.5
        finally:
            os.unlink(pred_path)
            os.unlink(ref_path)
            os.unlink(out_path)

    def test_cli_unknown_metric(self):
        preds = json.dumps(["a"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(preds)
            path = f.name

        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "nonexistent",
                 "--predictions", path],
                capture_output=True, text=True,
            )
            assert result.returncode == 2
        finally:
            os.unlink(path)

    def test_cli_evaluate_classification_no_refs(self):
        preds = json.dumps(["a"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(preds)
            path = f.name
        try:
            result = subprocess.run(
                [*JURYEVAL, "evaluate", "--metric", "classification",
                 "--predictions", path],
                capture_output=True, text=True,
            )
            assert result.returncode == 1
            assert "error:" in result.stderr
        finally:
            os.unlink(path)


class TestMetricsPipeline:
    """End-to-end metric computations with real data."""

    def test_classification_full_pipeline(self):
        preds = ["cat", "dog", "cat", "cat", "dog"]
        refs = ["cat", "dog", "dog", "cat", "cat"]
        result = eval_classification(preds, refs)
        assert result["accuracy"] == 0.6
        assert 0.0 < result["f1"] < 1.0

    def test_translation_full_pipeline(self):
        preds = ["hello world foo bar baz", "goodbye moon foo bar baz"]
        refs = ["hello world foo bar baz", "goodbye moon foo bar baz"]
        result = eval_translation(preds, refs)
        assert result["bleu"] == pytest.approx(100.0, abs=1e-6)

    def test_summarization_full_pipeline(self):
        preds = ["the quick brown fox"]
        refs = ["the quick brown fox"]
        result = eval_summarization(preds, refs)
        assert result["rouge1"] > 0.9
        assert result["rouge2"] > 0.9
        assert result["rougeL"] > 0.9

    def test_fluency_pipeline(self):
        text = "The cat sat on the mat."
        fk = flesch_kincaid(text)
        assert isinstance(fk, float)
        assert fk > 0.0

    def test_significance_pipeline(self):
        a = [1.0] * 30
        b = [2.0] * 30
        result = compare_models(a, b, num_resamples=200, seed=42)
        assert result["win_rate"] == 0.0
        assert result["tie_rate"] == 0.0
        assert result["p_value"] <= 0.05
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_prompt_variance_pipeline(self):
        def model_fn(prompt):
            return "42"

        pv = PromptVariance(model_fn=model_fn)
        result = pv.analyze("What is 2+2?", num_variants=3)
        assert result["num_variants"] == 3
        assert len(result["outputs"]) == 3
        assert len(result["prompts"]) == 3


class TestJudgeIntegration:
    """End-to-end judge flows with monkeypatched _call_judge."""

    def test_pairwise_judge_end_to_end(self):
        judge = PairwiseJudge("test-model", max_retries=1)

        def mock_call(q, a, b):
            return "Winner: A"

        judge._call_judge = mock_call
        result = judge.compare("answer A", "answer B", "test question")
        assert result["winner"] == "A"
        assert result["score"] == 1.0
        assert "Winner: A" in result["reason"]

    def test_pointwise_judge_end_to_end(self):
        judge = PointwiseJudge("test-model", max_retries=1)

        def mock_call(q, a, r):
            return "8/10"

        judge._call_judge = mock_call
        result = judge.score("some answer", "some question")
        assert result["score"] == 0.8
        assert "8/10" in result["reason"]

    def test_pairwise_judge_retry_then_succeed(self):
        judge = PairwiseJudge("test-model", max_retries=3)
        call_count = [0]

        def mock_call(q, a, b):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("transient failure")
            return "Winner: B"

        judge._call_judge = mock_call
        result = judge.compare("A", "B", "q")
        assert result["winner"] == "B"
        assert call_count[0] == 2

    def test_pairwise_judge_all_retries_exhausted(self):
        judge = PairwiseJudge("test-model", max_retries=2)

        def mock_call(q, a, b):
            raise ConnectionError("always fails")

        judge._call_judge = mock_call
        result = judge.compare("A", "B", "q")
        assert result["winner"] == "tie"
        assert result["reason"] == "all_retries_exhausted"

    def test_pointwise_judge_retry_then_succeed(self):
        judge = PointwiseJudge("test-model", max_retries=3)
        call_count = [0]

        def mock_call(q, a, r):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("transient failure")
            return "6/10"

        judge._call_judge = mock_call
        result = judge.score("answer", "question")
        assert result["score"] == 0.6
        assert call_count[0] == 2

    def test_ensemble_with_real_pairwise_judges(self):
        judge_a = PairwiseJudge("m1", max_retries=1)
        judge_a._call_judge = lambda q, a, b: "Winner: A"
        judge_b = PairwiseJudge("m2", max_retries=1)
        judge_b._call_judge = lambda q, a, b: "Winner: A"
        judge_c = PairwiseJudge("m3", max_retries=1)
        judge_c._call_judge = lambda q, a, b: "Winner: B"

        ensemble = MultiJudgeEnsemble([judge_a, judge_b, judge_c])
        result = ensemble.compare("ans_a", "ans_b", "question")
        assert result["majority_winner"] == "A"
        assert result["agreement"] == 2.0 / 3.0
        assert result["num_valid_votes"] == 3

    def test_calibration_with_real_pairwise_judge(self):
        judge = PairwiseJudge("test-model", max_retries=1)
        judge._call_judge = lambda q, a, b: "Winner: A"

        cal = JudgeCalibration()
        report = cal.evaluate(judge, num_samples=5)
        assert "position_bias" in report
        assert "consistency" in report
        assert "length_bias" in report
        assert "self_enhancement_bias" in report


class TestDataLoading:
    """Test _load_file handles various formats correctly."""

    def test_load_json_array(self):
        from juryeval.cli import _load_file
        data = json.dumps(["a", "b", "c"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(data)
            path = f.name
        try:
            result = _load_file(path)
            assert result == ["a", "b", "c"]
        finally:
            os.unlink(path)

    def test_load_jsonl(self):
        from juryeval.cli import _load_file
        data = "\n".join([
            json.dumps({"text": "hello"}),
            json.dumps({"text": "world"}),
        ])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write(data)
            path = f.name
        try:
            result = _load_file(path)
            assert len(result) == 2
            assert result[0]["text"] == "hello"
        finally:
            os.unlink(path)

    def test_load_text_fallback(self):
        from juryeval.cli import _load_file
        data = "hello\nworld\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(data)
            path = f.name
        try:
            result = _load_file(path)
            assert result == ["hello", "world"]
        finally:
            os.unlink(path)


class TestLmevalIntegration:
    """Test lm-eval-harness integration layer."""

    def test_register_all_imports_and_runs(self):
        from juryeval.lmeval import register_all
        register_all()

    def test_register_all_idempotent(self):
        from juryeval.lmeval import register_all
        register_all()
        register_all()

    def test_register_all_sets_flag(self):
        from juryeval.lmeval import register_all
        from juryeval.lmeval.metrics import _registered
        register_all()
        assert _registered is True

    def test_register_all_creates_metrics(self):
        import lm_eval.api.registry as reg
        from juryeval.lmeval import register_all
        register_all()
        assert "pairwise_judge" in reg.metric_registry._objs
        assert "pointwise_judge" in reg.metric_registry._objs

    def test_register_all_aggregations(self):
        import lm_eval.api.registry as reg
        from juryeval.lmeval import register_all
        register_all()
        metric = reg.metric_registry._objs["pairwise_judge"]
        assert callable(metric)

    def test_register_all_handles_missing_lm_eval(self):
        from unittest.mock import patch
        import builtins
        import sys
        import juryeval.lmeval.metrics as m

        m._registered = False
        saved = {}
        for key in list(sys.modules):
            if key == "lm_eval" or key.startswith("lm_eval."):
                saved[key] = sys.modules.pop(key)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "lm_eval" or name.startswith("lm_eval."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                m.register_all()
                assert m._registered is False
        finally:
            sys.modules.update(saved)


class TestSignificanceIntegration:
    """End-to-end significance testing."""

    def test_bootstrap_ci_reproducible(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        r1 = bootstrap_ci(samples, num_resamples=500, seed=42)
        r2 = bootstrap_ci(samples, num_resamples=500, seed=42)
        assert r1["estimate"] == r2["estimate"]
        assert r1["lower"] == r2["lower"]
        assert r1["upper"] == r2["upper"]

    def test_bootstrap_pvalue_identical_samples(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_pvalue(a, b, num_resamples=500, seed=42)
        assert result["observed_diff"] == 0.0
        assert result["p_value"] >= 0.9

    def test_win_rate_basic(self):
        result = win_rate([5.0, 5.0], [3.0, 3.0])
        assert result["win_rate"] == 1.0
        assert result["loss_rate"] == 0.0
        assert result["tie_rate"] == 0.0

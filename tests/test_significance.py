import pytest
from microeval.significance import bootstrap_ci, bootstrap_pvalue, win_rate, compare_models


class TestBootstrapCI:
    def test_basic_ci(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(samples, num_resamples=500, seed=42)
        assert result["estimate"] == 3.0
        assert result["lower"] < result["estimate"] < result["upper"]
        assert result["std_err"] > 0.0

    def test_empty(self):
        result = bootstrap_ci([], num_resamples=100)
        assert result["estimate"] == 0.0
        assert result["lower"] == 0.0
        assert result["upper"] == 0.0

    def test_constant(self):
        samples = [5.0, 5.0, 5.0]
        result = bootstrap_ci(samples, num_resamples=100, seed=42)
        assert result["estimate"] == 5.0
        assert result["std_err"] == 0.0


class TestBootstrapPValue:
    def test_significant_difference(self):
        a = [1.0] * 20
        b = [2.0] * 20
        result = bootstrap_pvalue(a, b, num_resamples=200, seed=42)
        assert result["observed_diff"] == -1.0
        assert result["p_value"] <= 0.01

    def test_no_difference(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        result = bootstrap_pvalue(a, b, num_resamples=200, seed=42)
        assert result["observed_diff"] == 0.0
        assert result["p_value"] >= 0.9

    def test_reproducible_seed(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        r1 = bootstrap_pvalue(a, b, num_resamples=200, seed=42)
        r2 = bootstrap_pvalue(a, b, num_resamples=200, seed=42)
        assert r1["p_value"] == r2["p_value"]


class TestWinRate:
    def test_a_wins_all(self):
        result = win_rate([3.0, 3.0], [1.0, 1.0])
        assert result["win_rate"] == 1.0
        assert result["loss_rate"] == 0.0

    def test_a_loses_all(self):
        result = win_rate([1.0, 1.0], [3.0, 3.0])
        assert result["win_rate"] == 0.0
        assert result["loss_rate"] == 1.0

    def test_all_ties(self):
        result = win_rate([2.0, 2.0], [2.0, 2.0])
        assert result["win_rate"] == 0.0
        assert result["tie_rate"] == 1.0

    def test_lower_is_better(self):
        result = win_rate([1.0, 1.0], [3.0, 3.0], higher_is_better=False)
        assert result["win_rate"] == 1.0


class TestCompareModels:
    def test_basic_comparison(self):
        a = [1.0] * 20
        b = [2.0] * 20
        result = compare_models(a, b, num_resamples=200, seed=42)
        assert result["win_rate"] == 0.0
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "p_value" in result

    def test_equal_models(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compare_models(scores, scores, num_resamples=200, seed=42)
        assert result["win_rate"] == 0.0
        assert result["tie_rate"] == 1.0

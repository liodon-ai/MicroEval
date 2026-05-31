"""Tests for lm-eval-harness integration layer."""

from microeval.lmeval import register_all


class TestLmevalIntegration:
    def test_register_all_runs(self):
        register_all()

    def test_register_all_idempotent(self):
        register_all()
        register_all()

    def test_register_all_metrics_registered(self):
        from microeval.lmeval.metrics import _registered

        register_all()
        assert _registered is True

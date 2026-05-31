import warnings

warnings.warn(
    "microeval.metrics is deprecated. Use microeval.metrics.fluency directly.",
    DeprecationWarning,
    stacklevel=2,
)

from microeval.metrics.fluency import perplexity, flesch_kincaid  # noqa: F401, E402

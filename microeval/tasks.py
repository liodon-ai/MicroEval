import warnings

warnings.warn(
    "microeval.tasks is deprecated. Use microeval.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from microeval.metrics.classification import eval_classification  # noqa: F401, E402
from microeval.metrics.translation import eval_translation  # noqa: F401, E402
from microeval.metrics.summarization import eval_summarization  # noqa: F401, E402

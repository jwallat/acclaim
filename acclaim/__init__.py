"""LLM Attribution Evaluation library."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ._version import __version__
from .evaluate import evaluate
from .batch import evaluate_batch
from .config.load import load_config
from .data_models import (
    Answer,
    Claim,
    AlignedClaim,
    Document,
    SupportLabel,
    SupportResult,
    ClaimResult,
    EvaluationMetadata,
    EvaluationResult,
    BatchExample,
    BatchEvaluationResult,
)

__all__ = [
    "__version__",
    "evaluate",
    "evaluate_batch",
    "BatchExample",
    "BatchEvaluationResult",
    "load_config",
    "Answer",
    "Claim",
    "AlignedClaim",
    "Document",
    "SupportLabel",
    "SupportResult",
    "ClaimResult",
    "EvaluationMetadata",
    "EvaluationResult",
]

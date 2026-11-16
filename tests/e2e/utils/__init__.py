"""Package initialization for tests/e2e/utils.
"""
from tests.e2e.utils.submission_validator import (
    SubmissionValidator,
    ValidationResult,
    ALLOWED_LABELS,
    TOTAL_FRAMES,
    PRE_MATCH_SPAN,
    H1_SPAN,
    HALFTIME_SPAN,
    H2_PREKICK_SPAN,
    H2_SPAN,
    POST_MATCH_SPAN,
)

__all__ = [
    "SubmissionValidator",
    "ValidationResult",
    "ALLOWED_LABELS",
    "TOTAL_FRAMES",
    "PRE_MATCH_SPAN",
    "H1_SPAN",
    "HALFTIME_SPAN",
    "H2_PREKICK_SPAN",
    "H2_SPAN",
    "POST_MATCH_SPAN",
]

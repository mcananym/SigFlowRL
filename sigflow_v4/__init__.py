"""Reusable infrastructure for the SigFlow v4 research runner."""

from .data import (
    DataDownloadError,
    NetworkUnavailableError,
    RateLimitError,
    RetryPolicy,
    request_bytes_with_retry,
)
from .protocol import (
    DEFAULT_RESEARCH_SEEDS,
    RollingOriginWindow,
    collect_reproducibility_manifest,
    make_rolling_origin_windows,
)

__all__ = [
    "DEFAULT_RESEARCH_SEEDS",
    "DataDownloadError",
    "NetworkUnavailableError",
    "RateLimitError",
    "RetryPolicy",
    "RollingOriginWindow",
    "collect_reproducibility_manifest",
    "make_rolling_origin_windows",
    "request_bytes_with_retry",
]

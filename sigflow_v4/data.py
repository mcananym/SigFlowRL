"""Market-data HTTP and cache primitives with explicit failure semantics.

This module deliberately uses only the Python standard library so its retry and
provenance behaviour can be tested even before the scientific stack is present.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import random
import socket
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class DataDownloadError(RuntimeError):
    """A provider responded, but a usable payload could not be obtained."""


class NetworkUnavailableError(DataDownloadError):
    """DNS, connection, or timeout failures exhausted the retry budget."""


class RateLimitError(DataDownloadError):
    """HTTP 429 exhausted the retry budget; this is not an offline signal."""

    def __init__(
        self,
        url: str,
        attempts: int,
        retry_after_seconds: float | None,
    ) -> None:
        retry_note = (
            "provider supplied no Retry-After value"
            if retry_after_seconds is None
            else f"last Retry-After={retry_after_seconds:.3g}s"
        )
        super().__init__(
            f"Market-data provider rate limited {url!r} after {attempts} "
            f"attempts ({retry_note}). Cached data remain usable; retry later."
        )
        self.url = url
        self.attempts = attempts
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff policy for idempotent GET requests."""

    attempts: int = 5
    initial_delay_seconds: float = 1.0
    maximum_delay_seconds: float = 60.0
    jitter_fraction: float = 0.15
    retryable_statuses: frozenset[int] = RETRYABLE_HTTP_STATUSES

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least one")
        if self.initial_delay_seconds < 0.0:
            raise ValueError("initial_delay_seconds cannot be negative")
        if self.maximum_delay_seconds < self.initial_delay_seconds:
            raise ValueError(
                "maximum_delay_seconds must be at least initial_delay_seconds"
            )
        if not 0.0 <= self.jitter_fraction <= 1.0:
            raise ValueError("jitter_fraction must lie in [0, 1]")


def _retry_after_seconds(
    headers: Mapping[str, str] | None,
    now: datetime | None = None,
) -> float | None:
    if not headers:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except (TypeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (parsed - reference).total_seconds())


def _backoff_delay(
    policy: RetryPolicy,
    retry_index: int,
    retry_after_seconds: float | None,
    random_value: float,
) -> float:
    exponential = min(
        policy.maximum_delay_seconds,
        policy.initial_delay_seconds * (2**retry_index),
    )
    multiplier = 1.0 + policy.jitter_fraction * (2.0 * random_value - 1.0)
    jittered_exponential = min(
        policy.maximum_delay_seconds,
        max(0.0, exponential * multiplier),
    )
    # Retry-After is a server-specified minimum. Jitter may lengthen the local
    # exponential delay, but must never cause an earlier retry than requested.
    return max(jittered_exponential, retry_after_seconds or 0.0)


def _network_reason(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout, ConnectionError)):
        return True
    if isinstance(error, urllib.error.URLError):
        return isinstance(
            error.reason,
            (TimeoutError, socket.timeout, ConnectionError, OSError),
        )
    return False


def request_bytes_with_retry(
    url: str,
    timeout_seconds: float,
    policy: RetryPolicy | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    opener: Callable[..., object] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    random_source: random.Random | None = None,
    on_retry: Callable[[int, int | None, float, str], None] | None = None,
) -> bytes:
    """Fetch bytes and distinguish rate limiting from connectivity failures.

    ``opener``, ``sleep`` and ``random_source`` are injectable to keep tests
    deterministic. The operation is an idempotent GET, so retrying these status
    codes is safe.
    """

    active_policy = policy or RetryPolicy()
    rng = random_source or random.Random()
    request_headers = {
        "User-Agent": "SigFlow-v4-research/1.0 (+cached market-data research)",
        "Accept": "application/json,text/csv,*/*",
        **dict(headers or {}),
    }
    last_error: BaseException | None = None
    last_retry_after: float | None = None

    for attempt in range(1, active_policy.attempts + 1):
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        try:
            response = opener(request, timeout=timeout_seconds)
            with response:
                payload = response.read()
            if not payload:
                raise DataDownloadError(f"Provider returned an empty payload for {url!r}.")
            return payload
        except urllib.error.HTTPError as error:
            last_error = error
            status = int(error.code)
            last_retry_after = _retry_after_seconds(error.headers)
            error.close()
            if status not in active_policy.retryable_statuses:
                raise DataDownloadError(
                    f"Provider returned non-retryable HTTP {status} for {url!r}."
                ) from error
            if (
                last_retry_after is not None
                and last_retry_after > active_policy.maximum_delay_seconds
            ):
                if status == 429:
                    raise RateLimitError(url, attempt, last_retry_after) from error
                raise DataDownloadError(
                    f"Provider returned HTTP {status} for {url!r} with "
                    f"Retry-After={last_retry_after:.3g}s, beyond the configured "
                    f"maximum wait of {active_policy.maximum_delay_seconds:.3g}s."
                ) from error
            if attempt == active_policy.attempts:
                if status == 429:
                    raise RateLimitError(url, attempt, last_retry_after) from error
                raise DataDownloadError(
                    f"Provider returned HTTP {status} for {url!r} on all "
                    f"{attempt} attempts."
                ) from error
            delay = _backoff_delay(
                active_policy,
                retry_index=attempt - 1,
                retry_after_seconds=last_retry_after,
                random_value=rng.random(),
            )
            if on_retry:
                on_retry(attempt, status, delay, "http")
            sleep(delay)
        except DataDownloadError:
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error
            if attempt == active_policy.attempts:
                if _network_reason(error):
                    raise NetworkUnavailableError(
                        f"Network access to {url!r} failed on all {attempt} attempts: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                raise DataDownloadError(
                    f"Request for {url!r} failed on all {attempt} attempts: {error}"
                ) from error
            delay = _backoff_delay(
                active_policy,
                retry_index=attempt - 1,
                retry_after_seconds=None,
                random_value=rng.random(),
            )
            if on_retry:
                on_retry(attempt, None, delay, "network")
            sleep(delay)

    raise AssertionError(f"Unreachable retry state: {last_error!r}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes after an atomic rename on POSIX."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically replace a JSON artifact without exposing a partial cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def cache_metadata(
    *,
    symbol: str,
    provider: str,
    source_url: str,
    start_date: str,
    end_date: str,
    row_count: int,
    first_date: str,
    last_date: str,
    csv_payload: bytes,
    retry_policy: RetryPolicy,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "source_kind": "real_market",
        "symbol": symbol,
        "provider": provider,
        "source_url": source_url,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "row_count": row_count,
        "first_date": first_date,
        "last_date": last_date,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": sha256_bytes(csv_payload),
        "retry_policy": {
            **asdict(retry_policy),
            "retryable_statuses": sorted(retry_policy.retryable_statuses),
        },
    }

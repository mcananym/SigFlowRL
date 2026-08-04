from __future__ import annotations

import socket
import unittest
import urllib.error
from email.message import Message

from sigflow_v4.data import (
    DataDownloadError,
    NetworkUnavailableError,
    RateLimitError,
    RetryPolicy,
    request_bytes_with_retry,
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.payload


class _Random:
    def random(self) -> float:
        return 0.5


class _MinimumRandom:
    def random(self) -> float:
        return 0.0


def _http_error(status: int, retry_after: str | None = None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://example.test/data", status, "test", headers, None
    )


class RetryTests(unittest.TestCase):
    def test_429_uses_retry_after_then_succeeds(self) -> None:
        outcomes = [_http_error(429, "2"), _http_error(429, "2"), _Response(b"ok")]
        delays: list[float] = []

        def opener(*_, **__):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        payload = request_bytes_with_retry(
            "https://example.test/data",
            1,
            RetryPolicy(
                attempts=3,
                initial_delay_seconds=1,
                maximum_delay_seconds=10,
                jitter_fraction=0,
            ),
            opener=opener,
            sleep=delays.append,
            random_source=_Random(),
        )
        self.assertEqual(payload, b"ok")
        self.assertEqual(delays, [2.0, 2.0])

    def test_exhausted_429_is_not_called_internet_unavailable(self) -> None:
        calls = 0

        def opener(*_, **__):
            nonlocal calls
            calls += 1
            raise _http_error(429, "0")

        with self.assertRaises(RateLimitError) as caught:
            request_bytes_with_retry(
                "https://example.test/data",
                1,
                RetryPolicy(
                    attempts=2,
                    initial_delay_seconds=0,
                    maximum_delay_seconds=0,
                    jitter_fraction=0,
                ),
                opener=opener,
                sleep=lambda _: None,
                random_source=_Random(),
            )
        self.assertEqual(calls, 2)
        message = str(caught.exception).lower()
        self.assertIn("rate limited", message)
        self.assertNotIn("internet unavailable", message)

    def test_negative_jitter_never_shortens_retry_after(self) -> None:
        outcomes = [_http_error(429, "2"), _Response(b"ok")]
        delays: list[float] = []

        def opener(*_, **__):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        request_bytes_with_retry(
            "https://example.test/data",
            1,
            RetryPolicy(
                attempts=2,
                initial_delay_seconds=1,
                maximum_delay_seconds=10,
                jitter_fraction=1,
            ),
            opener=opener,
            sleep=delays.append,
            random_source=_MinimumRandom(),
        )
        self.assertEqual(delays, [2.0])

    def test_oversized_retry_after_fails_truthfully_without_early_retry(self) -> None:
        calls = 0

        def opener(*_, **__):
            nonlocal calls
            calls += 1
            raise _http_error(429, "120")

        with self.assertRaises(RateLimitError) as caught:
            request_bytes_with_retry(
                "https://example.test/data",
                1,
                RetryPolicy(
                    attempts=5,
                    initial_delay_seconds=1,
                    maximum_delay_seconds=60,
                ),
                opener=opener,
                sleep=lambda _: self.fail("must not sleep less than Retry-After"),
            )
        self.assertEqual(calls, 1)
        self.assertEqual(caught.exception.retry_after_seconds, 120.0)

    def test_transient_http_errors_use_exponential_backoff(self) -> None:
        outcomes = [_http_error(503), _http_error(503), _Response(b"ok")]
        delays: list[float] = []

        def opener(*_, **__):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        request_bytes_with_retry(
            "https://example.test/data",
            1,
            RetryPolicy(
                attempts=3,
                initial_delay_seconds=1,
                maximum_delay_seconds=10,
                jitter_fraction=0,
            ),
            opener=opener,
            sleep=delays.append,
            random_source=_Random(),
        )
        self.assertEqual(delays, [1.0, 2.0])

    def test_404_is_not_retried(self) -> None:
        calls = 0

        def opener(*_, **__):
            nonlocal calls
            calls += 1
            raise _http_error(404)

        with self.assertRaises(DataDownloadError):
            request_bytes_with_retry(
                "https://example.test/data",
                1,
                RetryPolicy(attempts=5),
                opener=opener,
            )
        self.assertEqual(calls, 1)

    def test_network_failure_has_distinct_type(self) -> None:
        def opener(*_, **__):
            raise urllib.error.URLError(socket.gaierror("dns failed"))

        with self.assertRaises(NetworkUnavailableError):
            request_bytes_with_retry(
                "https://example.test/data",
                1,
                RetryPolicy(
                    attempts=2,
                    initial_delay_seconds=0,
                    maximum_delay_seconds=0,
                ),
                opener=opener,
                sleep=lambda _: None,
                random_source=_Random(),
            )


if __name__ == "__main__":
    unittest.main()

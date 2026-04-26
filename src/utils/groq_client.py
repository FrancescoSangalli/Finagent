"""Groq LLM client with rate limiting and retry logic."""

import sys
import time
import threading
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, before_log, RetryError
import logging

logger = logging.getLogger(__name__)


class GroqRateLimiter:
    """Enforces a maximum request rate to stay within Groq free-tier limits."""

    def __init__(self, max_rpm: int = 28) -> None:
        """Initialize the rate limiter with a requests-per-minute cap."""
        self.max_rpm = max_rpm
        self.min_interval = 60.0 / max_rpm
        self._lock = threading.Lock()
        self._last_call_time: Optional[float] = None

    def wait_if_needed(self) -> None:
        """Sleep if the minimum interval since the last call has not elapsed."""
        with self._lock:
            now = time.monotonic()
            if self._last_call_time is not None:
                elapsed = now - self._last_call_time
                sleep_for = self.min_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._last_call_time = time.monotonic()


_rate_limiter = GroqRateLimiter(max_rpm=28)


def call_with_retry(llm, messages):
    """Invoke the LLM with exponential backoff retry on failure."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        before=before_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call():
        _rate_limiter.wait_if_needed()
        return llm.invoke(messages)

    return _call()


def get_groq_llm():
    """Return a ChatGroq instance configured for LLaMA 3.3 70B."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

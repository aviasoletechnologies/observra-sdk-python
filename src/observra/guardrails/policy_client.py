"""Fetches org/environment guardrail policy from the gateway, cached with a TTL.

Never hard-fails the client because the policy fetch failed — falls back to
the built-in pattern set (``observra.guardrails.patterns.BUILTIN_PATTERNS``)
on any error, per the doc's explicit deliverable for this module.
"""

from __future__ import annotations

import logging
import re
import time

import httpx

from observra.guardrails.patterns import BUILTIN_PATTERNS, GuardrailPattern

logger = logging.getLogger("observra")

_DEFAULT_TTL_SECONDS = 300.0
_POLICY_PATH = "/v1/guardrails/policy"


class PolicyClient:
    def __init__(
        self,
        gateway_url: str,
        gateway_key: str,
        *,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        timeout: float = 3.0,
    ) -> None:
        self._endpoint = gateway_url.rstrip("/") + _POLICY_PATH
        self._gateway_key = gateway_key
        self._ttl_seconds = ttl_seconds
        self._timeout = timeout
        self._cached_patterns: list[GuardrailPattern] | None = None
        self._cached_at: float = 0.0

    def get_patterns(self) -> list[GuardrailPattern]:
        now = time.monotonic()
        if self._cached_patterns is not None and (now - self._cached_at) < self._ttl_seconds:
            return self._cached_patterns

        fetched = self._fetch()
        self._cached_patterns = fetched if fetched is not None else BUILTIN_PATTERNS
        self._cached_at = now
        return self._cached_patterns

    def _fetch(self) -> list[GuardrailPattern] | None:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    self._endpoint,
                    headers={"x-gateway-key": self._gateway_key},
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            logger.warning(
                "observra: guardrail policy fetch failed, falling back to built-in patterns",
                exc_info=True,
            )
            return None

        try:
            patterns = [
                GuardrailPattern(item["name"], re.compile(item["pattern"]))
                for item in data.get("patterns", [])
            ]
            return patterns or None
        except Exception:
            logger.warning(
                "observra: guardrail policy payload malformed, falling back to built-in patterns",
                exc_info=True,
            )
            return None

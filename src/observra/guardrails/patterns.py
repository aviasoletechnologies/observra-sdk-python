"""Built-in guardrail patterns, shipped as data — easy to extend or override.

Order matters: more specific patterns are listed before broader ones so a
credit-card-shaped string doesn't first get swallowed by the generic token
pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailPattern:
    name: str
    regex: "re.Pattern[str]"


BUILTIN_PATTERNS = [
    GuardrailPattern("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    GuardrailPattern(
        "credit_card",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    ),
    GuardrailPattern("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    GuardrailPattern(
        "phone",
        re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    GuardrailPattern("api_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    GuardrailPattern("generic_token", re.compile(r"\b(?:obs|ghp|gho|xox)[A-Za-z0-9_\-]{20,}\b")),
]

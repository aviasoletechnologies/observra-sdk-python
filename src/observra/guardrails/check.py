"""Guardrail payload checking: block / redact / warn.

Bounded cost per payload (Architecture requirement #3): only the first
``MAX_SCAN_CHARS`` characters of a payload are scanned, so a pathological
huge prompt/response can't stall the request path. The remainder is passed
through unscanned and unredacted in that case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from observra.guardrails.patterns import BUILTIN_PATTERNS, GuardrailPattern

MAX_SCAN_CHARS = 50_000

VALID_MODES = ("block", "redact", "warn")


@dataclass(frozen=True)
class Violation:
    pattern_name: str
    start: int
    end: int
    matched_text: str


class GuardrailViolation(Exception):
    """Raised by :func:`check_payload` when called with ``mode="block"`` and a violation is found.

    The SDK's own provider transport never uses ``block`` mode (guardrail
    checking there is fixed to scan-and-record-only, see
    ``providers/base.py``) — this is for direct programmatic use of
    ``check_payload`` outside that path.
    """

    def __init__(self, violations: Sequence[Violation]) -> None:
        self.violations = list(violations)
        names = ", ".join(sorted({v.pattern_name for v in violations}))
        super().__init__(f"Guardrail violation detected: {names}")


@dataclass
class CheckResult:
    violations: list[Violation] = field(default_factory=list)
    redacted_payload: str = ""

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


def _redact(text: str, violations: Sequence[Violation]) -> str:
    if not violations:
        return text

    ordered = sorted(violations, key=lambda v: v.start)
    parts = []
    cursor = 0
    for v in ordered:
        if v.start < cursor:
            continue  # overlapping match, already covered
        parts.append(text[cursor : v.start])
        parts.append(f"[REDACTED:{v.pattern_name}]")
        cursor = v.end
    parts.append(text[cursor:])
    return "".join(parts)


def check_payload(
    payload: str,
    mode: str = "warn",
    *,
    patterns: Sequence[GuardrailPattern] | None = None,
) -> CheckResult:
    """Scan ``payload`` for guardrail pattern matches and act per ``mode``.

    - ``block``: raises ``GuardrailViolation`` if any match is found, before
      any network call is made.
    - ``redact``: matched spans are masked in the returned payload; the
      caller sends the redacted version upstream.
    - ``warn``: violations are reported but the payload is returned
      unchanged (caller records them as span events without altering
      behavior).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"guardrail mode must be one of {VALID_MODES}, got {mode!r}")

    text = payload if isinstance(payload, str) else str(payload)
    active_patterns = patterns if patterns is not None else BUILTIN_PATTERNS

    scanned, remainder = text[:MAX_SCAN_CHARS], text[MAX_SCAN_CHARS:]

    violations: list[Violation] = []
    for pattern in active_patterns:
        for match in pattern.regex.finditer(scanned):
            violations.append(Violation(pattern.name, match.start(), match.end(), match.group(0)))

    if mode == "block" and violations:
        raise GuardrailViolation(violations)

    if mode == "redact" and violations:
        return CheckResult(violations=violations, redacted_payload=_redact(scanned, violations) + remainder)

    return CheckResult(violations=violations, redacted_payload=text)

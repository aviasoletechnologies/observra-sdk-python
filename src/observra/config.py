"""Config singleton for the Observra SDK.

Read-only after ``configure()`` (Architecture requirement #6): callers get an
immutable ``ObservraConfig`` back and may hold their own reference for
multi-tenant use instead of relying solely on the module-level singleton.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger("observra")

_ENV_GATEWAY_URL = "OBSERVRA_GATEWAY_URL"
_ENV_GATEWAY_KEY = "OBSERVRA_GATEWAY_KEY"

# Production gateway — users never need to pass gateway_url themselves.
# Override via configure(gateway_url=...) or OBSERVRA_GATEWAY_URL for local
# dev (e.g. "http://localhost:8787", combined with insecure=True).
DEFAULT_GATEWAY_URL = "https://gateway.observra.in"


class ObservraConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class ObservraConfig(BaseModel):
    """Immutable configuration resolved by ``configure()``.

    Never log or serialize ``gateway_key`` in full — ``__repr__``/``__str__``
    mask it (Architecture requirement #5: secrets handling).
    """

    model_config = ConfigDict(frozen=True)

    gateway_url: str = DEFAULT_GATEWAY_URL
    gateway_key: str
    insecure: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "ObservraConfig":
        if not self.gateway_url:
            raise ObservraConfigError("gateway_url is required")
        if not self.gateway_key:
            raise ObservraConfigError("gateway_key is required")
        if self.gateway_url.startswith("http://") and not self.insecure:
            raise ObservraConfigError(
                "plaintext http:// gateway_url is refused in production mode; "
                "pass insecure=True to configure() only for local development"
            )
        return self

    def _masked_key(self) -> str:
        if len(self.gateway_key) <= 8:
            return "***"
        return f"{self.gateway_key[:4]}...{self.gateway_key[-2:]}"

    def __repr__(self) -> str:
        return f"ObservraConfig(gateway_url={self.gateway_url!r}, gateway_key={self._masked_key()!r})"

    __str__ = __repr__


_lock = threading.Lock()
_singleton: Optional[ObservraConfig] = None


def configure(
    *,
    gateway_url: Optional[str] = None,
    gateway_key: Optional[str] = None,
    insecure: bool = False,
) -> ObservraConfig:
    """Build and install the module-level config singleton.

    ``gateway_url`` defaults to the production gateway (:data:`DEFAULT_GATEWAY_URL`)
    — not required. Only ``gateway_key`` must be provided (directly, or via
    the ``OBSERVRA_GATEWAY_KEY`` env var). Override ``gateway_url`` for local
    dev (e.g. ``http://localhost:8787`` with ``insecure=True``) via this
    kwarg or the ``OBSERVRA_GATEWAY_URL`` env var. The returned
    ``ObservraConfig`` can also be held directly and passed as ``config=`` to
    provider clients — useful for a multi-tenant process that needs more
    than one config live at once (requirement #6), instead of only ever
    reading the global singleton.
    """
    global _singleton

    resolved = ObservraConfig(
        gateway_url=gateway_url or os.environ.get(_ENV_GATEWAY_URL) or DEFAULT_GATEWAY_URL,
        gateway_key=gateway_key or os.environ.get(_ENV_GATEWAY_KEY, ""),
        insecure=insecure,
    )

    with _lock:
        _singleton = resolved

    _auto_patch_installed_providers()

    return resolved


def _auto_patch_installed_providers() -> None:
    """Patch whichever supported LLM provider SDKs are installed, so plain client code routes through the gateway.

    Import is local (not module-level) to avoid a circular import — provider
    patch modules import from this module. Never lets a patching failure
    break ``configure()`` itself.
    """
    try:
        from observra.providers.registry import auto_patch_providers

        auto_patch_providers()
    except Exception:  # noqa: BLE001
        logger.warning("observra: provider auto-patch failed", exc_info=True)


def get_config() -> ObservraConfig:
    """Return the active config, resolving from env vars if ``configure()`` was never called.

    Fails fast with ``ObservraConfigError`` (not a later, harder-to-trace
    error on first API call) when nothing is resolvable.
    """
    with _lock:
        current = _singleton

    if current is not None:
        return current

    if os.environ.get(_ENV_GATEWAY_KEY):
        return configure()

    raise ObservraConfigError(
        "Observra is not configured. Call observra.configure(gateway_key=...) "
        f"or set {_ENV_GATEWAY_KEY} (and optionally {_ENV_GATEWAY_URL}) environment variables "
        "before instantiating a provider client."
    )


def reset_config() -> None:
    """Clear the singleton. Test-only helper, not part of the public API."""
    global _singleton
    with _lock:
        _singleton = None

"""Auto-detects installed provider SDKs and patches each once, for zero-touch gateway routing + tracing.

Called by ``observra.configure()`` and ``observra.instrument()``. It patches
native provider SDKs (currently ``google-genai``) and global ``httpx``
constructors so direct clients, framework-owned clients, and raw requests
route through the gateway and are traced.
"""

from __future__ import annotations

import importlib
import logging
import threading

from observra._optional import module_available

logger = logging.getLogger("observra")

# import name to probe for -> observra patch module
_SUPPORTED_PROVIDERS = {
    "google.genai": "observra.providers.gemini",
}

_patched_providers: set[str] = set()
_lock = threading.Lock()


def auto_patch_providers() -> None:
    """Patch every installed, supported provider SDK, plus the global httpx patch. Safe to call more than once."""
    for probe_module, patch_module in _SUPPORTED_PROVIDERS.items():
        _maybe_patch(probe_module, patch_module)
    _maybe_patch_http()


def _maybe_patch_http() -> None:
    """``httpx`` is a hard dependency (always installed) — this patch always applies, no probe needed."""
    with _lock:
        if "httpx" in _patched_providers:
            return
        try:
            from observra.providers import httpx_patch

            httpx_patch.patch()
            _patched_providers.add("httpx")
        except Exception:
            logger.warning("observra: failed to auto-patch httpx for gateway routing", exc_info=True)


def _maybe_patch(probe_module: str, patch_module: str) -> None:
    with _lock:
        if probe_module in _patched_providers:
            return
        if not module_available(probe_module):
            return
        try:
            mod = importlib.import_module(patch_module)
            mod.patch()
            _patched_providers.add(probe_module)
        except Exception:
            logger.warning("observra: failed to auto-patch provider %r", probe_module, exc_info=True)

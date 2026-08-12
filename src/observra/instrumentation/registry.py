"""``observra.instrument()`` — activate provider, HTTPX, and agent-framework instrumentation.

Provider native-SDK and raw-HTTP interception are installed first. Installed agent
frameworks are then patched for agent, chain, LLM, and tool lifecycle spans.
Every patch is idempotent and failure-isolated.
"""

from __future__ import annotations

import importlib
import logging
import threading

from observra._optional import module_available

logger = logging.getLogger("observra")

# module import name -> observra instrumentation module name
_SUPPORTED_FRAMEWORKS = {
    "langchain": "observra.instrumentation.langchain",
    "langchain_core": "observra.instrumentation.langchain",
    "llama_index": "observra.instrumentation.llama_index",
    "crewai": "observra.instrumentation.crewai",
    "langgraph": "observra.instrumentation.langchain",
    "semantic_kernel": "observra.instrumentation.semantic_kernel",
}

_instrumented: set[str] = set()
_lock = threading.Lock()


def instrument() -> None:
    """Activate all installed Observra integrations.

    Enables native provider SDK patches and known-host ``httpx`` interception
    first, then patches installed agent frameworks. ``configure()`` is still
    required to provide gateway settings. Safe to call more than once.
    """
    try:
        from observra.providers.registry import auto_patch_providers

        auto_patch_providers()
    except Exception:
        logger.warning("observra: failed to instrument provider SDKs and httpx", exc_info=True)

    for framework_module, instrumentation_module in _SUPPORTED_FRAMEWORKS.items():
        _maybe_instrument(framework_module, instrumentation_module)


def _maybe_instrument(framework_module: str, instrumentation_module: str) -> None:
    with _lock:
        if framework_module in _instrumented:
            return
        if not module_available(framework_module):
            return
        try:
            mod = importlib.import_module(instrumentation_module)
            mod.patch()
            _instrumented.add(framework_module)
        except Exception:
            logger.warning("observra: failed to instrument %r", framework_module, exc_info=True)

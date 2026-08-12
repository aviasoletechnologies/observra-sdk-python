"""Safe discovery helpers for optional third-party integrations."""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger("observra")


def module_available(module_name: str) -> bool:
    """Return whether an optional module can be discovered without importing it.

    Optional integrations must never prevent core SDK configuration. Dotted
    module probes can raise when a parent namespace is absent, has broken
    metadata, or is provided by a faulty import hook; all such probe failures
    are treated as unavailable packages.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        logger.debug("observra: optional module probe failed for %r", module_name, exc_info=True)
        return False

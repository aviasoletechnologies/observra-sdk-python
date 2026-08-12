"""Observra Python SDK.

Public surface (strict semver, Architecture requirement #8):

    import observra

    observra.configure(gateway_key=...)
    observra.instrument()  # optional, only if using an agent framework

That's it — no wrapper clients. Configuring observra transparently patches
supported LLM provider SDKs (currently ``google-genai``) plus, at the
``httpx`` layer, any request to a known provider host (Gemini, OpenAI,
Anthropic) regardless of which library made it — direct SDK usage, a
framework's own integration (e.g. LangChain's ``ChatGoogleGenerativeAI``),
or raw hand-built HTTP calls. All of it routes through the gateway and gets
traced/guardrailed with zero further changes.

Everything else — ``observra.tracing.*``, ``observra.guardrails.*``,
``observra.providers.*`` internals — is private and may change between
minor versions.
"""

from observra.config import ObservraConfig, configure
from observra.gateway import gateway_headers, gateway_url
from observra.guardrails.check import GuardrailViolation
from observra.instrumentation.registry import instrument

__version__ = "0.1.0"

__all__ = [
    "GuardrailViolation",
    "ObservraConfig",
    "__version__",
    "configure",
    "gateway_headers",
    "gateway_url",
    "instrument",
]

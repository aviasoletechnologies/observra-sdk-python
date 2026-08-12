"""Public helpers for hand-rolled HTTP calls to the gateway — no provider SDK involved at all.

Most integrations never need these: configuring observra already
transparently patches supported provider SDKs (``providers/registry.py``),
so plain ``google.genai.Client`` code routes through the gateway
automatically. These exist for the case where there's no provider SDK in
the loop and the caller builds the HTTP request by hand — so the request
target and auth headers still come from the one place config lives,
instead of being hardcoded at every call site.
"""

from __future__ import annotations

from observra.config import get_config

# Header a given provider expects its own API key under, when forwarded
# through the gateway. Falls back to a generic header for unknown providers.
_PROVIDER_KEY_HEADER = {
    "gemini": "x-goog-api-key",
}
_DEFAULT_PROVIDER_KEY_HEADER = "X-Provider-Key"


def gateway_url(provider: str, path: str = "") -> str:
    """Build ``{gateway_url}/{provider}{path}`` from the active config."""
    config = get_config()
    base = config.gateway_url.rstrip("/")
    if path and not path.startswith("/"):
        path = f"/{path}"
    return f"{base}/{provider}{path}"


def gateway_headers(provider: str, *, provider_key: str | None = None) -> dict[str, str]:
    """Build the headers the gateway expects for a request to ``provider``.

    Always includes ``x-gateway-key``. Pass ``provider_key`` (the actual
    provider's own API key, e.g. a Gemini key) to also include it under
    whichever header that provider expects it forwarded under.
    """
    config = get_config()
    headers = {"x-gateway-key": config.gateway_key}
    if provider_key is not None:
        header_name = _PROVIDER_KEY_HEADER.get(provider, _DEFAULT_PROVIDER_KEY_HEADER)
        headers[header_name] = provider_key
    return headers

"""Shared, provider-agnostic transport plumbing (Step 4).

``ObservraTransport`` is an ``httpx.BaseTransport`` that every provider client
composes into its underlying SDK's HTTP client. It knows nothing about any
specific provider's request/response shape — provider modules (e.g.
``providers/gemini.py``) only supply small extractor callables for pulling a
model name / input text / output text / token usage out of a request or
response body. Everything else (routing to the gateway, span lifecycle,
guardrail checks, traceparent injection) lives here exactly once.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple

import httpx

from observra.config import ObservraConfig
from observra.guardrails.check import GuardrailViolation, check_payload
from observra.tracing.context import active_framework_llm_span, inject_traceparent
from observra.tracing.conventions import Attr, SpanKind
from observra.tracing.tracer import get_tracer, safe_set_attributes

logger = logging.getLogger("observra")

# Caps how much request/response text ever lands on a span attribute — keeps
# span payload size bounded independent of the guardrail scan cap.
MAX_ATTR_TEXT_LEN = 4000

# Guardrail mode is fixed, not user-configurable: scan and record violations
# as span events, never block or mutate the payload.
GUARDRAIL_MODE = "warn"

ExtractModelName = Callable[[str], Optional[str]]
ExtractText = Callable[[str], Optional[str]]
ExtractUsage = Callable[[str], Tuple[Optional[int], Optional[int]]]


def _default_extract_model_name(_body_text: str) -> Optional[str]:
    return None


def _default_extract_text(body_text: str) -> Optional[str]:
    if not body_text:
        return None
    return body_text[:MAX_ATTR_TEXT_LEN]


def _default_extract_usage(_body_text: str) -> Tuple[Optional[int], Optional[int]]:
    return None, None


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


class _ObservraTransportCore:
    """Shared logic for the sync and async transports — everything except the actual I/O call.

    Split out so ``ObservraTransport`` (sync) and ``AsyncObservraTransport``
    only differ in how they invoke the inner transport; guardrail/span/routing
    logic lives here exactly once.
    """

    def __init__(
        self,
        config: ObservraConfig,
        *,
        span_name: str = "llm.generate_content",
        span_kind: str = SpanKind.LLM,
        provider_name: str = "unknown",
        gateway_route: Optional[str] = None,
        extract_model_name: Optional[ExtractModelName] = None,
        extract_input_text: Optional[ExtractText] = None,
        extract_output_text: Optional[ExtractText] = None,
        extract_usage: Optional[ExtractUsage] = None,
        on_response_body: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        self._config = config
        self._tracer = get_tracer(config)
        self._span_name = span_name
        self._span_kind = span_kind
        self._provider_name = provider_name
        self._gateway_route = gateway_route or provider_name
        self._extract_model_name = extract_model_name or _default_extract_model_name
        self._extract_input_text = extract_input_text or _default_extract_text
        self._extract_output_text = extract_output_text or _default_extract_text
        self._extract_usage = extract_usage or _default_extract_usage
        self._on_response_body = on_response_body

    def _span_scope(self) -> Any:
        """Reuse active framework LLM span; otherwise create provider transport span."""
        framework_span = active_framework_llm_span()
        if framework_span is not None:
            return nullcontext(framework_span)
        return self._tracer.start_span(self._span_name, self._span_kind)

    def _rewrite_to_gateway(self, request: httpx.Request) -> httpx.Request:
        """Send the exact same payload to ``{gateway_url}/{gateway_route}{original_path}``.

        The gateway is a reverse proxy: it needs the provider name as a path
        segment to know which upstream to call (``http://localhost:8787/gemini``),
        but still expects the original provider REST path after that
        (``/v1beta/models/gemini-2.0-flash:generateContent``) and the original
        query string — only the scheme/host/port are replaced with the
        gateway's, and the provider segment is inserted at the front of the
        path. The request body is forwarded unchanged.
        """
        base = self._config.gateway_url.rstrip("/")
        original_path = request.url.raw_path.decode("ascii")
        # ``raw_path`` includes both path and query, e.g. ``/v1beta/models/x?y=1``.
        new_url = httpx.URL(f"{base}/{self._gateway_route}{original_path}")
        headers = httpx.Headers(request.headers)
        headers["x-gateway-key"] = self._config.gateway_key
        if "x-provider-key" not in headers:
            scheme, _, provider_key = headers.get("authorization", "").partition(" ")
            if scheme.lower() == "bearer" and provider_key.strip():
                headers["x-provider-key"] = provider_key.strip()
                del headers["authorization"]
        try:
            inject_traceparent(headers)
        except Exception:  # noqa: BLE001 - tracing must never block the call
            logger.warning("observra: failed to inject traceparent", exc_info=True)

        return httpx.Request(
            method=request.method,
            url=new_url,
            headers=headers,
            content=request.content,
        )

    def _apply_guardrail(self, span: Any, text: str) -> str:
        """Scan for guardrail violations; never block or alter the payload."""
        try:
            result = check_payload(text, GUARDRAIL_MODE)
        except GuardrailViolation:
            raise  # deliberate product-decision exception (requirement #1) — must propagate
        except Exception:  # noqa: BLE001
            logger.warning(
                "observra: guardrail check failed, passing payload through unchanged",
                exc_info=True,
            )
            return text

        if result.has_violations:
            violation_names = ",".join(sorted({v.pattern_name for v in result.violations}))
            safe_set_attributes(
                span,
                {
                    Attr.GUARDRAIL_VIOLATION: violation_names,
                    Attr.GUARDRAIL_ACTION: GUARDRAIL_MODE,
                },
            )

        return text

    def _prepare_request(self, request: httpx.Request, span: Any) -> httpx.Request:
        """Rewrite to gateway (already done by caller) + apply guardrail + record request attrs."""
        raw_body = request.content or b""
        body_text = _decode(raw_body)

        guardrailed_body = self._apply_guardrail(span, body_text)  # may raise GuardrailViolation
        if guardrailed_body != body_text:
            request = httpx.Request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                content=guardrailed_body.encode("utf-8"),
            )
            body_text = guardrailed_body

        try:
            safe_set_attributes(
                span,
                {
                    Attr.LLM_PROVIDER: self._provider_name,
                    "observra.gateway_route": self._gateway_route,
                    Attr.LLM_MODEL_NAME: self._extract_model_name(body_text),
                    Attr.INPUT_VALUE: self._extract_input_text(body_text),
                },
            )
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to record request attributes", exc_info=True)

        return request

    def _finalize_response(
        self, response: httpx.Response, request: httpx.Request, span: Any, start: float
    ) -> httpx.Response:
        try:
            resp_text = _decode(response.content)
            out_text = self._apply_guardrail(span, resp_text)  # may raise GuardrailViolation

            prompt_tokens, completion_tokens = self._extract_usage(resp_text)
            latency_ms = (time.monotonic() - start) * 1000
            safe_set_attributes(
                span,
                {
                    Attr.OUTPUT_VALUE: self._extract_output_text(out_text),
                    Attr.LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                    Attr.LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
                    Attr.LLM_LATENCY_MS: latency_ms,
                },
            )

            if out_text != resp_text:
                response = httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=out_text.encode("utf-8"),
                    request=request,
                )

            if self._on_response_body is not None:
                try:
                    self._on_response_body(out_text, span)
                except Exception:  # noqa: BLE001
                    logger.warning("observra: on_response_body hook raised", exc_info=True)
        except GuardrailViolation:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to record response attributes", exc_info=True)

        return response


class ObservraTransport(_ObservraTransportCore, httpx.BaseTransport):
    """Sync provider-agnostic ``httpx`` transport: routes to the gateway, traces, guardrails.

    Provider modules never subclass this for behavior changes — they pass
    extractor callables in. See Architecture requirement #1: guardrail/span
    machinery here is fail-open (wrapped in try/except and logged), except
    the deliberate ``GuardrailViolation`` raise in ``block`` mode, and except
    the real network call itself — a gateway-unreachable error is a genuine
    failure the caller must see, never swallowed.
    """

    def __init__(
        self,
        config: ObservraConfig,
        *,
        inner: Optional[httpx.BaseTransport] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request = self._rewrite_to_gateway(request)
        start = time.monotonic()

        with self._span_scope() as span:
            request = self._prepare_request(request, span)

            # The actual LLM call. Left outside any try/except that would
            # swallow it — a gateway-unreachable error here is real and must
            # reach the caller (requirement #1).
            response = self._inner.handle_request(request)
            response.read()

            return self._finalize_response(response, request, span, start)

    def close(self) -> None:
        try:
            self._inner.close()
        except Exception:  # noqa: BLE001
            pass


class AsyncObservraTransport(_ObservraTransportCore, httpx.AsyncBaseTransport):
    """Async counterpart of :class:`ObservraTransport`, used by ``AsyncGemini``."""

    def __init__(
        self,
        config: ObservraConfig,
        *,
        inner: Optional[httpx.AsyncBaseTransport] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request = self._rewrite_to_gateway(request)
        start = time.monotonic()

        with self._span_scope() as span:
            request = self._prepare_request(request, span)

            response = await self._inner.handle_async_request(request)
            await response.aread()

            return self._finalize_response(response, request, span, start)

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        except Exception:  # noqa: BLE001
            pass


def json_text_extractor(*keys: str) -> ExtractText:
    """Build an extractor that pulls a dotted/nested key path out of a JSON body, truncated."""

    def _extract(body_text: str) -> Optional[str]:
        try:
            data = json.loads(body_text)
        except (json.JSONDecodeError, TypeError):
            return _default_extract_text(body_text)

        current: Any = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return _default_extract_text(body_text)

        text = current if isinstance(current, str) else json.dumps(current)
        return text[:MAX_ATTR_TEXT_LEN]

    return _extract

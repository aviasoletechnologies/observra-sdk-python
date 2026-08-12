"""Gemini provider patch — zero-touch gateway routing + tracing for ``google.genai``.

No wrapper classes: users keep writing plain ``google.genai.Client(...)``
(directly, or via any framework that builds one internally, e.g. LangChain's
``ChatGoogleGenerativeAI``). ``patch()`` monkeypatches ``genai.Client.__init__``
to inject the gateway transport whenever the caller hasn't already customized
the client's own httpx transport, and patches ``Models.generate_content`` /
``AsyncModels.generate_content`` to wrap Python-callable tools for span
tracing. Applied automatically once, the first time ``observra.configure()``
runs and ``google-genai`` is importable — see ``providers/registry.py``.
"""

from __future__ import annotations

import functools
import json
import logging
import threading
from typing import Any, Callable

import httpx

from observra.config import ObservraConfig, ObservraConfigError, get_config
from observra.providers.profiles import GEMINI_PROFILE
from observra.providers.transport import AsyncObservraTransport, ObservraTransport
from observra.tracing.conventions import Attr, SpanKind
from observra.tracing.tracer import ObservraTracer, get_tracer, safe_set_attributes

logger = logging.getLogger("observra")

# Architecture requirement #7: declare the exact tested range, check it at
# patch time, warn (never hard-fail) if the installed version is outside it.
_TESTED_GENAI_MIN = (0, 1, 0)
_TESTED_GENAI_MAX = (3, 0, 0)

_patched = False
_patch_lock = threading.Lock()


def _parse_version(raw: str) -> tuple[int, ...]:
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _check_genai_version() -> None:
    try:
        from importlib.metadata import version

        installed_raw = version("google-genai")
    except Exception:  # noqa: BLE001
        return  # can't determine version; do not block usage over this

    try:
        installed = _parse_version(installed_raw)
        if not (_TESTED_GENAI_MIN <= installed < _TESTED_GENAI_MAX):
            logger.warning(
                "observra: google-genai %s is outside the tested range [%s, %s) — "
                "tracing/guardrails may not match the response shape this SDK expects",
                installed_raw,
                ".".join(map(str, _TESTED_GENAI_MIN)),
                ".".join(map(str, _TESTED_GENAI_MAX)),
            )
    except Exception:  # noqa: BLE001
        logger.warning("observra: could not parse google-genai version %r", installed_raw)


def _extract_function_calls(body_text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(body_text)
        candidates = data.get("candidates") or []
        parts = candidates[0]["content"]["parts"]
        return [part["functionCall"] for part in parts if "functionCall" in part]
    except Exception:  # noqa: BLE001
        return []


def _make_tool_call_hook(tracer: ObservraTracer) -> Callable[[str, Any], None]:
    def _hook(response_text: str, _parent_span: Any) -> None:
        for call in _extract_function_calls(response_text):
            name = call.get("name", "unknown_tool")
            args = call.get("args", {})
            # Parented to the still-active LLM span — this hook runs from
            # inside ObservraTransport.handle_request, before that span
            # closes. tool.result isn't known yet at detection time; it's
            # recorded separately when the callable itself actually runs
            # (see _wrap_tool_for_tracing below).
            with tracer.start_span(f"tool.{name}", SpanKind.TOOL, {Attr.TOOL_NAME: name, Attr.TOOL_PARAMETERS: json.dumps(args)}):
                pass

    return _hook


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str)[:4000]
    except Exception:  # noqa: BLE001
        return str(value)[:4000]


def _wrap_tool_for_tracing(fn: Any, tracer: ObservraTracer) -> Any:
    """Wrap a user tool callable so its real execution is recorded as a TOOL span."""
    if not callable(fn):
        return fn

    name = getattr(fn, "__name__", "tool")

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracer.start_span(
            f"tool.{name}",
            SpanKind.TOOL,
            {Attr.TOOL_NAME: name, Attr.TOOL_PARAMETERS: _safe_json(kwargs or args)},
        ) as span:
            result = fn(*args, **kwargs)
            safe_set_attributes(span, {Attr.TOOL_RESULT: _safe_json(result)})
            return result

    return wrapper


def _wrap_tools_kwarg(kwargs: dict[str, Any], tracer: ObservraTracer) -> dict[str, Any]:
    """Wrap any Python-callable tools for span tracing.

    ``google-genai`` takes tools via ``config=GenerateContentConfig(tools=[...])``
    rather than a top-level ``tools=`` kwarg — both shapes are handled here so
    callers following either convention still get traced tool execution.
    """
    kwargs = dict(kwargs)

    tools = kwargs.get("tools")
    if tools:
        kwargs["tools"] = [_wrap_tool_for_tracing(t, tracer) if callable(t) else t for t in tools]

    config = kwargs.get("config")
    if config is not None:
        if isinstance(config, dict):
            config_tools = config.get("tools")
            if config_tools:
                kwargs["config"] = {
                    **config,
                    "tools": [_wrap_tool_for_tracing(t, tracer) if callable(t) else t for t in config_tools],
                }
        else:
            config_tools = getattr(config, "tools", None)
            if config_tools:
                wrapped_tools = [_wrap_tool_for_tracing(t, tracer) if callable(t) else t for t in config_tools]
                if hasattr(config, "model_copy"):
                    kwargs["config"] = config.model_copy(update={"tools": wrapped_tools})
                else:
                    try:
                        config.tools = wrapped_tools
                    except Exception:
                        logger.warning("observra: could not wrap tools on config object for tracing", exc_info=True)

    return kwargs


def _transport_kwargs(tracer: ObservraTracer) -> dict[str, Any]:
    return {
        **GEMINI_PROFILE.transport_kwargs(),
        "span_kind": SpanKind.LLM,
        "on_response_body": _make_tool_call_hook(tracer),
    }


def _build_transport(config: ObservraConfig, tracer: ObservraTracer) -> ObservraTransport:
    return ObservraTransport(config, **_transport_kwargs(tracer))


def _build_async_transport(config: ObservraConfig, tracer: ObservraTracer) -> AsyncObservraTransport:
    return AsyncObservraTransport(config, **_transport_kwargs(tracer))


def _http_options_field(http_options: Any, field: str) -> Any:
    if http_options is None:
        return None
    if isinstance(http_options, dict):
        return http_options.get(field)
    return getattr(http_options, field, None)


def _http_options_as_dict(http_options: Any) -> dict[str, Any]:
    if http_options is None:
        return {}
    if isinstance(http_options, dict):
        return dict(http_options)
    if hasattr(http_options, "model_dump"):
        return dict(http_options.model_dump(exclude_none=True))
    return {}


def _inject_gateway_http_options(genai_module: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject the gateway transport into ``Client(http_options=...)`` unless the caller already set one."""
    try:
        config = get_config()
    except ObservraConfigError:
        return kwargs  # observra.configure() not called yet — leave the client untouched

    http_options = kwargs.get("http_options")
    if _http_options_field(http_options, "httpx_client") is not None or _http_options_field(http_options, "httpx_async_client") is not None:
        return kwargs  # caller already customized the transport — respect it, don't override

    tracer = get_tracer(config)
    fields = _http_options_as_dict(http_options)
    fields["httpx_client"] = httpx.Client(transport=_build_transport(config, tracer))
    fields["httpx_async_client"] = httpx.AsyncClient(transport=_build_async_transport(config, tracer))

    kwargs = dict(kwargs)
    kwargs["http_options"] = genai_module.types.HttpOptions(**fields)
    return kwargs


def patch() -> None:
    """Idempotently patch ``google.genai.Client`` + ``Models``/``AsyncModels.generate_content``.

    Safe to call whether or not ``observra.configure()`` has run yet — the
    patched methods resolve the active config at call time and no-op
    (pass straight through to the original implementation) if it hasn't.
    """
    global _patched
    with _patch_lock:
        if _patched:
            return

        try:
            from google import genai
            from google.genai import models as genai_models
        except ImportError:
            return

        _check_genai_version()

        original_client_init = genai.Client.__init__

        @functools.wraps(original_client_init)
        def patched_client_init(self: Any, *args: Any, **kwargs: Any) -> None:
            try:
                kwargs = _inject_gateway_http_options(genai, kwargs)
            except Exception:
                logger.warning("observra: failed to inject gateway transport into genai.Client", exc_info=True)
            original_client_init(self, *args, **kwargs)

        genai.Client.__init__ = patched_client_init  # type: ignore[method-assign]

        original_sync_generate = genai_models.Models.generate_content
        original_async_generate = genai_models.AsyncModels.generate_content

        @functools.wraps(original_sync_generate)
        def patched_sync_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                config = get_config()
                kwargs = _wrap_tools_kwarg(kwargs, get_tracer(config))
            except ObservraConfigError:
                pass
            except Exception:
                logger.warning("observra: failed to wrap tools for tracing", exc_info=True)
            return original_sync_generate(self, *args, **kwargs)

        @functools.wraps(original_async_generate)
        async def patched_async_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                config = get_config()
                kwargs = _wrap_tools_kwarg(kwargs, get_tracer(config))
            except ObservraConfigError:
                pass
            except Exception:
                logger.warning("observra: failed to wrap tools for tracing", exc_info=True)
            return await original_async_generate(self, *args, **kwargs)

        genai_models.Models.generate_content = patched_sync_generate  # type: ignore[method-assign]
        genai_models.AsyncModels.generate_content = patched_async_generate  # type: ignore[method-assign]

        _patched = True

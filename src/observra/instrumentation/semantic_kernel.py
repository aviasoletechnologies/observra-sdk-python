"""Semantic Kernel instrumentation for functions, agents, and AI services.

Semantic Kernel's public function filters provide function lifecycle boundaries.
Public chat/text completion methods provide a synchronous-enough LLM boundary
before connector provider I/O starts, so provider transport enriches that LLM
span rather than producing a duplicate span.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from observra.config import get_config
from observra.tracing.context import activate_framework_llm_span, deactivate_framework_llm_span
from observra.tracing.conventions import Attr, SpanKind
from observra.tracing.tracer import ObservraTracer, get_tracer, safe_end_span, safe_set_attributes

logger = logging.getLogger("observra")

_TESTED_MIN = (1, 38, 0)
_TESTED_MAX = (2, 0, 0)
_MAX_ATTR_TEXT_LEN = 4000

_patched = False
_patch_lock = threading.Lock()
_original_kernel_init: Any = None
_original_chat_completion: Any = None
_original_text_completion: Any = None
_original_streaming_chat_completion: Any = None
_original_streaming_text_completion: Any = None
_original_agent_get_response: Any = None
_original_agent_invoke: Any = None
_original_agent_invoke_stream: Any = None


@dataclass
class _SpanScope:
    span: Any
    context_token: Any
    framework_llm_token: Any = None


def _parse_version(raw: str) -> tuple[int, ...]:
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(character for character in chunk if character.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _tracer_or_none() -> ObservraTracer | None:
    try:
        return get_tracer(get_config())
    except Exception:
        logger.warning(
            "observra: Semantic Kernel instrumentation active but observra.configure() "
            "was never called; skipping trace for this operation",
            exc_info=True,
        )
        return None


def _safe_str(value: Any) -> str:
    try:
        return json.dumps(value, default=str)[:_MAX_ATTR_TEXT_LEN]
    except Exception:  # noqa: BLE001
        return str(value)[:_MAX_ATTR_TEXT_LEN]


def _start_scope(name: str, kind: str, attributes: dict[str, Any]) -> _SpanScope | None:
    tracer = _tracer_or_none()
    if tracer is None:
        return None
    try:
        from opentelemetry import context as otel_context
        from opentelemetry import trace as trace_api

        span = tracer.start_detached_span(
            name,
            kind,
            {Attr.FRAMEWORK: "semantic_kernel", **attributes},
        )
        if span is None:
            return None
        context_token = otel_context.attach(trace_api.set_span_in_context(span))
        framework_llm_token = activate_framework_llm_span(span) if kind == SpanKind.LLM else None
        return _SpanScope(span, context_token, framework_llm_token)
    except Exception:
        logger.warning("observra: failed to start Semantic Kernel span", exc_info=True)
        return None


def _end_scope(
    scope: _SpanScope | None,
    attributes: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    if scope is None:
        return
    try:
        from opentelemetry import context as otel_context

        if attributes:
            safe_set_attributes(scope.span, attributes)
        safe_end_span(scope.span, error=error)
        if scope.framework_llm_token is not None:
            deactivate_framework_llm_span(scope.framework_llm_token)
        otel_context.detach(scope.context_token)
    except Exception:
        logger.warning("observra: failed to end Semantic Kernel span", exc_info=True)


def _function_attributes(context: Any) -> dict[str, Any]:
    function = context.function
    name = getattr(function, "fully_qualified_name", getattr(function, "name", "function"))
    return {
        Attr.TOOL_NAME: name,
        Attr.TOOL_PARAMETERS: _safe_str(getattr(context, "arguments", {})),
        "semantic_kernel.function.prompt": bool(getattr(function, "is_prompt", False)),
    }


async def _function_filter(context: Any, next: Any) -> None:
    """Trace public Kernel function filters without changing invocation semantics."""
    function = context.function
    is_prompt = bool(getattr(function, "is_prompt", False))
    kind = SpanKind.CHAIN if is_prompt else SpanKind.TOOL
    name = getattr(function, "fully_qualified_name", getattr(function, "name", "function"))
    scope = _start_scope(name, kind, _function_attributes(context))
    try:
        await next(context)
    except BaseException as error:
        _end_scope(scope, error=error)
        raise
    else:
        result = getattr(context, "result", None)
        output_key = Attr.OUTPUT_VALUE if is_prompt else Attr.TOOL_RESULT
        _end_scope(scope, {output_key: _safe_str(result)})


def _chat_history_text(history: Any) -> str:
    messages = getattr(history, "messages", history)
    return _safe_str(messages)


def _service_attributes(service: Any, input_value: str) -> dict[str, Any]:
    return {
        Attr.LLM_MODEL_NAME: getattr(service, "ai_model_id", None),
        Attr.INPUT_VALUE: input_value,
        "semantic_kernel.service": type(service).__name__,
    }


def _contents_text(contents: Any) -> str:
    return _safe_str(contents)


def _patch_kernel_filters() -> None:
    global _original_kernel_init
    from semantic_kernel.filters.filter_types import FilterTypes
    from semantic_kernel.kernel import Kernel

    _original_kernel_init = Kernel.__init__

    def kernel_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _original_kernel_init(self, *args, **kwargs)
        if not any(filter_ is _function_filter for _, filter_ in self.function_invocation_filters):
            self.add_filter(FilterTypes.FUNCTION_INVOCATION, _function_filter)

    Kernel.__init__ = kernel_init  # type: ignore[method-assign]


def _patch_ai_services() -> None:
    global _original_chat_completion
    global _original_text_completion
    global _original_streaming_chat_completion
    global _original_streaming_text_completion
    from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
    from semantic_kernel.connectors.ai.text_completion_client_base import TextCompletionClientBase

    _original_chat_completion = ChatCompletionClientBase.get_chat_message_contents
    _original_text_completion = TextCompletionClientBase.get_text_contents
    _original_streaming_chat_completion = (
        ChatCompletionClientBase.get_streaming_chat_message_contents
    )
    _original_streaming_text_completion = TextCompletionClientBase.get_streaming_text_contents

    async def get_chat_message_contents(
        service: Any,
        chat_history: Any,
        settings: Any,
        **kwargs: Any,
    ) -> Any:
        scope = _start_scope(
            "chat.completions",
            SpanKind.LLM,
            _service_attributes(service, _chat_history_text(chat_history)),
        )
        try:
            result = await _original_chat_completion(service, chat_history, settings, **kwargs)
        except BaseException as error:
            _end_scope(scope, error=error)
            raise
        _end_scope(scope, {Attr.OUTPUT_VALUE: _contents_text(result)})
        return result

    async def get_text_contents(service: Any, prompt: str, settings: Any) -> Any:
        scope = _start_scope("text.completions", SpanKind.LLM, _service_attributes(service, prompt))
        try:
            result = await _original_text_completion(service, prompt, settings)
        except BaseException as error:
            _end_scope(scope, error=error)
            raise
        _end_scope(scope, {Attr.OUTPUT_VALUE: _contents_text(result)})
        return result

    async def get_streaming_chat_message_contents(
        service: Any,
        chat_history: Any,
        settings: Any,
        **kwargs: Any,
    ) -> Any:
        scope = _start_scope(
            "chat.completions",
            SpanKind.LLM,
            _service_attributes(service, _chat_history_text(chat_history)),
        )
        chunks = []
        error: BaseException | None = None
        completed = False
        try:
            async for result in _original_streaming_chat_completion(
                service, chat_history, settings, **kwargs
            ):
                chunks.append(_contents_text(result))
                yield result
            completed = True
        except BaseException as caught_error:
            error = caught_error
            raise
        finally:
            attributes = (
                {Attr.OUTPUT_VALUE: "".join(chunks)[:_MAX_ATTR_TEXT_LEN]} if completed else None
            )
            _end_scope(scope, attributes, error)

    async def get_streaming_text_contents(service: Any, prompt: str, settings: Any) -> Any:
        scope = _start_scope("text.completions", SpanKind.LLM, _service_attributes(service, prompt))
        chunks = []
        error: BaseException | None = None
        completed = False
        try:
            async for result in _original_streaming_text_completion(service, prompt, settings):
                chunks.append(_contents_text(result))
                yield result
            completed = True
        except BaseException as caught_error:
            error = caught_error
            raise
        finally:
            attributes = (
                {Attr.OUTPUT_VALUE: "".join(chunks)[:_MAX_ATTR_TEXT_LEN]} if completed else None
            )
            _end_scope(scope, attributes, error)

    ChatCompletionClientBase.get_chat_message_contents = get_chat_message_contents  # type: ignore
    TextCompletionClientBase.get_text_contents = get_text_contents  # type: ignore
    chat_completion_client_base: Any = ChatCompletionClientBase
    text_completion_client_base: Any = TextCompletionClientBase
    chat_completion_client_base.get_streaming_chat_message_contents = (
        get_streaming_chat_message_contents
    )
    text_completion_client_base.get_streaming_text_contents = get_streaming_text_contents


def _agent_attributes(agent: Any, messages: Any) -> dict[str, Any]:
    return {
        "agent.name": getattr(agent, "name", type(agent).__name__),
        "agent.description": getattr(agent, "description", None),
        Attr.INPUT_VALUE: _safe_str(messages),
    }


def _patch_chat_completion_agent() -> None:
    global _original_agent_get_response, _original_agent_invoke, _original_agent_invoke_stream
    from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent

    _original_agent_get_response = ChatCompletionAgent.get_response
    _original_agent_invoke = ChatCompletionAgent.invoke
    _original_agent_invoke_stream = ChatCompletionAgent.invoke_stream

    async def get_response(agent: Any, messages: Any = None, **kwargs: Any) -> Any:
        scope = _start_scope("agent.response", SpanKind.AGENT, _agent_attributes(agent, messages))
        try:
            result = await _original_agent_get_response(agent, messages, **kwargs)
        except BaseException as error:
            _end_scope(scope, error=error)
            raise
        _end_scope(scope, {Attr.OUTPUT_VALUE: _safe_str(result)})
        return result

    async def invoke(agent: Any, messages: Any = None, **kwargs: Any) -> Any:
        scope = _start_scope("agent.invoke", SpanKind.AGENT, _agent_attributes(agent, messages))
        error: BaseException | None = None
        try:
            async for result in _original_agent_invoke(agent, messages, **kwargs):
                yield result
        except BaseException as caught_error:
            error = caught_error
            raise
        finally:
            _end_scope(scope, error=error)

    async def invoke_stream(agent: Any, messages: Any = None, **kwargs: Any) -> Any:
        scope = _start_scope("agent.invoke", SpanKind.AGENT, _agent_attributes(agent, messages))
        error: BaseException | None = None
        try:
            async for result in _original_agent_invoke_stream(agent, messages, **kwargs):
                yield result
        except BaseException as caught_error:
            error = caught_error
            raise
        finally:
            _end_scope(scope, error=error)

    ChatCompletionAgent.get_response = get_response  # type: ignore[method-assign,assignment]
    ChatCompletionAgent.invoke = invoke  # type: ignore[method-assign,assignment]
    ChatCompletionAgent.invoke_stream = invoke_stream  # type: ignore[method-assign,assignment]


def patch() -> None:
    """Install Semantic Kernel function, agent, and AI-service instrumentation."""
    global _patched
    with _patch_lock:
        if _patched:
            return
        try:
            from importlib.metadata import version

            installed = version("semantic-kernel")
            if not (_TESTED_MIN <= _parse_version(installed) < _TESTED_MAX):
                logger.warning(
                    "observra: semantic-kernel %s outside tested range; skipping", installed
                )
                return
            _patch_kernel_filters()
            _patch_ai_services()
            _patch_chat_completion_agent()
            _patched = True
        except Exception:
            logger.warning(
                "observra: failed to install Semantic Kernel instrumentation", exc_info=True
            )

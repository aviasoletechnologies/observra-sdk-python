# ruff: noqa: E501
"""CrewAI event-bus instrumentation for crews, flows, agents, LLMs, and tools.

CrewAI emits public lifecycle events for normal LLM calls and agentic execution.
This integration maps supported paired events to OpenTelemetry spans without
patching CrewAI private execution internals. Provider HTTP transport enriches
an active CrewAI LLM span rather than creating a duplicate child LLM span.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from observra.config import get_config
from observra.tracing.context import activate_framework_llm_span, deactivate_framework_llm_span
from observra.tracing.conventions import Attr, SpanKind
from observra.tracing.tracer import (
    ObservraTracer,
    get_tracer,
    safe_add_event,
    safe_end_span,
    safe_set_attributes,
)

logger = logging.getLogger("observra")

_TESTED_MIN = (0, 80, 0)
_TESTED_MAX = (2, 0, 0)
_MAX_ATTR_TEXT_LEN = 4000
_MAX_STREAM_EVENTS = 100

_patched = False
_patch_lock = threading.Lock()
_original_emit_llm_start: Any = None
_original_emit_llm_complete: Any = None
_original_emit_llm_failed: Any = None
_llm_scope_lock = threading.Lock()
_llm_scope_tokens: Dict[str, Tuple[Any, Any]] = {}


def _parse_version(raw: str) -> Tuple[int, ...]:
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(character for character in chunk if character.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _tracer_or_none() -> Optional[ObservraTracer]:
    try:
        return get_tracer(get_config())
    except Exception:  # noqa: BLE001
        logger.warning(
            "observra: CrewAI instrumentation active but observra.configure() "
            "was never called; skipping trace for this run",
            exc_info=True,
        )
        return None


def _safe_str(value: Any) -> str:
    try:
        return json.dumps(value, default=str)[:_MAX_ATTR_TEXT_LEN]
    except Exception:  # noqa: BLE001
        return str(value)[:_MAX_ATTR_TEXT_LEN]


def _correlation_key(source: Any, event: Any) -> Any:
    """Return stable key for legacy event pairs without ``started_event_id``."""
    call_id = getattr(event, "call_id", None)
    if call_id is not None:
        return call_id
    task = getattr(event, "task", None)
    if task is not None:
        return ("task", id(task))
    return ("source", id(source))


def _start_key(source: Any, event: Any) -> Any:
    """Use a call ID for LLM streaming; otherwise use CrewAI event identity."""
    if getattr(event, "call_id", None) is not None:
        return _correlation_key(source, event)
    event_id = getattr(event, "event_id", None)
    return ("event", event_id) if event_id else _correlation_key(source, event)


def _end_key(source: Any, event: Any) -> Any:
    if getattr(event, "call_id", None) is not None:
        return _correlation_key(source, event)
    started_event_id = getattr(event, "started_event_id", None)
    return ("event", started_event_id) if started_event_id else _correlation_key(source, event)


def _parent_keys(event: Any) -> Iterable[Any]:
    for attribute in ("parent_event_id", "triggered_by_event_id"):
        event_id = getattr(event, attribute, None)
        if event_id:
            yield ("event", event_id)


@dataclass
class _SpanState:
    span: Any
    stream_token_count: int = 0
    stream_parts: list[str] = field(default_factory=list)


class _SpanTracker:
    def __init__(self) -> None:
        self._by_key: Dict[Any, _SpanState] = {}
        self._lock = threading.Lock()

    def _parent_span(self, parent_keys: Iterable[Any]) -> Any:
        with self._lock:
            for key in parent_keys:
                state = self._by_key.get(key)
                if state is not None:
                    return state.span
        return None

    def start(
        self,
        key: Any,
        name: str,
        kind: str,
        attributes: Dict[str, Any],
        *,
        parent_keys: Iterable[Any] = (),
        framework_llm: bool = False,
    ) -> None:
        tracer = _tracer_or_none()
        if tracer is None:
            return
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as trace_api

            parent_token = None
            parent_span = self._parent_span(parent_keys)
            if parent_span is not None:
                parent_token = otel_context.attach(trace_api.set_span_in_context(parent_span))
            try:
                span = tracer.start_detached_span(
                    name,
                    kind,
                    {Attr.FRAMEWORK: "crewai", **attributes},
                )
            finally:
                if parent_token is not None:
                    otel_context.detach(parent_token)
            if span is None:
                return

            with self._lock:
                self._by_key[key] = _SpanState(span=span)
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to start span for CrewAI event", exc_info=True)

    def end(
        self,
        key: Any,
        attributes: Optional[Dict[str, Any]] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        with self._lock:
            state = self._by_key.pop(key, None)
        if state is None:
            return
        try:
            final_attributes = dict(attributes or {})
            if state.stream_token_count:
                final_attributes[Attr.LLM_STREAM_TOKEN_COUNT] = state.stream_token_count
            if state.stream_parts and not final_attributes.get(Attr.OUTPUT_VALUE):
                final_attributes[Attr.OUTPUT_VALUE] = "".join(state.stream_parts)[
                    :_MAX_ATTR_TEXT_LEN
                ]
            if final_attributes:
                safe_set_attributes(state.span, final_attributes)
            safe_end_span(state.span, error=error)
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to end span for CrewAI event", exc_info=True)

    def add_event(
        self,
        key: Any,
        name: str,
        attributes: Dict[str, Any],
        *,
        stream_chunk: Any = None,
    ) -> None:
        with self._lock:
            state = self._by_key.get(key)
            if state is None:
                return
            if stream_chunk is not None:
                state.stream_token_count += 1
                if sum(len(part) for part in state.stream_parts) < _MAX_ATTR_TEXT_LEN:
                    state.stream_parts.append(str(stream_chunk))
                if state.stream_token_count > _MAX_STREAM_EVENTS:
                    return
        safe_add_event(state.span, name, attributes)


_tracker = _SpanTracker()


def _activate_llm_call_scope(call_id: str) -> None:
    """Expose matching CrewAI span to provider I/O in the caller Context."""
    with _tracker._lock:
        state = _tracker._by_key.get(call_id)
    if state is None:
        return

    from opentelemetry import context as otel_context
    from opentelemetry import trace as trace_api

    context_token = otel_context.attach(trace_api.set_span_in_context(state.span))
    framework_token = activate_framework_llm_span(state.span)
    with _llm_scope_lock:
        _llm_scope_tokens[call_id] = (context_token, framework_token)


def _deactivate_llm_call_scope(call_id: str) -> None:
    """Restore caller Context after CrewAI emits completion or failure."""
    with _llm_scope_lock:
        tokens = _llm_scope_tokens.pop(call_id, None)
    if tokens is None:
        return

    from opentelemetry import context as otel_context

    context_token, framework_token = tokens
    deactivate_framework_llm_span(framework_token)
    otel_context.detach(context_token)


def _patch_llm_execution() -> None:
    """Bind all native and LiteLLM CrewAI calls to their event span."""
    global _original_emit_llm_start
    global _original_emit_llm_complete
    global _original_emit_llm_failed
    if _original_emit_llm_start is not None:
        return

    try:
        from crewai.llms.base_llm import BaseLLM, get_current_call_id
    except ImportError:
        return

    _original_emit_llm_start = BaseLLM._emit_call_started_event
    _original_emit_llm_complete = BaseLLM._emit_call_completed_event
    _original_emit_llm_failed = BaseLLM._emit_call_failed_event

    def emit_start(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = _original_emit_llm_start(self, *args, **kwargs)
        _activate_llm_call_scope(get_current_call_id())
        return result

    def emit_complete(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return _original_emit_llm_complete(self, *args, **kwargs)
        finally:
            _deactivate_llm_call_scope(get_current_call_id())

    def emit_failed(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return _original_emit_llm_failed(self, *args, **kwargs)
        finally:
            _deactivate_llm_call_scope(get_current_call_id())

    BaseLLM._emit_call_started_event = emit_start  # type: ignore[method-assign]
    BaseLLM._emit_call_completed_event = emit_complete  # type: ignore[method-assign]
    BaseLLM._emit_call_failed_event = emit_failed  # type: ignore[method-assign]


def _event_attributes(event: Any, *names: str) -> Dict[str, Any]:
    return {
        name: _safe_str(getattr(event, name, None))
        for name in names
        if getattr(event, name, None) is not None
    }


def _error_from(event: Any) -> RuntimeError:
    return RuntimeError(str(getattr(event, "error", "CrewAI operation failed")))


def _subscribe_start(
    event_bus: Any,
    event_type: Any,
    name: str,
    kind: str,
    attributes: Any,
    *,
    llm: bool = False,
) -> None:
    def handler(source: Any, event: Any) -> None:
        _tracker.start(
            _start_key(source, event),
            name.format(name=getattr(event, "tool_name", "tool")),
            kind,
            attributes(event),
            parent_keys=_parent_keys(event),
            framework_llm=llm,
        )

    event_bus.on(event_type)(handler)


def _subscribe_end(event_bus: Any, event_type: Any, attributes: Any) -> None:
    def handler(source: Any, event: Any) -> None:
        _tracker.end(_end_key(source, event), attributes(event))

    event_bus.on(event_type)(handler)


def _subscribe_error(event_bus: Any, event_type: Any) -> None:
    def handler(source: Any, event: Any) -> None:
        _tracker.end(_end_key(source, event), error=_error_from(event))

    event_bus.on(event_type)(handler)


def _event_type(module_name: str, class_name: str) -> Any:
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError):
        return None


def _subscribe_if_present(
    event_bus: Any,
    module_name: str,
    start: str,
    end: Optional[str],
    failed: Optional[str],
    name: str,
    kind: str,
    start_attributes: Any,
    end_attributes: Any,
    *,
    llm: bool = False,
) -> None:
    start_type = _event_type(module_name, start)
    if start_type is None:
        return
    _subscribe_start(event_bus, start_type, name, kind, start_attributes, llm=llm)
    if end is not None:
        end_type = _event_type(module_name, end)
        if end_type is not None:
            _subscribe_end(event_bus, end_type, end_attributes)
    if failed is not None:
        failed_type = _event_type(module_name, failed)
        if failed_type is not None:
            _subscribe_error(event_bus, failed_type)


def _register_lifecycle_handlers(event_bus: Any) -> None:
    crew = "crewai.events.types.crew_events"
    task = "crewai.events.types.task_events"
    agent = "crewai.events.types.agent_events"
    llm = "crewai.events.types.llm_events"
    tool = "crewai.events.types.tool_usage_events"
    flow = "crewai.events.types.flow_events"
    memory = "crewai.events.types.memory_events"
    knowledge = "crewai.events.types.knowledge_events"
    mcp = "crewai.events.types.mcp_events"
    reasoning = "crewai.events.types.reasoning_events"

    subscriptions: tuple[Any, ...] = (
        (crew, "CrewKickoffStartedEvent", "CrewKickoffCompletedEvent", "CrewKickoffFailedEvent", "crewai.crew", SpanKind.CHAIN, lambda event: {Attr.INPUT_VALUE: _safe_str(getattr(event, "inputs", None)), "crewai.crew.name": getattr(event, "crew_name", None)}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "output", None)), "llm.token_count.total": getattr(event, "total_tokens", None)}, False),
        (task, "TaskStartedEvent", "TaskCompletedEvent", "TaskFailedEvent", "crewai.task", SpanKind.CHAIN, lambda event: {Attr.INPUT_VALUE: _safe_str(getattr(event, "context", None)), "crewai.task.name": getattr(event, "task_name", None)}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "output", None))}, False),
        (agent, "AgentExecutionStartedEvent", "AgentExecutionCompletedEvent", "AgentExecutionErrorEvent", "crewai.agent", SpanKind.AGENT, lambda event: {Attr.INPUT_VALUE: _safe_str(getattr(event, "task_prompt", None)), "crewai.agent.role": getattr(getattr(event, "agent", None), "role", None)}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "output", None))}, False),
        (agent, "LiteAgentExecutionStartedEvent", "LiteAgentExecutionCompletedEvent", "LiteAgentExecutionErrorEvent", "crewai.lite_agent", SpanKind.AGENT, lambda event: {Attr.INPUT_VALUE: _safe_str(getattr(event, "messages", None)), "crewai.agent.info": _safe_str(getattr(event, "agent_info", None))}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "output", None))}, False),
        (llm, "LLMCallStartedEvent", "LLMCallCompletedEvent", "LLMCallFailedEvent", "crewai.llm_call", SpanKind.LLM, lambda event: {Attr.LLM_MODEL_NAME: getattr(event, "model", None), Attr.INPUT_VALUE: _safe_str(getattr(event, "messages", None)), "llm.request.tools": _safe_str(getattr(event, "tools", None)), "llm.request.stream": getattr(event, "stream", None)}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "response", None)), Attr.LLM_TOKEN_COUNT_PROMPT: (getattr(event, "usage", None) or {}).get("prompt_tokens"), Attr.LLM_TOKEN_COUNT_COMPLETION: (getattr(event, "usage", None) or {}).get("completion_tokens"), "llm.finish_reason": getattr(event, "finish_reason", None)}, True),
        (tool, "ToolUsageStartedEvent", "ToolUsageFinishedEvent", "ToolUsageErrorEvent", "tool.{name}", SpanKind.TOOL, lambda event: {Attr.TOOL_NAME: getattr(event, "tool_name", "tool"), Attr.TOOL_PARAMETERS: _safe_str(getattr(event, "tool_args", None))}, lambda event: {Attr.TOOL_RESULT: _safe_str(getattr(event, "output", None)), "tool.from_cache": getattr(event, "from_cache", None)}, False),
        (flow, "FlowStartedEvent", "FlowFinishedEvent", "FlowFailedEvent", "crewai.flow", SpanKind.CHAIN, lambda event: {Attr.INPUT_VALUE: _safe_str(getattr(event, "inputs", None)), "crewai.flow.name": getattr(event, "flow_name", None)}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "result", None))}, False),
        (flow, "MethodExecutionStartedEvent", "MethodExecutionFinishedEvent", "MethodExecutionFailedEvent", "crewai.flow.method", SpanKind.CHAIN, lambda event: {Attr.INPUT_VALUE: _safe_str(getattr(event, "params", None)), "crewai.flow.method": getattr(event, "method_name", None)}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "result", None))}, False),
        (memory, "MemoryQueryStartedEvent", "MemoryQueryCompletedEvent", "MemoryQueryFailedEvent", "crewai.memory.query", SpanKind.RETRIEVER, lambda event: {Attr.RETRIEVER_QUERY: getattr(event, "query", None)}, lambda event: {Attr.RETRIEVER_DOCUMENT_COUNT: len(getattr(event, "results", []) or [])}, False),
        (memory, "MemoryRetrievalStartedEvent", "MemoryRetrievalCompletedEvent", "MemoryRetrievalFailedEvent", "crewai.memory.retrieve", SpanKind.RETRIEVER, lambda event: {Attr.RETRIEVER_QUERY: getattr(event, "task_id", None)}, lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "memory_content", None))}, False),
        (knowledge, "KnowledgeRetrievalStartedEvent", "KnowledgeRetrievalCompletedEvent", "KnowledgeSearchQueryFailedEvent", "crewai.knowledge.retrieve", SpanKind.RETRIEVER, lambda event: {}, lambda event: {Attr.RETRIEVER_QUERY: getattr(event, "query", None), Attr.OUTPUT_VALUE: _safe_str(getattr(event, "retrieved_knowledge", None))}, False),
        (reasoning, "AgentReasoningStartedEvent", "AgentReasoningCompletedEvent", "AgentReasoningFailedEvent", "crewai.agent.reasoning", SpanKind.CHAIN, lambda event: _event_attributes(event, "agent_role", "task_id", "attempt"), lambda event: {Attr.OUTPUT_VALUE: _safe_str(getattr(event, "plan", None)), "crewai.reasoning.ready": getattr(event, "ready", None)}, False),
        (mcp, "MCPToolExecutionStartedEvent", "MCPToolExecutionCompletedEvent", "MCPToolExecutionFailedEvent", "tool.mcp.{name}", SpanKind.TOOL, lambda event: {Attr.TOOL_NAME: getattr(event, "tool_name", "mcp"), Attr.TOOL_PARAMETERS: _safe_str(getattr(event, "tool_args", None)), "mcp.server.name": getattr(event, "server_name", None)}, lambda event: {Attr.TOOL_RESULT: _safe_str(getattr(event, "result", None))}, False),
    )
    for module, start, end, failed, name, kind, start_attrs, end_attrs, is_llm in subscriptions:
        _subscribe_if_present(event_bus, module, start, end, failed, name, kind, start_attrs, end_attrs, llm=is_llm)

    stream_type = _event_type(llm, "LLMStreamChunkEvent")
    if stream_type is not None:
        def on_llm_stream(source: Any, event: Any) -> None:
            _tracker.add_event(
                _correlation_key(source, event),
                "llm.stream.chunk",
                {"chunk": _safe_str(getattr(event, "chunk", None))},
                stream_chunk=getattr(event, "chunk", None),
            )

        event_bus.on(stream_type)(on_llm_stream)

    thinking_type = _event_type(llm, "LLMThinkingChunkEvent")
    if thinking_type is not None:
        def on_llm_thinking(source: Any, event: Any) -> None:
            _tracker.add_event(
                _correlation_key(source, event),
                "llm.thinking.chunk",
                {"chunk": _safe_str(getattr(event, "chunk", None))},
            )

        event_bus.on(thinking_type)(on_llm_thinking)


def patch() -> None:
    """Idempotently subscribe supported public CrewAI event lifecycles."""
    global _patched
    with _patch_lock:
        if _patched:
            return
        try:
            from importlib.metadata import version

            installed = version("crewai")
            version_tuple = _parse_version(installed)
            if not (_TESTED_MIN <= version_tuple < _TESTED_MAX):
                logger.warning("observra: crewai %s outside tested range; skipping instrumentation", installed)
                return
        except Exception:  # noqa: BLE001
            pass

        try:
            try:
                from crewai.events import crewai_event_bus
            except ImportError:
                from crewai.utilities.events import crewai_event_bus  # type: ignore[import-not-found, no-redef]
            _register_lifecycle_handlers(crewai_event_bus)
            _patch_llm_execution()
            _patched = True
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to subscribe to CrewAI event bus, skipping instrumentation", exc_info=True)

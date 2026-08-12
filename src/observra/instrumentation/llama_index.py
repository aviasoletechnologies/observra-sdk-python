"""LlamaIndex instrumentation — same pattern as ``instrumentation/langchain.py``.

LlamaIndex has its own callback system (``llama_index.core.callbacks``) with a
different shape (event start/end pairs keyed by ``event_id`` and a
``CBEventType`` enum) but the same overall approach applies: patch the
callback manager's constructor so every manager LlamaIndex builds inherits
our handler, map LLM/AGENT_STEP/FUNCTION_CALL events onto the same
observra span conventions used everywhere else.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from observra.config import get_config
from observra.tracing.context import activate_framework_llm_span, deactivate_framework_llm_span
from observra.tracing.conventions import Attr, SpanKind
from observra.tracing.tracer import ObservraTracer, get_tracer, safe_end_span, safe_set_attributes

logger = logging.getLogger("observra")

_TESTED_MIN = (0, 10, 0)
_TESTED_MAX = (0, 15, 0)

_patched = False
_patch_lock = threading.Lock()
_original_callback_manager_init: Any = None
_original_function_tool_call: Any = None
_original_function_tool_acall: Any = None
_original_function_agent_take_step: Any = None
_original_react_agent_take_step: Any = None
_original_agent_workflow_run: Any = None


def _parse_version(raw: str) -> Tuple[int, ...]:
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _tracer_or_none() -> Optional[ObservraTracer]:
    try:
        return get_tracer(get_config())
    except Exception:  # noqa: BLE001
        logger.warning(
            "observra: LlamaIndex instrumentation active but observra.configure() "
            "was never called; skipping trace for this run",
            exc_info=True,
        )
        return None


def _safe_str(value: Any) -> str:
    def redact(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: (
                    "[REDACTED]"
                    if "key" in key.lower() or "token" in key.lower()
                    else redact(item_value)
                )
                for key, item_value in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [redact(entry) for entry in item]
        return item

    try:
        return json.dumps(redact(value), default=str)[:4000]
    except Exception:  # noqa: BLE001
        return str(value)[:4000]


_EVENT_KIND_MAP: Dict[str, str] = {
    "LLM": SpanKind.LLM,
    "EMBEDDING": SpanKind.LLM,
    "AGENT_STEP": SpanKind.AGENT,
    "FUNCTION_CALL": SpanKind.TOOL,
    "RETRIEVE": SpanKind.RETRIEVER,
}


_LLM_EVENTS = {"LLM", "EMBEDDING"}


@dataclass
class _EventState:
    span: Any
    context_token: Any
    framework_llm_token: Any = None
    event_type: str = ""
    suppressed: bool = False


def _base_handler_class() -> type:
    """Return LlamaIndex BaseCallbackHandler, or object when dependency is absent."""
    try:
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler

        return BaseCallbackHandler
    except ImportError:
        return object


class ObservraLlamaIndexCallbackHandler(_base_handler_class()):  # type: ignore[misc]
    """Bridges LlamaIndex's ``on_event_start``/``on_event_end`` pairs to observra spans.

    Must subclass the real ``BaseCallbackHandler`` — ``CallbackManager``
    reads ``event_starts_to_ignore``/``event_ends_to_ignore`` off every
    handler it holds, which only that base class sets up.
    """

    def __init__(self) -> None:
        try:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        except TypeError:
            pass  # base_handler_class() fell back to plain object — no-op
        self._events: Dict[str, _EventState] = {}
        self._lock = threading.Lock()

    def _parent_state(self, parent_id: str) -> Optional[_EventState]:
        if not parent_id:
            return None
        with self._lock:
            return self._events.get(parent_id)

    def _parent_span(self, parent_id: str) -> Any:
        state = self._parent_state(parent_id)
        return state.span if state is not None else None

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        tracer = _tracer_or_none()
        if tracer is None:
            return event_id

        type_name = getattr(event_type, "name", str(event_type))
        kind = _EVENT_KIND_MAP.get(type_name, SpanKind.CHAIN)
        parent_state = self._parent_state(parent_id)
        if type_name == "LLM" and parent_state is not None and parent_state.event_type == "LLM":
            with self._lock:
                self._events[event_id] = _EventState(
                    span=None,
                    context_token=None,
                    event_type=type_name,
                    suppressed=True,
                )
            return event_id

        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as trace_api

            parent_span = self._parent_span(parent_id)
            parent_token = None
            if parent_span is not None:
                parent_token = otel_context.attach(trace_api.set_span_in_context(parent_span))
            try:
                span = tracer.start_detached_span(
                    f"llama_index.{type_name.lower()}",
                    kind,
                    {
                        Attr.FRAMEWORK: "llama_index",
                        Attr.LLAMA_INDEX_EVENT_NAME: type_name,
                        **({Attr.INPUT_VALUE: _safe_str(payload)} if payload else {}),
                    },
                )
            finally:
                if parent_token is not None:
                    otel_context.detach(parent_token)
            if span is None:
                return event_id

            context_token = otel_context.attach(trace_api.set_span_in_context(span))
            framework_llm_token = (
                activate_framework_llm_span(span) if type_name in _LLM_EVENTS else None
            )
            with self._lock:
                self._events[event_id] = _EventState(
                    span=span,
                    context_token=context_token,
                    framework_llm_token=framework_llm_token,
                    event_type=type_name,
                )
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to start span for LlamaIndex event", exc_info=True)

        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        with self._lock:
            entry = self._events.pop(event_id, None)
        if entry is None:
            return

        state = entry
        if state.suppressed:
            return
        try:
            from opentelemetry import context as otel_context

            if payload:
                safe_set_attributes(state.span, {Attr.OUTPUT_VALUE: _safe_str(payload)})
            safe_end_span(state.span)
            if state.framework_llm_token is not None:
                deactivate_framework_llm_span(state.framework_llm_token)
            otel_context.detach(state.context_token)
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to end span for LlamaIndex event", exc_info=True)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass


_handler = ObservraLlamaIndexCallbackHandler()


def _handlers_with_observra(
    handlers: Optional[list[Any]],
) -> list[Any]:
    """Return callback handlers with this integration attached exactly once."""
    configured_handlers = list(handlers or [])
    if not any(
        isinstance(handler, ObservraLlamaIndexCallbackHandler)
        for handler in configured_handlers
    ):
        configured_handlers.append(_handler)
    return configured_handlers


def _tool_attributes(tool: Any, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, str]:
    return {
        Attr.FRAMEWORK: "llama_index",
        Attr.LLAMA_INDEX_EVENT_NAME: "FUNCTION_CALL",
        Attr.TOOL_NAME: tool.metadata.get_name(),
        Attr.TOOL_PARAMETERS: _safe_str({"args": args, "kwargs": kwargs}),
    }


def _finish_tool_span(
    span: Any,
    result: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    if result is not None:
        safe_set_attributes(span, {Attr.TOOL_RESULT: _safe_str(result)})
    safe_end_span(span, error=error)


def _patch_function_tools() -> None:
    """Trace FunctionTool calls that bypass LlamaIndex's legacy callbacks."""
    global _original_function_tool_call, _original_function_tool_acall
    from llama_index.core.tools import FunctionTool

    _original_function_tool_call = getattr(
        FunctionTool.call,
        "__wrapped__",
        FunctionTool.call,
    )
    _original_function_tool_acall = getattr(
        FunctionTool.acall,
        "__wrapped__",
        FunctionTool.acall,
    )

    def call(tool: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = _tracer_or_none()
        if tracer is None:
            return _original_function_tool_call(tool, *args, **kwargs)

        span = tracer.start_detached_span(
            "llama_index.function_call",
            SpanKind.TOOL,
            _tool_attributes(tool, args, kwargs),
        )
        if span is None:
            return _original_function_tool_call(tool, *args, **kwargs)
        try:
            result = _original_function_tool_call(tool, *args, **kwargs)
        except Exception as error:
            _finish_tool_span(span, error=error)
            raise
        _finish_tool_span(span, result)
        return result

    async def acall(tool: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = _tracer_or_none()
        if tracer is None:
            return await _original_function_tool_acall(tool, *args, **kwargs)

        span = tracer.start_detached_span(
            "llama_index.function_call",
            SpanKind.TOOL,
            _tool_attributes(tool, args, kwargs),
        )
        if span is None:
            return await _original_function_tool_acall(tool, *args, **kwargs)
        try:
            result = await _original_function_tool_acall(tool, *args, **kwargs)
        except Exception as error:
            _finish_tool_span(span, error=error)
            raise
        _finish_tool_span(span, result)
        return result

    FunctionTool.call = call  # type: ignore[method-assign,assignment]
    FunctionTool.acall = acall  # type: ignore[method-assign,assignment]


def _patch_agent_steps() -> None:
    """Trace dispatcher-based agent steps absent from legacy callbacks."""
    global _original_function_agent_take_step, _original_react_agent_take_step
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent

    _original_function_agent_take_step = FunctionAgent.take_step
    _original_react_agent_take_step = ReActAgent.take_step

    def wrap(original: Any) -> Any:
        async def take_step(agent: Any, *args: Any, **kwargs: Any) -> Any:
            tracer = _tracer_or_none()
            if tracer is None:
                return await original(agent, *args, **kwargs)

            span = tracer.start_detached_span(
                "llama_index.agent_step",
                SpanKind.AGENT,
                {
                    Attr.FRAMEWORK: "llama_index",
                    Attr.LLAMA_INDEX_EVENT_NAME: "AGENT_STEP",
                    "agent.name": agent.name,
                    "agent.description": agent.description,
                },
            )
            if span is None:
                return await original(agent, *args, **kwargs)

            from opentelemetry import context as otel_context
            from opentelemetry import trace as trace_api

            context_token = otel_context.attach(trace_api.set_span_in_context(span))
            try:
                result = await original(agent, *args, **kwargs)
            except Exception as error:
                safe_end_span(span, error=error)
                raise
            else:
                safe_end_span(span)
                return result
            finally:
                otel_context.detach(context_token)

        return take_step

    FunctionAgent.take_step = wrap(_original_function_agent_take_step)  # type: ignore[method-assign]
    ReActAgent.take_step = wrap(_original_react_agent_take_step)  # type: ignore[method-assign]


def _patch_agent_workflow_run() -> None:
    """Create one workflow root inherited by every scheduled agent operation."""
    global _original_agent_workflow_run
    from llama_index.core.agent.workflow import AgentWorkflow

    _original_agent_workflow_run = AgentWorkflow.run

    def run(workflow: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = _tracer_or_none()
        if tracer is None:
            return _original_agent_workflow_run(workflow, *args, **kwargs)

        span = tracer.start_detached_span(
            "llama_index.workflow",
            SpanKind.CHAIN,
            {
                Attr.FRAMEWORK: "llama_index",
                Attr.LLAMA_INDEX_EVENT_NAME: "WORKFLOW",
                "workflow.name": type(workflow).__name__,
                "workflow.root_agent": workflow.root_agent,
            },
        )
        if span is None:
            return _original_agent_workflow_run(workflow, *args, **kwargs)

        from opentelemetry import context as otel_context
        from opentelemetry import trace as trace_api

        context_token = otel_context.attach(trace_api.set_span_in_context(span))
        try:
            handler = _original_agent_workflow_run(workflow, *args, **kwargs)
        except Exception as error:
            safe_end_span(span, error=error)
            raise
        finally:
            otel_context.detach(context_token)

        def end_workflow(task: Any) -> None:
            if task.cancelled():
                safe_end_span(span, error=RuntimeError("workflow cancelled"))
                return
            try:
                error = task.exception()
            except Exception as exception:  # noqa: BLE001
                safe_end_span(span, error=exception)
                return
            safe_end_span(span, error=error)

        handler._result_task.add_done_callback(end_workflow)
        return handler

    AgentWorkflow.run = run  # type: ignore[method-assign,assignment]


def patch() -> None:
    """Attach to global and independently created LlamaIndex callback managers."""
    global _patched, _original_callback_manager_init
    with _patch_lock:
        if _patched:
            return

        try:
            from importlib.metadata import version

            installed = version("llama-index-core")
            version_tuple = _parse_version(installed)
            if not (_TESTED_MIN <= version_tuple < _TESTED_MAX):
                logger.warning(
                    "observra: llama-index-core %s is outside the tested range [%s, %s) — "
                    "skipping instrumentation",
                    installed,
                    ".".join(map(str, _TESTED_MIN)),
                    ".".join(map(str, _TESTED_MAX)),
                )
                return
        except Exception:  # noqa: BLE001
            pass  # version undeterminable: proceed, best-effort

        try:
            from llama_index.core import Settings
            from llama_index.core.callbacks import CallbackManager
        except ImportError:
            logger.warning("observra: llama_index.core not importable, skipping instrumentation")
            return

        try:
            existing = list(Settings.callback_manager.handlers) if Settings.callback_manager else []
            Settings.callback_manager = CallbackManager(_handlers_with_observra(existing))

            _original_callback_manager_init = CallbackManager.__init__

            def callback_manager_init(self: Any, handlers: Optional[list[Any]] = None) -> None:
                _original_callback_manager_init(self, _handlers_with_observra(handlers))

            CallbackManager.__init__ = callback_manager_init  # type: ignore[method-assign]
            _patch_function_tools()
            _patch_agent_steps()
            _patch_agent_workflow_run()
            _patched = True
        except Exception:  # noqa: BLE001
            logger.warning(
                "observra: failed to install LlamaIndex callback handler, skipping instrumentation",
                exc_info=True,
            )

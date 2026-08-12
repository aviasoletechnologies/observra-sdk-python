"""LangChain callback instrumentation for chains, agents, tools, retrievers, and LLMs.

``patch()`` adds one callback handler to every sync and async LangChain callback
manager. The handler records standard LangChain lifecycle callbacks without
requiring changes to application chains, agents, graphs, tools, or providers.
When a LangChain LLM callback is active, provider transport metadata enriches
that same LLM span instead of creating a duplicate provider child span.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from observra.config import get_config
from observra.tracing.context import activate_framework_llm_span, deactivate_framework_llm_span
from observra.tracing.conventions import Attr, SpanKind
from observra.tracing.tracer import (
    ObservraTracer,
    safe_add_event,
    safe_end_span,
    safe_set_attributes,
    get_tracer,
)

logger = logging.getLogger("observra")

_TESTED_MIN = (0, 2, 0)
_TESTED_MAX = (2, 0, 0)
_MAX_ATTR_TEXT_LEN = 4000
_MAX_STREAM_EVENTS = 100

_patched = False
_patch_lock = threading.Lock()
_original_langgraph_methods: Dict[str, Any] = {}


@dataclass
class _RunState:
    span: Any
    context_token: Any
    framework: str
    langchain_llm_token: Any = None
    stream_token_count: int = 0
    stream_parts: List[str] = field(default_factory=list)


def _parse_version(raw: str) -> Tuple[int, ...]:
    parts = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(character for character in chunk if character.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _installed_version() -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version

    for distribution in ("langchain", "langchain-core"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return None


def _tracer_or_none() -> Optional[ObservraTracer]:
    try:
        return get_tracer(get_config())
    except Exception:  # noqa: BLE001
        logger.warning(
            "observra: LangChain instrumentation active but observra.configure() "
            "was never called; skipping trace for this run",
            exc_info=True,
        )
        return None


class ObservraLangChainCallbackHandler(BaseCallbackHandler):
    """Map standard LangChain callback lifecycles to Observra spans.

    ``parent_run_id`` is used explicitly, not inferred only from ambient
    context. This keeps multi-agent, graph, async, and nested runnable spans
    connected even when LangChain dispatches callbacks from different contexts.
    """

    def __init__(self) -> None:
        self._runs: Dict[UUID, _RunState] = {}
        self._lock = threading.Lock()

    def _parent_span(self, parent_run_id: Optional[UUID]) -> Any:
        if parent_run_id is None:
            return None
        with self._lock:
            parent = self._runs.get(parent_run_id)
            return parent.span if parent is not None else None

    def _parent_framework(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        if parent_run_id is None:
            return None
        with self._lock:
            parent = self._runs.get(parent_run_id)
            return parent.framework if parent is not None else None

    def _start(
        self,
        run_id: UUID,
        name: str,
        kind: str,
        attributes: Dict[str, Any],
        parent_run_id: Optional[UUID] = None,
        framework: Optional[str] = None,
    ) -> None:
        tracer = _tracer_or_none()
        if tracer is None:
            return
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as trace_api

            parent_span = self._parent_span(parent_run_id)
            parent_framework = self._parent_framework(parent_run_id)
            framework_name = framework or parent_framework or "langchain"
            parent_token = None
            if parent_span is not None:
                parent_token = otel_context.attach(trace_api.set_span_in_context(parent_span))
            try:
                span = tracer.start_detached_span(
                    name,
                    kind,
                    {**attributes, Attr.FRAMEWORK: framework_name},
                )
            finally:
                if parent_token is not None:
                    otel_context.detach(parent_token)
            if span is None:
                return

            context_token = otel_context.attach(trace_api.set_span_in_context(span))
            framework_llm_token = (
                activate_framework_llm_span(span)
                if name == "llm.generate" and kind == SpanKind.LLM
                else None
            )
            with self._lock:
                self._runs[run_id] = _RunState(
                    span=span,
                    context_token=context_token,
                    framework=framework_name,
                    langchain_llm_token=framework_llm_token,
                )
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to start span for LangChain run", exc_info=True)

    def _end(
        self,
        run_id: UUID,
        attributes: Optional[Dict[str, Any]] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        with self._lock:
            state = self._runs.pop(run_id, None)
        if state is None:
            return

        try:
            from opentelemetry import context as otel_context

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
            if state.langchain_llm_token is not None:
                deactivate_framework_llm_span(state.langchain_llm_token)
            otel_context.detach(state.context_token)
        except Exception:  # noqa: BLE001
            logger.warning("observra: failed to end span for LangChain run", exc_info=True)

    def _event(self, run_id: Optional[UUID], name: str, attributes: Dict[str, Any]) -> None:
        if run_id is None:
            return
        with self._lock:
            state = self._runs.get(run_id)
        if state is not None:
            safe_add_event(state.span, name, attributes)

    # -- chain / agent -------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        qualified_name = ".".join((serialized or {}).get("id", []))
        name = kwargs.get("name") or (serialized or {}).get("name") or qualified_name or "chain"
        kind = SpanKind.AGENT if _is_agent(name, qualified_name, tags, metadata) else SpanKind.CHAIN
        self._start(
            run_id,
            name,
            kind,
            {Attr.INPUT_VALUE: _safe_str(inputs)},
            parent_run_id,
            _framework_name(name, qualified_name, tags, metadata),
        )

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, {Attr.OUTPUT_VALUE: _safe_str(outputs)})

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    def on_agent_action(self, action: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._event(run_id, "langchain.agent.action", {"agent.action": _safe_str(action)})

    def on_agent_finish(self, finish: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._event(run_id, "langchain.agent.finish", {"agent.finish": _safe_str(finish)})

    # -- LLM / chat / streaming ---------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        model_name = _extract_model_name(serialized, kwargs)
        self._start(
            run_id,
            "llm.generate",
            SpanKind.LLM,
            {
                Attr.LLM_MODEL_NAME: model_name,
                Attr.INPUT_VALUE: "\n".join(prompts)[:_MAX_ATTR_TEXT_LEN],
            },
            parent_run_id,
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        model_name = _extract_model_name(serialized, kwargs)
        flattened = [
            str(getattr(message, "content", message)) for batch in messages for message in batch
        ]
        self._start(
            run_id,
            "llm.generate",
            SpanKind.LLM,
            {
                Attr.LLM_MODEL_NAME: model_name,
                Attr.INPUT_VALUE: "\n".join(flattened)[:_MAX_ATTR_TEXT_LEN],
            },
            parent_run_id,
        )

    def on_llm_new_token(self, token: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return
            state.stream_token_count += 1
            if sum(len(part) for part in state.stream_parts) < _MAX_ATTR_TEXT_LEN:
                state.stream_parts.append(str(token))
            event_number = state.stream_token_count
        if event_number <= _MAX_STREAM_EVENTS:
            safe_add_event(
                state.span,
                "llm.stream.token",
                {"token": str(token)[:_MAX_ATTR_TEXT_LEN]},
            )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        output_text = _extract_llm_output_text(response)
        prompt_tokens, completion_tokens = _extract_llm_usage(response)
        self._end(
            run_id,
            {
                Attr.OUTPUT_VALUE: output_text,
                Attr.LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                Attr.LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
            },
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # -- tools ----------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "tool")
        self._start(
            run_id,
            f"tool.{name}",
            SpanKind.TOOL,
            {Attr.TOOL_NAME: name, Attr.TOOL_PARAMETERS: input_str[:_MAX_ATTR_TEXT_LEN]},
            parent_run_id,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, {Attr.TOOL_RESULT: _safe_str(output)})

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # -- retrieval ------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "retriever")
        self._start(
            run_id,
            f"retriever.{name}",
            SpanKind.RETRIEVER,
            {Attr.RETRIEVER_QUERY: query[:_MAX_ATTR_TEXT_LEN]},
            parent_run_id,
        )

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        document_count = len(documents) if hasattr(documents, "__len__") else None
        self._end(
            run_id,
            {
                Attr.RETRIEVER_DOCUMENT_COUNT: document_count,
                Attr.OUTPUT_VALUE: _safe_str(documents),
            },
        )

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # -- application-defined callback data ----------------------------

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._event(
            run_id,
            "langchain.custom_event",
            {Attr.LANGCHAIN_EVENT_NAME: name, "langchain.event.data": _safe_str(data)},
        )


def _framework_name(
    name: str,
    qualified_name: str,
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
) -> str:
    """Classify LangGraph callbacks and leave normal LangChain runs unchanged."""
    identity = " ".join(
        [name, qualified_name, " ".join(tags or []), _safe_str(metadata or {})]
    ).lower()
    return "langgraph" if "langgraph" in identity or "graph:step:" in identity else "langchain"


def _is_agent(
    name: str,
    qualified_name: str,
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
) -> bool:
    identity = " ".join(
        [name, qualified_name, " ".join(tags or []), _safe_str(metadata or {})]
    ).lower()
    if "langgraph" in identity and (
        "langgraph_node" in identity or "graph:step:" in identity
    ):
        return any(marker in identity for marker in ("agent", "supervisor", "multi_agent"))
    return any(marker in identity for marker in ("agent", "supervisor", "multi_agent", "langgraph"))


def _safe_str(value: Any) -> str:
    try:
        return json.dumps(value, default=str)[:_MAX_ATTR_TEXT_LEN]
    except Exception:  # noqa: BLE001
        return str(value)[:_MAX_ATTR_TEXT_LEN]


def _extract_model_name(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> Optional[str]:
    invocation_params = kwargs.get("invocation_params") or {}
    return (
        invocation_params.get("model")
        or invocation_params.get("model_name")
        or (serialized or {}).get("name")
    )


def _extract_llm_output_text(response: Any) -> Optional[str]:
    try:
        generations = response.generations
        texts = []
        for batch in generations:
            for generation in batch:
                text = getattr(generation, "text", None)
                if text:
                    texts.append(text)
                    continue
                message = getattr(generation, "message", None)
                if message is not None:
                    texts.append(getattr(message, "content", str(message)))
        return "\n".join(texts)[:_MAX_ATTR_TEXT_LEN] if texts else None
    except Exception:  # noqa: BLE001
        return None


def _extract_llm_usage(response: Any) -> Tuple[Optional[int], Optional[int]]:
    try:
        usage = (response.llm_output or {}).get("token_usage", {})
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    except Exception:  # noqa: BLE001
        return None, None


_handler = ObservraLangChainCallbackHandler()


def patch() -> None:
    """Idempotently patch LangChain callback manager construction."""
    global _patched
    with _patch_lock:
        if _patched:
            return

        installed = _installed_version()
        if installed is not None:
            version_tuple = _parse_version(installed)
            if not (_TESTED_MIN <= version_tuple < _TESTED_MAX):
                logger.warning(
                    "observra: langchain %s is outside the tested range [%s, %s) — "
                    "skipping instrumentation",
                    installed,
                    ".".join(map(str, _TESTED_MIN)),
                    ".".join(map(str, _TESTED_MAX)),
                )
                return

        try:
            from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
        except ImportError:
            logger.warning(
                "observra: langchain_core.callbacks.manager not importable, "
                "skipping instrumentation"
            )
            return

        _wrap_callback_manager_init(CallbackManager)
        _wrap_callback_manager_init(AsyncCallbackManager)
        _patch_langgraph_execution()
        _patched = True


def _config_with_langgraph_handler(config: Any) -> Any:
    """Add one inheritable handler to graph execution configuration."""
    if config is None:
        return {"callbacks": [_handler]}
    if not isinstance(config, dict):
        return config
    configured = dict(config)
    callbacks = configured.get("callbacks")
    if callbacks is None:
        configured["callbacks"] = [_handler]
    elif isinstance(callbacks, list):
        if not any(isinstance(handler, ObservraLangChainCallbackHandler) for handler in callbacks):
            configured["callbacks"] = [*callbacks, _handler]
    elif not any(
        isinstance(handler, ObservraLangChainCallbackHandler)
        for handler in getattr(callbacks, "handlers", [])
    ):
        configured["callbacks"] = [_handler]
    return configured


def _patch_langgraph_execution() -> None:
    """Ensure every public LangGraph graph execution propagates this handler."""
    try:
        from langgraph.pregel import Pregel
    except ImportError:
        return

    for method_name in ("invoke", "ainvoke", "stream", "astream"):
        if method_name in _original_langgraph_methods:
            continue
        original = getattr(Pregel, method_name)
        _original_langgraph_methods[method_name] = original

        def wrap(method: Any) -> Any:
            def execute(
                graph: Any,
                input: Any,
                config: Any = None,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                return method(graph, input, _config_with_langgraph_handler(config), *args, **kwargs)

            return execute

        setattr(Pregel, method_name, wrap(original))


def _wrap_callback_manager_init(manager_cls: Any) -> None:
    original_init = manager_cls.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if not any(
            isinstance(handler, ObservraLangChainCallbackHandler) for handler in self.handlers
        ):
            self.handlers.append(_handler)
        if not any(
            isinstance(handler, ObservraLangChainCallbackHandler)
            for handler in self.inheritable_handlers
        ):
            self.inheritable_handlers.append(_handler)

    patched_init.__observra_patched__ = True  # type: ignore[attr-defined]
    manager_cls.__init__ = patched_init

"""Single source of truth for span kinds and attribute names.

Modeled on OpenInference's semantic conventions. Every other module imports
these constants rather than hardcoding string literals, so a naming change
happens in exactly one place.
"""

from __future__ import annotations


class SpanKind:
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    AGENT = "AGENT"
    RETRIEVER = "RETRIEVER"


class Attr:
    # LLM call attributes
    LLM_PROVIDER = "llm.provider"
    LLM_MODEL_NAME = "llm.model_name"
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_LATENCY_MS = "llm.latency_ms"
    LLM_COST = "llm.cost"

    # I/O
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"

    # Tool call attributes
    TOOL_NAME = "tool.name"
    TOOL_PARAMETERS = "tool.parameters"
    TOOL_RESULT = "tool.result"

    # Retriever attributes
    RETRIEVER_QUERY = "retriever.query"
    RETRIEVER_DOCUMENT_COUNT = "retriever.document_count"

    # Agent-framework attribution
    FRAMEWORK = "framework.name"

    # Framework callback attributes
    LLM_STREAM_TOKEN_COUNT = "llm.stream.token_count"
    LANGCHAIN_EVENT_NAME = "langchain.event.name"
    LLAMA_INDEX_EVENT_NAME = "llama_index.event.name"

    # Guardrails
    GUARDRAIL_VIOLATION = "guardrail.violation"
    GUARDRAIL_ACTION = "guardrail.action"

    # Service / resource
    SERVICE_NAME = "service.name"

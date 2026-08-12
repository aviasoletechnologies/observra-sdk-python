"""GatewayExporter — prints finished spans instead of sending them anywhere.

Gateway ingest contract isn't wired up; this just surfaces what would be
exported so it's visible during development.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger("observra")


def _span_to_dict(span: ReadableSpan) -> Dict[str, Any]:
    ctx = span.get_span_context()
    trace_id = ctx.trace_id if ctx is not None else 0
    span_id = ctx.span_id if ctx is not None else 0
    parent = span.parent
    return {
        "trace_id": format(trace_id, "032x"),
        "span_id": format(span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent is not None else None,
        "name": span.name,
        "kind": (span.attributes or {}).get("observra.span_kind", span.kind.name),
        "start_time": span.start_time,
        "end_time": span.end_time,
        "attributes": dict(span.attributes or {}),
        "status": span.status.status_code.name,
        "events": [
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": dict(event.attributes or {}),
            }
            for event in (span.events or [])
        ],
    }


class GatewayExporter(SpanExporter):
    """OTel ``SpanExporter`` that prints finished spans instead of sending them anywhere.

    Never raises out of ``export`` (requirement #1) — printing is best-effort
    and swallowed on failure like every other telemetry-path operation.
    """

    def __init__(self, gateway_url: str, gateway_key: str) -> None:
        self._gateway_url = gateway_url
        self._gateway_key = gateway_key

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            for span in spans:
                print(json.dumps(_span_to_dict(span), indent=2, default=str))
        except Exception:  # noqa: BLE001 - telemetry path must never raise
            logger.warning("observra: failed to print spans", exc_info=True)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

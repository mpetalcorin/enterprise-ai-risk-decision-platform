from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_tracing(app) -> None:
    """Enable OpenTelemetry tracing when an OTLP endpoint is configured."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        logger.info("OpenTelemetry export disabled; OTEL_EXPORTER_OTLP_ENDPOINT is not configured")
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OpenTelemetry packages are unavailable; tracing disabled")
        return

    provider = TracerProvider(resource=Resource.create({"service.name": "enterprise-risk-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry tracing enabled", extra={"otlp_endpoint": endpoint})

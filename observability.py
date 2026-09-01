# observability.py
"""
Structured logging, request correlation and Prometheus metrics.
───────────────────────────────────────────────────────────────
Three capabilities a service needs before anyone will run it in production:

1. **Structured logs.** Human-formatted log lines cannot be queried. Emitting JSON
   means every line is a record with stable fields, so aggregation tools can filter
   by level, route, status or request id without regex parsing.
2. **Request correlation.** A request id is generated (or accepted from an inbound
   ``X-Request-ID``) and attached to every log line produced while handling that
   request, then echoed in the response. Without it, debugging a single failure in
   interleaved concurrent logs is guesswork.
3. **Metrics.** Counters and histograms for request volume, latency, errors and cache
   behaviour, exposed at ``/metrics``. These are what alerting and dashboards read;
   logs alone cannot answer "what is p99 latency right now".

Secrets are never logged: values whose keys look sensitive are redacted.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import (  # noqa: E402
    LOG_FORMAT, LOG_LEVEL, METRICS_ENABLED, SERVICE_NAME, SERVICE_VERSION,
)

# Correlation id for the request currently being handled by this task/thread.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_SENSITIVE_HINTS = ("key", "token", "secret", "password", "authorization", "cookie")


def redact(value: Any, hint: str = "") -> Any:
    """Mask anything whose name suggests it is a credential."""
    if any(h in hint.lower() for h in _SENSITIVE_HINTS) and value:
        text = str(value)
        return f"{text[:4]}...{text[-2:]}" if len(text) > 8 else "***"
    return value


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class RequestIdFilter(logging.Filter):
    """Inject the current correlation id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    _SKIP = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Preserve structured extras: log.info("...", extra={"route": "/query"}).
        for key, value in record.__dict__.items():
            if key not in self._SKIP and key not in payload and not key.startswith("_"):
                payload[key] = redact(value, key)
        return json.dumps(payload, default=str)


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """
    Install a single root handler with the chosen formatter.

    Replaces any handler installed by ``logging.basicConfig`` elsewhere in the
    codebase so log output is consistent no matter which module is imported first.
    """
    chosen_level = (level or LOG_LEVEL or "INFO").upper()
    chosen_fmt = (fmt or LOG_FORMAT or "json").lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if chosen_fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] — %(message)s"
        ))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, chosen_level, logging.INFO))

    # These libraries are extremely chatty at INFO and drown out application logs.
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "sentence_transformers",
                  "chromadb", "datasets", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class _NoopMetric:
    """Stand-in so instrumented code needs no conditional branches."""

    def labels(self, *_a, **_k): return self
    def inc(self, *_a, **_k): return None
    def observe(self, *_a, **_k): return None
    def set(self, *_a, **_k): return None


class Metrics:
    """
    Prometheus collectors for the serving path.

    Degrades to no-ops when ``prometheus_client`` is absent or metrics are disabled,
    so instrumentation can be added unconditionally at call sites.
    """

    def __init__(self, enabled: bool = METRICS_ENABLED):
        self.enabled = False
        self.requests = _NoopMetric()
        self.latency = _NoopMetric()
        self.errors = _NoopMetric()
        self.cache_events = _NoopMetric()
        self.cache_entries = _NoopMetric()
        self.inflight = _NoopMetric()
        self.rag_latency = _NoopMetric()
        self._registry = None

        if not enabled:
            return
        try:
            from prometheus_client import (
                CollectorRegistry, Counter, Gauge, Histogram,
            )
            self._registry = CollectorRegistry(auto_describe=True)
            self.requests = Counter(
                "rag_http_requests_total", "HTTP requests.",
                ["method", "route", "status"], registry=self._registry,
            )
            self.latency = Histogram(
                "rag_http_request_duration_seconds", "HTTP request latency.",
                ["method", "route"],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
                registry=self._registry,
            )
            self.errors = Counter(
                "rag_errors_total", "Handled errors.",
                ["route", "kind"], registry=self._registry,
            )
            self.cache_events = Counter(
                "rag_cache_events_total", "Semantic cache events.",
                ["event"], registry=self._registry,
            )
            self.cache_entries = Gauge(
                "rag_cache_entries", "Live semantic cache entries.",
                registry=self._registry,
            )
            self.inflight = Gauge(
                "rag_inflight_requests", "Requests currently being served.",
                registry=self._registry,
            )
            self.rag_latency = Histogram(
                "rag_pipeline_duration_seconds", "Retrieval+generation latency.",
                ["architecture", "cache_hit"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
                registry=self._registry,
            )
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).info(
                "Metrics disabled (%s) — install prometheus-client to enable.", exc
            )

    def render(self) -> tuple:
        """Return ``(body, content_type)`` for the /metrics endpoint."""
        if not self.enabled or self._registry is None:
            return b"# metrics disabled\n", "text/plain; charset=utf-8"
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return generate_latest(self._registry), CONTENT_TYPE_LATEST


# Process-wide metrics instance.
METRICS = Metrics()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

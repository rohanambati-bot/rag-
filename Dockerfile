# syntax=docker/dockerfile:1.7
# ── Stage 1: builder (compile any C extensions, then discard build deps) ──────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ── Stage 2: runtime (slim, no build tools, non-root) ─────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT_API=8000 \
    PORT_UI=8501

# Install runtime-only deps (curl needed for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system --gid 1001 appuser && \
    useradd  --system --uid 1001 --gid appuser --create-home appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=appuser:appuser . .

# Create writable dirs for the non-root user
RUN mkdir -p /app/data /app/outputs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# Liveness probe. Deliberately /healthz (process up) and NOT /readyz: a container
# that is merely still warming its models should not be killed and restarted, which
# would loop forever. Readiness is what the load balancer polls, and docker-compose
# overrides this per service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:${PORT_API}/healthz || exit 1

# Default: launch Streamlit. Override CMD to run the API instead.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

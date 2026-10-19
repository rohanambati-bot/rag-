# syntax=docker/dockerfile:1.7
# ── Stage 1: builder (compile any C extensions, then discard build deps) ──────
FROM python:3.12.14-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock ./
RUN pip install --upgrade pip && \
    pip install --prefix=/install --requirement requirements.lock


# ── Stage 2: runtime (slim, no build tools, non-root) ─────────────────────────
FROM python:3.12.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT_API=8000 \
    PORT_UI=8501

# Install runtime-only deps (curl needed for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
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

# The image serves the authenticated API, never the unauthenticated local-only UI.
# Liveness remains public even when the corpus or authentication is not ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:${PORT_API}/healthz || exit 1

# An API key is still required; binding a container interface is not authentication.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]

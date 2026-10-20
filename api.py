# api.py
"""
RAG Evaluation Framework — HTTP API
───────────────────────────────────
Production characteristics:

* **Auth fails closed.** Protected routes require a matching ``X-API-Key``. If
  ``API_KEY`` is unset the routes return 503 rather than serving openly; local
  development opts out explicitly via ``ALLOW_UNAUTHENTICATED=true``. Comparison is
  constant-time to avoid leaking the key through timing.
* **Async with bounded concurrency.** Retrieval and generation are blocking CPU/IO
  work, so they run in a worker thread with a semaphore and a per-request timeout.
  A slow upstream sheds load (503/504) instead of exhausting the event loop.
* **One retrieval stack.** All architectures and the corpus come from
  ``serving.CONTEXT``, the same MS MARCO store the CLI evaluates, so the served
  system matches the measured system.
* **Observable.** JSON logs carry a correlation id; ``/metrics`` exposes Prometheus
  counters and histograms; liveness (``/healthz``) is separate from readiness
  (``/readyz``) so orchestrators can distinguish "alive" from "able to serve".
* **Graceful lifecycle.** Models are warmed at startup and released on shutdown.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from configs.config import (  # noqa: E402
    ALLOW_UNAUTHENTICATED, ALLOWED_ORIGINS, API_BIND_HOST, API_KEY, MAX_CONCURRENT_REQUESTS,
    MAX_QUESTION_LENGTH, RATE_LIMIT_PER_MINUTE, REQUEST_TIMEOUT_SECONDS,
    SERVICE_NAME, SERVICE_VERSION, VECTOR_STORE_DIR, validate_runtime_config,
)
from observability import (  # noqa: E402
    METRICS, configure_logging, get_logger, new_request_id, request_id_var,
)
from serving import CONTEXT  # noqa: E402
from security_boundary import is_loopback_literal, valid_api_key  # noqa: E402
from phase3.semantic_cache import GenerationFencedError, require_generation_cache  # noqa: E402

configure_logging()
logger = get_logger("api")

# Bounds the number of concurrent blocking pipeline executions.
_WORK_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# ─── Lifecycle ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Bind the port immediately and warm the stack in the background.

    Warmup is genuinely slow — embedding model load, an 11.5k-chunk corpus, BM25
    indexing and (for GRACE-Ω) LLM-generated RAPTOR summaries. Doing that inline
    would delay the listening socket by a minute or more, so ``/healthz`` and
    ``/readyz`` would be unreachable during startup and an orchestrator would see a
    dead container rather than a warming one.

    Warming in a background task means probes answer straight away: liveness is
    green, readiness reports 503 until the stack is genuinely usable.
    """
    for problem in validate_runtime_config():
        logger.warning("Config check: %s", problem)
    logger.info("Starting %s v%s.", SERVICE_NAME, SERVICE_VERSION)

    # run_in_executor returns a Future, not a coroutine, so it is scheduled with
    # ensure_future rather than create_task (which rejects Futures under uvloop).
    warmup_generation = CONTEXT.begin_warmup()
    warm_future = asyncio.ensure_future(
        asyncio.get_running_loop().run_in_executor(None, CONTEXT.warmup, warmup_generation)
    )
    try:
        yield
    finally:
        if not warm_future.done():
            warm_future.cancel()
        CONTEXT.shutdown()
        logger.info("Shutdown complete.")


limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"])

app = FastAPI(
    title="RAG Evaluation Framework API",
    description=(
        "REST interface for the RAG evaluation framework: every registered "
        "architecture, qrels-graded metrics, and a vector-indexed semantic cache."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    METRICS.errors.labels(route=request.url.path, kind="rate_limited").inc()
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# CORS — explicit origins only. Never combine "*" with credentials.
if "*" in ALLOWED_ORIGINS:
    logger.warning("ALLOWED_ORIGINS contains '*'; disabling allow_credentials.")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
else:
    app.add_middleware(
        CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )


# ─── Middleware: correlation id, metrics, access log ──────────────────────────

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Attach a request id, time the request, and record metrics for every route."""
    incoming_rid = request.headers.get("X-Request-ID", "")
    rid = incoming_rid if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", incoming_rid) else new_request_id()
    token = request_id_var.set(rid)
    route = request.url.path
    started = time.perf_counter()
    METRICS.inflight.inc()
    status_code = 500
    try:
        # Protect the whole non-probe surface before body parsing or validation.
        # CORS preflight performs no protected work and needs no credentials.
        preflight = request.method == "OPTIONS" and "access-control-request-method" in request.headers
        if route not in {"/", "/health", "/healthz", "/readyz"} and not preflight:
            try:
                require_api_key(request.headers.get("X-API-Key"), request)
            except HTTPException as exc:
                response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        METRICS.errors.labels(route=route, kind="unhandled").inc()
        logger.exception("Unhandled error.", extra={"route": route})
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error."},
            headers={"X-Request-ID": rid},
        )
    finally:
        elapsed = time.perf_counter() - started
        METRICS.inflight.dec()
        METRICS.latency.labels(method=request.method, route=route).observe(elapsed)
        METRICS.requests.labels(
            method=request.method, route=route, status=str(status_code)
        ).inc()
        logger.info(
            "request", extra={
                "route": route, "method": request.method,
                "status": status_code, "duration_ms": round(elapsed * 1000, 2),
            },
        )
        request_id_var.reset(token)


# ─── Auth (fails closed) ──────────────────────────────────────────────────────

def _auth_config_problem() -> Optional[str]:
    if API_KEY:
        if not valid_api_key(API_KEY):
            return "API_KEY configuration is invalid."
        return None
    if not ALLOW_UNAUTHENTICATED:
        return "API_KEY is not configured; protected routes are unavailable."
    if not is_loopback_literal(API_BIND_HOST):
        return "Unauthenticated access requires a numeric loopback HOST."
    return None


def require_api_key(x_api_key: Optional[str] = Header(default=None),
                    request: Request = None) -> None:
    """
    Enforce ``X-API-Key`` on protected routes.

    Fails **closed**: with no ``API_KEY`` configured the route is unavailable (503)
    unless ``ALLOW_UNAUTHENTICATED=true`` was set deliberately. The previous
    behaviour served every protected route openly and only logged a warning.
    """
    problem = _auth_config_problem()
    if problem:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=problem,
        )
    if not API_KEY:
        # Never trust X-Forwarded-For for this opt-out. Uvicorn proxy headers must
        # stay disabled for local dev. A local tunnel can still mask remote use.
        forwarded = request is not None and any(
            name == "forwarded" or name.startswith("x-forwarded-")
            for name in request.headers
        )
        if (request is None or request.client is None
                or not is_loopback_literal(request.client.host) or forwarded):
            raise HTTPException(status_code=403, detail="Unauthenticated access is local-only.")
        return
    # hmac.compare_digest avoids leaking the key via response timing.
    duplicate = request is not None and len(request.headers.getlist("x-api-key")) != 1
    if (duplicate or not valid_api_key(x_api_key)
            or not hmac.compare_digest(x_api_key.encode("ascii"), API_KEY.encode("ascii"))):
        METRICS.errors.labels(route="auth", kind="unauthorized").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )


def require_ready() -> None:
    """Reject work when the retrieval stack is not usable."""
    if not CONTEXT.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready; warming up or temporarily unavailable.",
        )


class RateLimit:
    """
    Per-client, per-route rate limit expressed as a FastAPI dependency.

    Used instead of slowapi's ``@limiter.limit`` decorator on async endpoints:
    that decorator does not preserve the wrapped function's signature, so FastAPI
    stops seeing the Pydantic body model and rejects valid requests with 422. A
    dependency composes correctly with async handlers and is directly unit-testable.

    Global default limits still come from ``SlowAPIMiddleware``; this adds tighter
    per-route budgets (e.g. evaluation, which is far more expensive than a query).
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, request: Request) -> None:
        client = get_remote_address(request) or "unknown"
        key = f"{client}:{request.url.path}"
        now = time.monotonic()
        async with self._lock:
            window = [t for t in self._hits.get(key, []) if now - t < self.window_seconds]
            if len(window) >= self.max_requests:
                METRICS.errors.labels(route=request.url.path, kind="rate_limited").inc()
                retry_after = int(self.window_seconds - (now - window[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded ({self.max_requests}/"
                           f"{int(self.window_seconds)}s).",
                    headers={"Retry-After": str(retry_after)},
                )
            window.append(now)
            self._hits[key] = window
            # Opportunistic cleanup so the dict cannot grow unbounded.
            if len(self._hits) > 10_000:
                for stale_key in [
                    k for k, v in self._hits.items()
                    if not v or now - v[-1] > self.window_seconds
                ]:
                    self._hits.pop(stale_key, None)


# Evaluation is orders of magnitude more expensive than a single query.
evaluate_rate_limit = RateLimit(max_requests=5, window_seconds=60)


async def run_blocking(fn, *args, timeout: float = REQUEST_TIMEOUT_SECONDS):
    """
    Execute blocking work off the event loop, bounded and time-limited.

    Retrieval and LLM calls are synchronous; without this a single slow upstream
    would stall the whole server. The semaphore caps concurrent heavy work and the
    timeout converts a hang into a 504.
    """
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(_WORK_SEMAPHORE.acquire(), timeout=5)
    except asyncio.TimeoutError:
        METRICS.errors.labels(route="queue", kind="overloaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is at capacity; retry shortly.",
        ) from None
    executor_future = None
    released_on_completion = False

    def _release_when_done(_future) -> None:
        _WORK_SEMAPHORE.release()

    try:
        executor_future = loop.run_in_executor(None, fn, *args)
        # ``wait_for`` cancels the awaitable it wraps on timeout. Shielding the
        # executor future is essential: cancelling that asyncio wrapper does not
        # stop a synchronous function already running in a thread, and releasing
        # the semaphore here would admit another task on top of it.
        return await asyncio.wait_for(
            asyncio.shield(executor_future), timeout=timeout
        )
    except asyncio.TimeoutError:
        # The thread keeps running after the request receives its 504. Transfer
        # ownership of the semaphore to a completion callback so the concurrency
        # bound covers the actual executor work, not just the request lifetime.
        released_on_completion = True
        executor_future.add_done_callback(_release_when_done)
        METRICS.errors.labels(route="pipeline", kind="timeout").inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request exceeded {timeout:.0f}s budget.",
        ) from None
    except asyncio.CancelledError:
        # Client disconnects cancel the coroutine, not the synchronous function
        # already running in the executor. Keep the slot owned by that function
        # until it really finishes; otherwise repeated disconnects can bypass the
        # concurrency bound and exhaust the worker pool.
        if executor_future.done():
            _WORK_SEMAPHORE.release()
            released_on_completion = True
        else:
            released_on_completion = True
            executor_future.add_done_callback(_release_when_done)
        raise
    finally:
        if executor_future is None:
            _WORK_SEMAPHORE.release()
        elif not released_on_completion:
            if executor_future.done():
                _WORK_SEMAPHORE.release()
            else:
                # Request cancellation has the same ownership requirement as a
                # timeout: the underlying synchronous task may still be running.
                executor_future.add_done_callback(_release_when_done)


def _resolve_architecture_sync(name: str):
    """Blocking resolution — may construct the architecture on first use."""
    try:
        return CONTEXT.architecture(name)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown architecture {name!r}. "
                   f"Available: {CONTEXT.available_keys()}",
        ) from None
    except Exception as exc:  # noqa: BLE001 — construction failed
        METRICS.errors.labels(route="architecture", kind="construction_failed").inc()
        logger.exception("Architecture construction failed.",
                         extra={"architecture": name})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            # Do not send exception strings (paths, SDK request details, or
            # provider metadata) to callers. The correlated server log has the
            # diagnostic; the public response stays intentionally generic.
            detail=f"Architecture {name!r} is temporarily unavailable.",
        ) from exc


async def resolve_architecture(name: str):
    """
    Resolve an architecture without blocking the event loop.

    Names are validated first (cheap, in-memory) so a bad request fails fast. Actual
    construction is dispatched to the thread pool because building GRACE-Ω generates
    an LLM-backed RAPTOR tree — measured at over two minutes. Doing that inline froze
    every other in-flight request; off-loop it only delays the caller that asked for
    it, and subsequent callers reuse the cached instance.
    """
    if name not in CONTEXT.available_keys():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown architecture {name!r}. "
                   f"Available: {CONTEXT.available_keys()}",
        )
    # Already built: return immediately without a thread hop.
    built = CONTEXT.architectures()
    if name in built:
        return built[name]
    return await run_blocking(
        _resolve_architecture_sync, name,
        timeout=max(REQUEST_TIMEOUT_SECONDS, 300),  # first build is slow
    )


# ─── Schemas ──────────────────────────────────────────────────────────────────

def _require_non_blank_question(value: str) -> str:
    """Reject whitespace-only questions consistently across request schemas."""
    if not value.strip():
        raise ValueError("question must not be blank")
    return value

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    architecture: str = Field("naive", description="See GET /architectures")
    use_cache: bool = True

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        return _require_non_blank_question(v)


class QueryResponse(BaseModel):
    architecture: str
    question: str
    answer: str
    contexts: List[str]
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float
    cache_hit: bool = False
    similarity_score: Optional[float] = None
    request_id: Optional[str] = None


class ABRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    arch_a: str = "naive"
    arch_b: str = "rerank"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        return _require_non_blank_question(v)


class ABVariantResult(BaseModel):
    architecture: str
    answer: str
    latency_ms: float
    num_chunks: int


class ABResponse(BaseModel):
    question: str
    variant_a: ABVariantResult
    variant_b: ABVariantResult
    delta_latency_ms: float


class EvalRequest(BaseModel):
    architectures: List[str] = Field(default_factory=lambda: ["naive"], max_length=4)
    samples: int = Field(15, ge=1, le=200)
    use_ragas: bool = False

    @field_validator("architectures")
    @classmethod
    def _non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("architectures must not be empty")
        return v


class EvalResponse(BaseModel):
    samples: int
    architectures: List[str]
    summary: List[Dict[str, Any]]


class CacheLookup(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    total_queries: int
    hit_rate: float
    entries: int
    capacity: int
    avg_lookup_ms: float
    backend: str
    ann: bool
    evictions: int
    errors: int
    total_latency_saved_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    llm_available: bool
    vector_store_exists: bool
    auth_enabled: bool


# ─── Health / meta ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return f"""<!doctype html><html><head><title>{SERVICE_NAME}</title>
    <style>
      body{{font-family:system-ui,sans-serif;max-width:720px;margin:48px auto;
           padding:0 24px;line-height:1.6;color:#222;}}
      h1{{background:linear-gradient(90deg,#845ec2,#4ea8de);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         margin-bottom:4px;}}
      .sub{{color:#666;margin-top:0;}}
      a{{display:block;padding:14px 18px;margin:10px 0;background:#f5f6fa;
        border-left:4px solid #845ec2;border-radius:6px;text-decoration:none;
        color:#222;}}
      a:hover{{background:#eef0fa;}}
    </style></head><body>
      <h1>RAG Evaluation Framework</h1>
      <p class="sub">v{SERVICE_VERSION} &middot; every registered architecture
      &middot; qrels-graded metrics &middot; vector-indexed semantic cache.</p>
      <a href="/docs">Interactive API docs (Swagger)</a>
      <a href="/architectures">Available architectures</a>
      <a href="/healthz">Liveness</a>
      <a href="/readyz">Readiness</a>
      <a href="/metrics">Prometheus metrics</a>
    </body></html>"""


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness: the process is up. Does not assert the stack is usable."""
    from configs.config import GROQ_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        llm_available=bool(OPENROUTER_API_KEY or GROQ_API_KEY or MISTRAL_API_KEY),
        vector_store_exists=Path(VECTOR_STORE_DIR).exists(),
        auth_enabled=bool(API_KEY and valid_api_key(API_KEY)),
    )


@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def health_legacy() -> HealthResponse:
    """Backwards-compatible alias for /healthz."""
    return healthz()


@app.get("/readyz")
def readyz(response: Response) -> Dict[str, Any]:
    """
    Readiness: can this instance actually serve traffic?

    Returns 503 until warmup succeeds, so an orchestrator keeps a broken or still-
    loading instance out of the load-balancer rotation.
    """
    payload = dict(CONTEXT.readiness())
    problem = _auth_config_problem()
    if problem:
        payload["ready"] = False
        payload["reason"] = "Authentication configuration unavailable."
    elif not payload["ready"]:
        # Warmup exceptions can contain private paths, upstream URLs or secrets.
        payload["reason"] = "Warming up or temporarily unavailable."
    else:
        payload["reason"] = None
    if not payload["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    CONTEXT.refresh_cache_gauge()
    body, content_type = METRICS.render()
    return Response(content=body, media_type=content_type)


@app.get("/architectures")
def list_architectures() -> Dict[str, Any]:
    """
    Every servable architecture and whether its retrieval needs an LLM.

    Read from declared metadata, so listing costs nothing — instantiating the set
    would trigger LLM-backed RAPTOR builds.
    """
    from module2.rag_architectures import ARCHITECTURE_REGISTRY

    keys = CONTEXT.available_keys()
    built = set(CONTEXT.architectures().keys())
    return {
        "architectures": keys,
        "count": len(keys),
        "detail": {
            key: {
                "name": ARCHITECTURE_REGISTRY[key]["name"],
                "llm_free_retrieval": ARCHITECTURE_REGISTRY[key]["llm_free"],
                # Lazily-built architectures pay a one-off cost on first request.
                "loaded": key in built,
            }
            for key in keys if key in ARCHITECTURE_REGISTRY
        },
    }


# ─── Core endpoints ───────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse,
          dependencies=[Depends(require_api_key), Depends(require_ready)])
async def query(request: Request, req: QueryRequest) -> QueryResponse:
    if req.architecture not in CONTEXT.available_keys():
        raise HTTPException(status_code=400, detail="Unknown architecture.")
    # Even cache-bypass responses are fenced. Capture before lazy constructors,
    # retrieval and generation; their work can outlive a concurrent clear.
    cache = require_generation_cache(CONTEXT.cache(req.architecture))
    token = await run_blocking(cache.begin_generation, timeout=15)

    if req.use_cache:
        cached = await run_blocking(cache.lookup_generation, token, req.question, timeout=15)
        if cached:
            METRICS.cache_events.labels(event="hit").inc()
            METRICS.rag_latency.labels(
                architecture=req.architecture, cache_hit="true"
            ).observe(cached.get("latency_ms", 0) / 1000)
            return QueryResponse(
                architecture=req.architecture, question=req.question,
                answer=cached["answer"], contexts=cached.get("contexts", []),
                citations=cached.get("citations", []),
                latency_ms=cached.get("latency_ms", 0.0), cache_hit=True,
                similarity_score=cached.get("similarity_score"),
                request_id=request_id_var.get(),
            )
        METRICS.cache_events.labels(event="miss").inc()

    arch = await resolve_architecture(req.architecture)
    result = await run_blocking(arch.run, req.question)
    METRICS.rag_latency.labels(
        architecture=req.architecture, cache_hit="false"
    ).observe(result.get("latency_ms", 0) / 1000)

    if req.use_cache:
        await run_blocking(
            cache.publish_generation, token, req.question, result["answer"],
            result.get("contexts", []), result.get("citations", []), timeout=15,
        )
    await run_blocking(cache.validate_generation, token, timeout=15)

    return QueryResponse(
        architecture=req.architecture, question=req.question,
        answer=result["answer"], contexts=result.get("contexts", []),
        citations=result.get("citations", []),
        latency_ms=result.get("latency_ms", 0.0), cache_hit=False,
        request_id=request_id_var.get(),
    )


@app.post("/ab-test", response_model=ABResponse,
          dependencies=[Depends(require_api_key), Depends(require_ready)])
async def ab_test(request: Request, req: ABRequest) -> ABResponse:
    fences = []
    for name in (req.arch_a, req.arch_b):
        if name not in CONTEXT.available_keys():
            raise HTTPException(status_code=400, detail="Unknown architecture.")
        cache = require_generation_cache(CONTEXT.cache(name))
        token = await run_blocking(cache.begin_generation, timeout=15)
        fences.append((cache, token))
    a = await resolve_architecture(req.arch_a)
    b = await resolve_architecture(req.arch_b)
    res_a, res_b = await asyncio.gather(
        run_blocking(a.run, req.question),
        run_blocking(b.run, req.question),
    )
    for cache, token in fences:
        await run_blocking(cache.validate_generation, token, timeout=15)
    return ABResponse(
        question=req.question,
        variant_a=ABVariantResult(
            architecture=req.arch_a, answer=res_a["answer"],
            latency_ms=res_a["latency_ms"], num_chunks=len(res_a.get("contexts", [])),
        ),
        variant_b=ABVariantResult(
            architecture=req.arch_b, answer=res_b["answer"],
            latency_ms=res_b["latency_ms"], num_chunks=len(res_b.get("contexts", [])),
        ),
        delta_latency_ms=res_b["latency_ms"] - res_a["latency_ms"],
    )


@app.post("/evaluate", response_model=EvalResponse,
          dependencies=[Depends(require_api_key), Depends(require_ready),
                        Depends(evaluate_rate_limit)])
async def evaluate(request: Request, req: EvalRequest) -> EvalResponse:
    import json as _json

    from module3.evaluation import EvaluationPipeline, RAGASEvaluator, build_qrels

    syn_path = ROOT / "data" / "synthetic_dataset.json"
    if not syn_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="synthetic_dataset.json not found. Run preprocessing first.",
        )
    synthetic = _json.loads(syn_path.read_text())[: req.samples]

    ragas = None
    if req.use_ragas:
        try:
            ragas = RAGASEvaluator()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAGAS evaluator temporarily unavailable.",
            ) from exc

    archs = {name: await resolve_architecture(name) for name in req.architectures}

    def _run() -> List[Dict[str, Any]]:
        # Relevance judgements enable rank-aware metrics (nDCG@10, MRR@10, Recall@k).
        qrels = build_qrels(CONTEXT.chunks())
        # HTTP jobs are request-scoped.  Do not let concurrent callers overwrite the
        # CLI's historical outputs/eval_results.csv artifact.
        pipeline = EvaluationPipeline(ragas_evaluator=ragas, output_path=os.devnull)
        results = {
            name: pipeline.evaluate_architecture(
                architecture_name=name, rag_system=arch,
                synthetic_dataset=synthetic, max_samples=req.samples, qrels=qrels,
            )
            for name, arch in archs.items()
        }
        return pipeline.compare_architectures(results).to_dict(orient="records")

    # Evaluation is long-running; allow a wider budget than a single query.
    summary = await run_blocking(_run, timeout=max(REQUEST_TIMEOUT_SECONDS, 600))
    return EvalResponse(
        samples=req.samples, architectures=list(archs), summary=summary
    )


# ─── Cache management ─────────────────────────────────────────────────────────

@app.get("/cache/stats", response_model=CacheStatsResponse,
         dependencies=[Depends(require_api_key)])
def cache_stats(architecture: str = "naive") -> CacheStatsResponse:
    if architecture not in CONTEXT.available_keys():
        raise HTTPException(status_code=400, detail="Unknown cache architecture.")
    stats = CONTEXT.cache(architecture).stats()
    return CacheStatsResponse(**{
        field: stats.get(field, 0) for field in CacheStatsResponse.model_fields
    })


@app.post("/cache/lookup", dependencies=[Depends(require_api_key), Depends(require_ready)])
async def cache_lookup(request: Request, req: CacheLookup,
                       architecture: str = "naive") -> Dict[str, Any]:
    if architecture not in CONTEXT.available_keys():
        raise HTTPException(status_code=400, detail="Unknown cache architecture.")
    cache = require_generation_cache(CONTEXT.cache(architecture))
    token = await run_blocking(cache.begin_generation, timeout=15)
    result = await run_blocking(cache.lookup_generation, token, req.query, timeout=15)
    return {"hit": False} if result is None else {"hit": True, **result}


@app.delete("/cache", dependencies=[Depends(require_api_key)])
def cache_clear(architecture: str = "naive") -> Dict[str, str]:
    if architecture not in CONTEXT.available_keys():
        raise HTTPException(status_code=400, detail="Unknown cache architecture.")
    if require_generation_cache(CONTEXT.cache(architecture)).clear() is not True:
        raise HTTPException(status_code=503, detail="Shared cache invalidation could not be confirmed.")
    logger.info("Cache cleared by API request.")
    return {"status": "cleared"}


@app.exception_handler(GenerationFencedError)
async def generation_fenced_handler(_request: Request, _exc: GenerationFencedError):
    return JSONResponse(status_code=503, content={
        "detail": "Generation was revoked or its current authority could not be confirmed. Retry the request.",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=API_BIND_HOST,  # bind locally by default
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_config=None,  # keep our JSON logging
        proxy_headers=False,  # no implicit forwarded-header trust for dev opt-out
    )

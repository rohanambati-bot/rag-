#!/usr/bin/env python3
# api.py — FastAPI REST backend for the RAG framework
"""
Run with:
    uvicorn api:app --reload --port 8000

Auto-generated docs:
    http://localhost:8000/docs       (Swagger)
    http://localhost:8000/redoc      (ReDoc)

Security:
    • API-key authentication via the X-API-Key header (set API_KEY env var).
      If API_KEY is empty, the API runs in open mode (dev only) and logs a
      WARNING on every protected request.
    • Per-IP rate limiting via slowapi (defaults to RATE_LIMIT_PER_MINUTE).
    • CORS restricted to ALLOWED_ORIGINS (no wildcard with credentials).
    • Bounded inputs (question length, sample count, architecture list size).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("api")

from configs.config import (  # noqa: E402
    ALLOWED_ORIGINS, API_KEY, MAX_QUESTION_LENGTH,
    MISTRAL_API_KEY, RATE_LIMIT_PER_MINUTE, VECTOR_STORE_DIR,
)

# ─── App & middleware ─────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"])

app = FastAPI(
    title="RAG Evaluation Framework API",
    description=(
        "REST interface for the RAG evaluation framework "
        "(4 architectures + custom RAGAS-style metrics + semantic cache)."
    ),
    version="1.1.0",
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# CORS — explicit origins only. Never combine "*" with allow_credentials.
if "*" in ALLOWED_ORIGINS:
    logger.warning("ALLOWED_ORIGINS contains '*'; disabling allow_credentials for safety.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


# ─── Auth dependency ──────────────────────────────────────────────────────────

def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    If API_KEY is configured, require X-API-Key header to match.
    If API_KEY is unset (dev mode), log a warning but allow.
    """
    if not API_KEY:
        logger.warning("API_KEY is not set — running in OPEN mode. Do not deploy this way.")
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )


# ─── Lazy singletons ──────────────────────────────────────────────────────────

_state: Dict[str, Any] = {}
_VALID_ARCHS = {"naive", "hybrid", "parent_child", "crag"}


def _embedding():
    if "embedding" not in _state:
        from module1.data_preprocessing import get_embedding_model
        _state["embedding"] = get_embedding_model()
    return _state["embedding"]


def _vectorstore():
    if "vs" not in _state:
        from module1.data_preprocessing import (
            build_vector_store, chunk_documents, load_squad_documents, load_vector_store,
        )
        emb = _embedding()
        if Path(VECTOR_STORE_DIR).exists() and any(Path(VECTOR_STORE_DIR).iterdir()):
            _state["vs"] = load_vector_store(emb)
        else:
            logger.info("Bootstrapping vector store from SQuAD (first run)…")
            docs = load_squad_documents()
            chunks = chunk_documents(docs)
            _state["vs"] = build_vector_store(chunks, emb)
    return _state["vs"]


def _chunks():
    if "chunks" not in _state:
        from module1.data_preprocessing import chunk_documents, load_squad_documents
        _state["chunks"] = chunk_documents(load_squad_documents())
    return _state["chunks"]


def _architectures():
    if "archs" not in _state:
        from module2.rag_architectures import (
            CorrectiveRAG, HybridMultiQueryRAG, NaiveRAG, ParentChildRAG,
        )
        vs = _vectorstore()
        ch = _chunks()
        _state["archs"] = {
            "naive":        NaiveRAG(vectorstore=vs),
            "hybrid":       HybridMultiQueryRAG(vectorstore=vs, chunks=ch),
            "parent_child": ParentChildRAG(vectorstore=vs, chunks=ch),
            "crag":         CorrectiveRAG(vectorstore=vs),
        }
    return _state["archs"]


def _cache():
    if "cache" not in _state:
        from phase3.cicd_pipeline import SemanticCache
        _state["cache"] = SemanticCache(embedding_model=_embedding())
    return _state["cache"]


# ─── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    architecture: str = Field("crag", description="naive | hybrid | parent_child | crag")
    use_cache: bool = True

    @field_validator("architecture")
    @classmethod
    def _arch_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_ARCHS:
            raise ValueError(f"architecture must be one of {sorted(_VALID_ARCHS)}")
        return v


class QueryResponse(BaseModel):
    architecture: str
    question: str
    answer: str
    contexts: List[str]
    latency_ms: float
    cache_hit: bool = False
    similarity_score: Optional[float] = None


class ABRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    arch_a: str = "naive"
    arch_b: str = "crag"

    @field_validator("arch_a", "arch_b")
    @classmethod
    def _arch_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_ARCHS:
            raise ValueError(f"architecture must be one of {sorted(_VALID_ARCHS)}")
        return v


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
    architectures: List[str] = Field(default_factory=lambda: list(_VALID_ARCHS), max_length=4)
    samples: int = Field(15, ge=1, le=200)
    use_ragas: bool = False

    @field_validator("architectures")
    @classmethod
    def _archs_must_be_valid(cls, v: List[str]) -> List[str]:
        invalid = set(v) - _VALID_ARCHS
        if invalid:
            raise ValueError(f"unknown architectures: {sorted(invalid)}")
        if not v:
            raise ValueError("architectures must not be empty")
        return v


class EvalResponse(BaseModel):
    samples: int
    architectures: List[str]
    summary: List[Dict[str, Any]]


class CacheLookup(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)


class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    total_queries: int
    hit_rate: float
    total_latency_saved_ms: float


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    vector_store_exists: bool
    auth_enabled: bool


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> str:
    return """
    <!DOCTYPE html>
    <html><head><title>RAG Framework API</title>
    <style>
      body{font-family:system-ui,sans-serif;max-width:720px;margin:48px auto;
           padding:0 24px;line-height:1.6;color:#222;}
      h1{background:linear-gradient(90deg,#845ec2,#4ea8de);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         margin-bottom:4px;}
      .sub{color:#666;margin-top:0;}
      a{display:block;padding:14px 18px;margin:10px 0;background:#f5f6fa;
        border-left:4px solid #845ec2;border-radius:6px;text-decoration:none;
        color:#222;transition:transform .15s;}
      a:hover{transform:translateX(4px);background:#eef0fa;}
      code{background:#f0f0f0;padding:2px 6px;border-radius:4px;font-size:.9em;}
    </style></head><body>
      <h1>RAG Evaluation Framework</h1>
      <p class="sub">4 architectures &middot; custom RAGAS-style metrics &middot; semantic cache.</p>
      <a href="/docs">Interactive API docs (Swagger)</a>
      <a href="/redoc">ReDoc documentation</a>
      <a href="/health">Health check</a>
      <a href="/architectures">List available architectures</a>
    </body></html>
    """


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_available=bool(MISTRAL_API_KEY or os.getenv("GROQ_API_KEY")),
        vector_store_exists=Path(VECTOR_STORE_DIR).exists(),
        auth_enabled=bool(API_KEY),
    )


@app.get("/architectures")
def list_architectures() -> Dict[str, List[str]]:
    return {"architectures": sorted(_VALID_ARCHS)}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
def query(request: Request, req: QueryRequest) -> QueryResponse:
    archs = _architectures()
    if req.use_cache:
        cached = _cache().lookup(req.question)
        if cached:
            return QueryResponse(
                architecture=req.architecture,
                question=req.question,
                answer=cached["answer"],
                contexts=cached.get("contexts", []),
                latency_ms=0.0,
                cache_hit=True,
                similarity_score=cached.get("similarity_score"),
            )

    res = archs[req.architecture].run(req.question)
    if req.use_cache:
        _cache().store(req.question, res["answer"], res.get("contexts", []))
    return QueryResponse(
        architecture=req.architecture,
        question=req.question,
        answer=res["answer"],
        contexts=res.get("contexts", []),
        latency_ms=res.get("latency_ms", 0.0),
        cache_hit=False,
    )


@app.post("/ab-test", response_model=ABResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
def ab_test(request: Request, req: ABRequest) -> ABResponse:
    archs = _architectures()
    res_a = archs[req.arch_a].run(req.question)
    res_b = archs[req.arch_b].run(req.question)
    return ABResponse(
        question=req.question,
        variant_a=ABVariantResult(
            architecture=req.arch_a,
            answer=res_a["answer"],
            latency_ms=res_a["latency_ms"],
            num_chunks=len(res_a.get("contexts", [])),
        ),
        variant_b=ABVariantResult(
            architecture=req.arch_b,
            answer=res_b["answer"],
            latency_ms=res_b["latency_ms"],
            num_chunks=len(res_b.get("contexts", [])),
        ),
        delta_latency_ms=res_b["latency_ms"] - res_a["latency_ms"],
    )


@app.post("/evaluate", response_model=EvalResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")  # eval is expensive — tighter limit
def evaluate(request: Request, req: EvalRequest) -> EvalResponse:
    import json as _json

    from module3.evaluation import EvaluationPipeline, RAGASEvaluator

    syn_path = ROOT / "data" / "synthetic_dataset.json"
    if not syn_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "synthetic_dataset.json not found. Run module1 preprocessing "
                "before calling /evaluate."
            ),
        )
    synthetic = _json.loads(syn_path.read_text())[: req.samples]

    ragas = None
    if req.use_ragas:
        try:
            ragas = RAGASEvaluator()
        except Exception as exc:  # noqa: BLE001 — surface as HTTP 503
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"RAGAS evaluator unavailable: {exc}",
            ) from exc

    pipeline = EvaluationPipeline(ragas_evaluator=ragas)
    archs = _architectures()
    results = {}
    for name in req.architectures:
        df = pipeline.evaluate_architecture(
            architecture_name=name,
            rag_system=archs[name],
            synthetic_dataset=synthetic,
            max_samples=req.samples,
        )
        results[name] = df

    summary = pipeline.compare_architectures(results)
    return EvalResponse(
        samples=req.samples,
        architectures=list(results.keys()),
        summary=summary.to_dict(orient="records"),
    )


@app.get("/cache/stats", response_model=CacheStatsResponse, dependencies=[Depends(require_api_key)])
def cache_stats() -> CacheStatsResponse:
    s = _cache().stats()
    return CacheStatsResponse(**{k: s.get(k, 0) for k in CacheStatsResponse.model_fields})


@app.post("/cache/lookup", dependencies=[Depends(require_api_key)])
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
def cache_lookup(request: Request, req: CacheLookup) -> Dict[str, Any]:
    res = _cache().lookup(req.query)
    if res is None:
        return {"hit": False}
    return {"hit": True, **res}


@app.delete("/cache", dependencies=[Depends(require_api_key)])
def cache_clear() -> Dict[str, str]:
    _cache().clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

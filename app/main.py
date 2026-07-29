import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pythonjsonlogger import jsonlogger
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import settings
from app.db import SessionLocal, engine
from app.embeddings import EmbeddingProvider
from app.models import Base
from app.observability import slo_metrics
from app.ocr_preprocessor import OcrPreprocessError
from app.prometheus import RETRIEVAL_RESULTS, record_http_request, render_metrics
from app.parser import RAGAnythingParser, log_dependency_compatibility
from app.repository import EmbeddingModelMismatchError, RagRepository
from app.reranker import CrossEncoderReranker
from app.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourcesRequest,
    SourcesResponse,
)
from app.security import require_rate_limit
from app.mineru_runner import MineruUnavailableError
from app.service import IngestLimitError, RagService


logger = logging.getLogger("rag_service")
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(handler)
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


@contextmanager
def db_session() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db() -> Session:
    with db_session() as db:
        yield db


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def require_admin_api_key(x_admin_api_key: str = Header(default="")) -> None:
    if not secrets.compare_digest(x_admin_api_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin API key")


def _validate_secure_settings() -> None:
    if settings.app_env.lower() not in {"prod", "production"}:
        return
    if (
        settings.api_key == "change-me"
        or settings.api_key.lower().startswith("replace-with")
        or len(settings.api_key) < 32
    ):
        raise RuntimeError("API_KEY must be a non-default secret of at least 32 characters in production")
    if (
        settings.admin_api_key == "change-me-admin"
        or settings.admin_api_key.lower().startswith("replace-with")
        or len(settings.admin_api_key) < 32
    ):
        raise RuntimeError("ADMIN_API_KEY must be a non-default secret of at least 32 characters in production")
    if secrets.compare_digest(settings.api_key, settings.admin_api_key):
        raise RuntimeError("API_KEY and ADMIN_API_KEY must be different")
    if "replace-with" in settings.database_url.lower():
        raise RuntimeError("DATABASE_URL contains a placeholder password")
    if settings.uvicorn_workers > 1 and settings.require_redis_in_production and not settings.redis_url:
        raise RuntimeError("REDIS_URL is required in production when UVICORN_WORKERS > 1")
    if not settings.embed_offline:
        raise RuntimeError("EMBED_OFFLINE=true is required in production")
    if not settings.reranker_offline:
        raise RuntimeError("RERANKER_OFFLINE=true is required in production")
    if not settings.fail_on_embedding_fallback:
        raise RuntimeError("FAIL_ON_EMBEDDING_FALLBACK=true is required in production")
    if not settings.enforce_embedding_model_compatibility:
        raise RuntimeError("ENFORCE_EMBEDDING_MODEL_COMPATIBILITY=true is required in production")
    if settings.auto_create_schema:
        raise RuntimeError("AUTO_CREATE_SCHEMA must be false in production; use versioned migrations")
    if not settings.allowed_hosts or "*" in settings.allowed_hosts:
        raise RuntimeError("ALLOWED_HOSTS must contain explicit hostnames in production")


@contextmanager
def exclusive_ingest_lock(service: RagService):
    db = getattr(getattr(service, "repository", None), "db", None)
    if db is None:
        yield
        return

    bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        yield
        return

    lock_id = 7_240_716_001
    # Keep the session-level lock on a dedicated autocommit connection. Reusing
    # the repository Session here leaves a transaction idle during long MinerU
    # runs, so PostgreSQL eventually terminates it.
    lock_connection = bind.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        acquired = bool(
            lock_connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_id)"),
                {"lock_id": lock_id},
            ).scalar()
        )
        if not acquired:
            raise HTTPException(status_code=409, detail="Another ingest job is already running")
        yield
    finally:
        try:
            lock_connection.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"),
                {"lock_id": lock_id},
            )
        except Exception:
            logger.exception("ingest_advisory_unlock_failed")
        finally:
            lock_connection.close()


def _validate_ingest_path(input_path: str) -> None:
    path = Path(input_path).resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="input_path must be an existing directory")
    if not settings.ingest_path_must_be_under_storage_raw:
        return

    allowed_root = Path(settings.storage_raw).resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"input_path must be under {allowed_root}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_secure_settings()
    log_dependency_compatibility()
    if settings.auto_create_schema:
        Base.metadata.create_all(bind=engine)
    app.state.parser = RAGAnythingParser()
    app.state.embeddings = EmbeddingProvider()
    app.state.reranker = CrossEncoderReranker()
    try:
        yield
    finally:
        engine.dispose()


production_mode = settings.app_env.lower() in {"prod", "production"}
app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url=None if production_mode else "/docs",
    redoc_url=None if production_mode else "/redoc",
    openapi_url=None if production_mode else "/openapi.json",
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    latency_seconds = time.perf_counter() - start
    latency_ms = latency_seconds * 1000
    slo_metrics.record(latency_ms, response.status_code)
    route = request.scope.get("route")
    path = getattr(route, "path", "__unmatched__")
    record_http_request(request.method, path, response.status_code, latency_seconds)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


def get_service(request: Request, db: Session = Depends(get_db)) -> RagService:
    return RagService(
        parser=request.app.state.parser,
        repository=RagRepository(
            db=db,
            embeddings=request.app.state.embeddings,
            reranker=request.app.state.reranker,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exc_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})



@app.exception_handler(MineruUnavailableError)
async def mineru_unavailable_handler(_: Request, exc: MineruUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.exception_handler(OcrPreprocessError)
async def ocr_preprocess_error_handler(_: Request, exc: OcrPreprocessError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(IngestLimitError)
async def ingest_limit_handler(_: Request, exc: IngestLimitError) -> JSONResponse:
    return JSONResponse(status_code=413, content={"error": str(exc)})


@app.exception_handler(EmbeddingModelMismatchError)
async def embedding_mismatch_handler(_: Request, exc: EmbeddingModelMismatchError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(SQLAlchemyError)
async def db_exc_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("database_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"error": "Database error"})


@app.exception_handler(Exception)
async def generic_exc_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/livez")
def livez() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz", dependencies=[Depends(require_api_key)])
def healthz(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/readyz", dependencies=[Depends(require_api_key)])
def readyz(request: Request, db: Session = Depends(get_db)) -> dict:
    db_ok = True
    pgvector_ok = True
    try:
        db.execute(text("SELECT 1"))
        ext = db.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector' LIMIT 1")).scalar_one_or_none()
        pgvector_ok = ext is not None
    except Exception:
        db_ok = False
        pgvector_ok = False

    parser_loaded = getattr(request.app.state, "parser", None) is not None
    embeddings_loaded = getattr(request.app.state, "embeddings", None) is not None and not request.app.state.embeddings.using_fallback
    reranker_obj = getattr(request.app.state, "reranker", None)
    reranker_loaded = reranker_obj is not None and reranker_obj.available
    embedding_compatibility = {"compatible": False, "incompatible_sources": []}
    if db_ok and getattr(request.app.state, "embeddings", None) is not None:
        try:
            embedding_compatibility = RagRepository(
                db=db,
                embeddings=request.app.state.embeddings,
                reranker=reranker_obj,
            ).embedding_compatibility()
        except Exception as exc:
            embedding_compatibility = {
                "compatible": False,
                "incompatible_sources": [],
                "reason": str(exc),
            }

    ready = all(
        [
            db_ok,
            pgvector_ok,
            parser_loaded,
            embeddings_loaded,
            reranker_loaded,
            embedding_compatibility.get("compatible", False),
        ]
    )
    payload = {
        "status": "ok" if ready else "degraded",
        "checks": {
            "db": db_ok,
            "pgvector": pgvector_ok,
            "parser_loaded": parser_loaded,
            "embeddings_loaded": embeddings_loaded,
            "embedding_fingerprint": getattr(
                getattr(request.app.state, "embeddings", None),
                "model_fingerprint",
                None,
            ),
            "embedding_index_compatible": embedding_compatibility.get("compatible", False),
            "embedding_index_incompatible_sources": embedding_compatibility.get("incompatible_sources", []),
            "reranker_loaded": reranker_loaded,
            "reranker_model": getattr(reranker_obj, "model_name", None),
            "reranker_error": getattr(reranker_obj, "load_error", None),
        },
    }
    if not ready:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics() -> dict:
    return {"slo": slo_metrics.snapshot()}


@app.get("/metrics/prometheus", dependencies=[Depends(require_api_key)])
def prometheus_metrics() -> Response:
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin_api_key), Depends(require_rate_limit)])
def ingest(payload: IngestRequest, service: RagService = Depends(get_service)) -> IngestResponse:
    _validate_ingest_path(payload.input_path)
    with exclusive_ingest_lock(service):
        stats = service.ingest(
            payload.input_path,
            payload.collection,
            payload.reindex,
            reparse=payload.reparse,
        )
    return IngestResponse(**stats)


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key), Depends(require_rate_limit)])
def retrieve(payload: RetrieveRequest, service: RagService = Depends(get_service)) -> RetrieveResponse:
    if payload.debug and settings.app_env.lower() in {"prod", "production"} and not settings.allow_retrieval_debug:
        raise HTTPException(status_code=403, detail="Retrieval debug is disabled")
    if payload.debug and hasattr(service, "retrieve_with_debug"):
        hits, debug = service.retrieve_with_debug(
            payload.query,
            payload.top_k,
            payload.min_score,
            payload.collection,
            payload.source_uris,
            payload.return_text,
            payload.include_toc,
            payload.include_low_quality,
            payload.include_navigation,
            payload.return_context,
        )
        RETRIEVAL_RESULTS.observe(len(hits))
        return RetrieveResponse(hits=hits, debug=debug)

    hits = service.retrieve(
        payload.query,
        payload.top_k,
        payload.min_score,
        payload.collection,
        payload.source_uris,
        payload.return_text,
        payload.include_toc,
        payload.include_low_quality,
        payload.include_navigation,
        payload.return_context,
    )
    RETRIEVAL_RESULTS.observe(len(hits))
    return RetrieveResponse(hits=hits)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key), Depends(require_rate_limit)])
def query(payload: QueryRequest, service: RagService = Depends(get_service)) -> QueryResponse:
    return service.query(
        payload.query,
        payload.top_k,
        payload.min_score,
        payload.collection,
        payload.source_uris,
        return_sources=payload.return_sources,
    )


@app.post("/sources", response_model=SourcesResponse, dependencies=[Depends(require_api_key), Depends(require_rate_limit)])
def sources(payload: SourcesRequest, service: RagService = Depends(get_service)) -> SourcesResponse:
    return SourcesResponse(sources=service.list_sources(payload.collection))

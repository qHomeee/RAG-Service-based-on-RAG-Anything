import logging
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, engine
from app.embeddings import EmbeddingProvider
from app.models import Base
from app.observability import slo_metrics
from app.parser import RAGAnythingParser, log_dependency_compatibility
from app.repository import RagRepository
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
from app.service import RagService


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
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def require_admin_api_key(x_admin_api_key: str = Header(default="")) -> None:
    if x_admin_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")


def _validate_secure_settings() -> None:
    if settings.app_env.lower() in {"prod", "production"} and settings.api_key == "change-me":
        raise RuntimeError("API_KEY must be changed in production")
    if settings.app_env.lower() in {"prod", "production"} and settings.admin_api_key == "change-me-admin":
        raise RuntimeError("ADMIN_API_KEY must be changed in production")


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
    Base.metadata.create_all(bind=engine)
    app.state.parser = RAGAnythingParser()
    app.state.embeddings = EmbeddingProvider()
    app.state.reranker = CrossEncoderReranker()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    slo_metrics.record(latency_ms, response.status_code)
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


@app.exception_handler(SQLAlchemyError)
async def db_exc_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("database_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"error": "Database error"})


@app.exception_handler(Exception)
async def generic_exc_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


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

    ready = all([db_ok, pgvector_ok, parser_loaded, embeddings_loaded, reranker_loaded])
    return {
        "status": "ok" if ready else "degraded",
        "checks": {
            "db": db_ok,
            "pgvector": pgvector_ok,
            "parser_loaded": parser_loaded,
            "embeddings_loaded": embeddings_loaded,
            "reranker_loaded": reranker_loaded,
            "reranker_model": getattr(reranker_obj, "model_name", None),
            "reranker_error": getattr(reranker_obj, "load_error", None),
        },
    }


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics() -> dict:
    return {"slo": slo_metrics.snapshot()}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin_api_key), Depends(require_rate_limit)])
def ingest(payload: IngestRequest, service: RagService = Depends(get_service)) -> IngestResponse:
    _validate_ingest_path(payload.input_path)
    stats = service.ingest(payload.input_path, payload.collection, payload.reindex)
    return IngestResponse(**stats)


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key), Depends(require_rate_limit)])
def retrieve(payload: RetrieveRequest, service: RagService = Depends(get_service)) -> RetrieveResponse:
    if len(payload.query) > settings.max_query_chars:
        raise HTTPException(status_code=413, detail="Query too large")
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
        )
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
    )
    return RetrieveResponse(hits=hits)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key), Depends(require_rate_limit)])
def query(payload: QueryRequest, service: RagService = Depends(get_service)) -> QueryResponse:
    if len(payload.query) > settings.max_query_chars:
        raise HTTPException(status_code=413, detail="Query too large")
    return service.query(payload.query, payload.top_k, payload.min_score, payload.collection, payload.source_uris)


@app.post("/sources", response_model=SourcesResponse, dependencies=[Depends(require_api_key), Depends(require_rate_limit)])
def sources(payload: SourcesRequest, service: RagService = Depends(get_service)) -> SourcesResponse:
    return SourcesResponse(sources=service.list_sources(payload.collection))

import logging
from contextlib import contextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, engine
from app.embeddings import EmbeddingProvider
from app.models import Base
from app.parser import RAGAnythingParser
from app.repository import RagRepository
from app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse
from app.service import RagService


logger = logging.getLogger("rag_service")
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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


def get_service(db: Session = Depends(get_db)) -> RagService:
    return RagService(parser=RAGAnythingParser(), repository=RagRepository(db=db, embeddings=EmbeddingProvider()))


app = FastAPI(title=settings.app_name)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.exception_handler(HTTPException)
async def http_exc_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(SQLAlchemyError)
async def db_exc_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("database_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"error": "Database error"})


@app.exception_handler(Exception)
async def generic_exc_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(payload: IngestRequest, service: RagService = Depends(get_service)) -> IngestResponse:
    stats = service.ingest(payload.input_path, payload.collection, payload.reindex)
    return IngestResponse(**stats)


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
def retrieve(payload: RetrieveRequest, service: RagService = Depends(get_service)) -> RetrieveResponse:
    if len(payload.query) > settings.max_query_chars:
        raise HTTPException(status_code=413, detail="Query too large")
    hits = service.retrieve(payload.query, payload.top_k, payload.min_score, payload.collection, payload.return_text)
    return RetrieveResponse(hits=hits)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(payload: QueryRequest, service: RagService = Depends(get_service)) -> QueryResponse:
    if len(payload.query) > settings.max_query_chars:
        raise HTTPException(status_code=413, detail="Query too large")
    return service.query(payload.query, payload.top_k, payload.min_score, payload.collection)

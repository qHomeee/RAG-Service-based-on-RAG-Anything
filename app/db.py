from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings


connect_args = {}
if settings.database_url.startswith("postgresql"):
    connect_args["options"] = " ".join(
        [
            f"-c statement_timeout={max(1, settings.db_statement_timeout_ms)}",
            f"-c lock_timeout={max(1, settings.db_lock_timeout_ms)}",
            (
                "-c idle_in_transaction_session_timeout="
                f"{max(1, settings.db_idle_transaction_timeout_ms)}"
            ),
        ]
    )

engine = create_engine(
    settings.database_url,
    future=True,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_recycle=settings.db_pool_recycle_seconds,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

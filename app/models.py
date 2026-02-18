import uuid

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover
    Vector = None


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_uri: Mapped[str] = mapped_column(String, unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    collection: Mapped[str] = mapped_column(String, default="default", index=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    fragments = relationship("Fragment", back_populates="document", cascade="all, delete-orphan")


class Fragment(Base):
    __tablename__ = "fragments"

    fragment_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True)
    source_uri: Mapped[str] = mapped_column(String, index=True)
    type: Mapped[str] = mapped_column(String)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    element_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    snippet: Mapped[str] = mapped_column(String(450))
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    document = relationship("Document", back_populates="fragments")
    embeddings = relationship("Embedding", back_populates="fragment", cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint("fragment_id", "subchunk_index", name="uq_fragment_subchunk"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fragment_id: Mapped[str] = mapped_column(ForeignKey("fragments.fragment_id", ondelete="CASCADE"), index=True)
    subchunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384) if Vector else JSON)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    fragment = relationship("Fragment", back_populates="embeddings")

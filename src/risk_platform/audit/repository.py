from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PredictionAudit(Base):
    __tablename__ = "prediction_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    model_backend: Mapped[str] = mapped_column(String(32))
    input_hash: Mapped[str] = mapped_column(String(64))
    risk_probability: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(32))
    latency_ms: Mapped[float] = mapped_column(Float)


def input_hash(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


class AuditRepository:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        try:
            self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
            Base.metadata.create_all(self.engine)
        except Exception:
            logger.exception("Primary database unavailable; falling back to local SQLite audit store")
            self.engine = create_engine("sqlite:///./risk_audit.db", connect_args={"check_same_thread": False})
            Base.metadata.create_all(self.engine)

    def _row(
        self,
        *,
        request_id: str,
        model_version: str,
        model_backend: str,
        record: dict,
        risk_probability: float,
        decision: str,
        latency_ms: float,
    ) -> PredictionAudit:
        # Raw customer/transaction payloads are deliberately not persisted here.
        return PredictionAudit(
            request_id=request_id,
            created_at=datetime.now(timezone.utc),
            model_version=model_version,
            model_backend=model_backend,
            input_hash=input_hash(record),
            risk_probability=risk_probability,
            decision=decision,
            latency_ms=latency_ms,
        )

    def write(self, **kwargs) -> None:
        with Session(self.engine) as session:
            session.add(self._row(**kwargs))
            session.commit()

    def write_many(self, rows: list[dict], chunk_size: int = 1000) -> None:
        """Persist audit events in bounded transactions for batch throughput."""
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            with Session(self.engine) as session:
                session.add_all([self._row(**item) for item in chunk])
                session.commit()

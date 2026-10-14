"""Neon (Postgres) through the SQLAlchemy ORM: the tables are Python classes.

Create the tables:   python -m fraud.api.db
Card numbers are NEVER stored: card_key = HMAC-SHA256(cc_num) from fraud.api.card_key.
"""
import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import CheckConstraint, DateTime, Double, Index, SmallInteger, String, Text, create_engine, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from fraud.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


class Base(DeclarativeBase):
    """Every table class inherits from this; Base.metadata knows all of them."""


class CardTransaction(Base):
    """Every card's transactions: the API reads a card's history from here."""
    __tablename__ = "card_transactions"
    __table_args__ = (
        CheckConstraint("amt > 0", name="amt_positive"),
        # history lookup = "this card's rows before this time": the index makes it a jump, not a scan
        Index("idx_card_transactions_card_time", "card_key", "txn_time"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    card_key: Mapped[str] = mapped_column(String(64))       # HMAC of the card number
    txn_time: Mapped[datetime] = mapped_column(DateTime)
    amt: Mapped[float] = mapped_column(Double)              # Double = float64: exact round-trip -> parity


class PredictionLog(Base):
    """Every prediction the API makes: the audit trail, and the input for drift monitoring later."""
    __tablename__ = "prediction_log"
    __table_args__ = (CheckConstraint("decision IN ('flag', 'approve')", name="decision_valid"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_version: Mapped[str] = mapped_column(Text)
    card_key: Mapped[str] = mapped_column(String(64))
    txn_time: Mapped[datetime] = mapped_column(DateTime)
    amt: Mapped[float] = mapped_column(Double)
    category: Mapped[str] = mapped_column(Text)
    features: Mapped[dict] = mapped_column(JSONB)            # the exact 9 model inputs, as JSON
    score: Mapped[float] = mapped_column(Double)
    decision: Mapped[str] = mapped_column(Text)
    true_label: Mapped[int | None] = mapped_column(SmallInteger)   # "| None" = may be empty (filled later)


def get_engine():
    """One engine per process: it holds a pool of open connections to Neon."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    url = url.replace("postgresql://", "postgresql+psycopg://", 1)   # tell SQLAlchemy to use psycopg 3
    return create_engine(url, pool_pre_ping=True)   # pre_ping: Neon sleeps idle connections -> test before use


def create_tables(engine) -> None:
    """Create every table defined above that doesn't exist yet. Safe to repeat."""
    Base.metadata.create_all(engine)
    logger.info("Tables ready: %s", ", ".join(Base.metadata.tables))


if __name__ == "__main__":
    create_tables(get_engine())

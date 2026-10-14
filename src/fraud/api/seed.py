"""Seed Neon with the TRAIN-period history (the test period is replayed through the API later).

Run once:   python -m fraud.api.seed
"""
import pandas as pd
from sqlalchemy import func, insert, select, text
from sqlalchemy.orm import Session

from fraud.api.card_key import card_key
from fraud.api.db import CardTransaction, create_tables, get_engine
from fraud.logger import get_logger
from fraud.utils.common import read_yaml

logger = get_logger(__name__)
CHUNK_SIZE = 50_000


def seed() -> None:
    params = read_yaml("params.yaml")
    df = pd.read_parquet(params["data"]["ingested_train"], columns=["cc_num", "trans_date_trans_time", "amt"])

    # HMAC once per card (~983), not once per row (1.3M); raw card numbers never leave this function
    keys = {cc: card_key(int(cc)) for cc in df["cc_num"].unique()}
    df["card_key"] = df["cc_num"].map(keys)
    df = df.rename(columns={"trans_date_trans_time": "txn_time"})[["card_key", "txn_time", "amt"]]

    engine = get_engine()
    create_tables(engine)
    with Session(engine) as session:
        existing = session.scalar(select(func.count()).select_from(CardTransaction))
        if existing > 0:
            logger.warning("card_transactions already has %d rows: not seeding again", existing)
            return

        total = len(df)
        for start in range(0, total, CHUNK_SIZE):
            rows = df.iloc[start:start + CHUNK_SIZE].to_dict("records")
            session.execute(insert(CardTransaction), rows)
            session.commit()                                  # each chunk saved for good
            logger.info("%d / %d rows inserted", min(start + CHUNK_SIZE, total), total)

        count = session.scalar(select(func.count()).select_from(CardTransaction))
        size = session.scalar(text("SELECT pg_size_pretty(pg_total_relation_size('card_transactions'))"))
        logger.info("Seeded %d rows (expected %d). Storage used by card_transactions: %s", count, total, size)


if __name__ == "__main__":
    seed()

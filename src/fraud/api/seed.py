"""Seed Neon with the card state at the END of the train period (the test period is replayed through the API).

Run once:   python -m fraud.api.seed
"""
import pandas as pd
from sqlalchemy import func, insert, select, text
from sqlalchemy.orm import Session

from fraud.api.card_key import card_key
from fraud.api.db import CardStats, RecentTransaction, create_tables, get_engine
from fraud.logger import get_logger
from fraud.utils.common import read_yaml

logger = get_logger(__name__)


def seed() -> None:
    params = read_yaml("params.yaml")
    window = pd.Timedelta(params["features"]["txn_window"])
    df = pd.read_parquet(params["data"]["ingested_train"], columns=["cc_num", "trans_date_trans_time", "amt"])

    # HMAC once per card; raw card numbers never leave this function
    keys = {cc: card_key(int(cc)) for cc in df["cc_num"].unique()}
    df["card_key"] = df["cc_num"].map(keys)
    df = df.rename(columns={"trans_date_trans_time": "txn_time"})

    # 1 row per card: the running totals (named aggregation, notebook 01)
    stats = df.groupby("card_key").agg(
        txn_count=("amt", "size"),
        amt_total=("amt", "sum"),
        last_txn_time=("txn_time", "max"),
    ).reset_index()

    # only the last 24 h of the train period: later transactions can't look further back than that
    recent = df.loc[df["txn_time"] > df["txn_time"].max() - window, ["card_key", "txn_time", "amt"]]

    engine = get_engine()
    create_tables(engine)
    with Session(engine) as session, session.begin():          # all or nothing
        if session.scalar(select(func.count()).select_from(CardStats)) > 0:
            logger.warning("card_stats already has rows: not seeding again")
            return
        session.execute(insert(CardStats), stats.to_dict("records"))
        session.execute(insert(RecentTransaction), recent.to_dict("records"))

    with Session(engine) as session:
        size = session.scalar(text("SELECT pg_size_pretty(pg_database_size(current_database()))"))
    logger.info("Seeded %d cards (from %d train rows) + %d recent transactions. Database size: %s",
                len(stats), len(df), len(recent), size)


if __name__ == "__main__":
    seed()

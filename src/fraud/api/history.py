"""Card history for /predict: the API only asks get_card_state(); the storage behind it can change.

A card's "state" = running totals (count, amount total, last purchase time) + its last 24 h of purchase times.
That is everything the 5 card features need.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, and_, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from fraud.api.card_key import card_key
from fraud.api.db import CardStats, RecentTransaction
from fraud.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CardState:
    """Everything known about a card's purchases BEFORE the transaction being scored."""
    txn_count: int = 0
    amt_total: float = 0.0
    last_txn_time: datetime | None = None
    recent_txn_times: list = field(default_factory=list)   # purchase times inside the window


class HistoryStore:
    """The 'plug shape': every store answers the same question the same way."""

    def get_card_state(self, cc_num: int, before: datetime) -> CardState:
        raise NotImplementedError


class FileHistoryStore(HistoryStore):
    """From the ingested Parquet files (read-only): offline development and the parity test.

    Known edge case: the files also contain the transaction being scored, so "before" must be strict (<).
    Two purchases on one card in the SAME second (0.005% of rows) therefore can't see each other here;
    the live Neon store handles them correctly.
    """

    def __init__(self, paths: list[Path], window: str):
        cols = ["cc_num", "trans_date_trans_time", "amt"]
        df = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
        df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
        self._by_card = dict(tuple(df.groupby("cc_num")))
        self.window = pd.Timedelta(window)
        logger.info("History loaded: %s cards, %s transactions", len(self._by_card), len(df))

    def get_card_state(self, cc_num: int, before: datetime) -> CardState:
        card = self._by_card.get(cc_num)
        if card is None:
            return CardState()
        past = card[card["trans_date_trans_time"] < before]
        if past.empty:
            return CardState()
        times = past["trans_date_trans_time"]
        return CardState(
            txn_count=len(past),
            amt_total=float(past["amt"].sum()),
            last_txn_time=times.iloc[-1],
            recent_txn_times=times[times > pd.Timestamp(before) - self.window].tolist(),
        )


class NeonHistoryStore(HistoryStore):
    """From Neon: one row per card + the last 24 h. Two tiny indexed queries per request."""

    def __init__(self, engine: Engine, window: str):
        self.engine = engine
        self.window = pd.Timedelta(window)

    def get_card_state(self, cc_num: int, before: datetime) -> CardState:
        # ONE query (card totals LEFT JOIN its recent purchases) on an autocommit connection:
        # no separate BEGIN/COMMIT, so only one trip to the server (each trip ~250 ms from India to us-east-2)
        query = (
            select(CardStats.txn_count, CardStats.amt_total, CardStats.last_txn_time, RecentTransaction.txn_time)
            .outerjoin(RecentTransaction, and_(
                RecentTransaction.card_key == CardStats.card_key,
                RecentTransaction.txn_time > before - self.window,
                # <= : the live store never holds THIS txn yet, so same-second earlier ones count
                RecentTransaction.txn_time <= before,
            ))
            .where(CardStats.card_key == card_key(cc_num))
        )
        with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            rows = conn.execute(query).all()
        if not rows:                                    # card never seen
            return CardState()
        count, total, last_time, _ = rows[0]
        if last_time > before:
            # running totals assume transactions arrive in time order (true live, and in the replay)
            logger.warning("Out-of-order transaction for a card: totals include later purchases")
        return CardState(count, total, last_time, [r.txn_time for r in rows if r.txn_time is not None])


def add_to_card_state(session: Session, key: str, txn_time: datetime, amt: float, window: str) -> None:
    """After scoring: update the card's totals, remember this purchase, forget purchases older than the window."""
    upsert = pg_insert(CardStats).values(card_key=key, txn_count=1, amt_total=amt, last_txn_time=txn_time)
    session.execute(upsert.on_conflict_do_update(          # new card -> insert; known card -> add to its totals
        index_elements=[CardStats.card_key],
        set_={
            "txn_count": CardStats.txn_count + 1,
            "amt_total": CardStats.amt_total + amt,
            "last_txn_time": txn_time,
        },
    ))
    session.add(RecentTransaction(card_key=key, txn_time=txn_time, amt=amt))
    session.execute(delete(RecentTransaction).where(
        RecentTransaction.card_key == key,
        RecentTransaction.txn_time <= txn_time - pd.Timedelta(window),
    ))

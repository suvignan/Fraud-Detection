"""Card history for /predict: the API only knows get_history(); the storage behind it can change."""
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from fraud.api.card_key import card_key
from fraud.api.db import CardTransaction
from fraud.logger import get_logger

logger = get_logger(__name__)

HISTORY_COLUMNS = ["cc_num", "trans_date_trans_time", "amt"]


class HistoryStore:
    """The 'plug shape': every store answers the same question the same way."""

    def get_history(self, cc_num: int, before: datetime) -> pd.DataFrame:
        """This card's transactions strictly BEFORE `before`, oldest first, as HISTORY_COLUMNS."""
        raise NotImplementedError


class FileHistoryStore(HistoryStore):
    """History from the ingested Parquet files, grouped by card once at startup (read-only)."""

    def __init__(self, paths: list[Path]):
        frames = [pd.read_parquet(path, columns=HISTORY_COLUMNS) for path in paths]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
        self._history_by_card = dict(tuple(df.groupby("cc_num")))
        logger.info("History loaded: %s cards, %s transactions", len(self._history_by_card), len(df))

    def get_history(self, cc_num: int, before: datetime) -> pd.DataFrame:
        card_history = self._history_by_card.get(cc_num)
        if card_history is None:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        return card_history[card_history["trans_date_trans_time"] < before]


class NeonHistoryStore(HistoryStore):
    """History from Neon: one indexed query per request, looked up by the HMAC card_key."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_history(self, cc_num: int, before: datetime) -> pd.DataFrame:
        query = (
            select(CardTransaction.txn_time, CardTransaction.amt)
            .where(CardTransaction.card_key == card_key(cc_num), CardTransaction.txn_time < before)
            .order_by(CardTransaction.txn_time)
        )
        with Session(self.engine) as session:
            rows = session.execute(query).all()
        history = pd.DataFrame(rows, columns=["trans_date_trans_time", "amt"])
        history["trans_date_trans_time"] = pd.to_datetime(history["trans_date_trans_time"])
        history["amt"] = history["amt"].astype("float64")
        history.insert(0, "cc_num", cc_num)   # same 3 columns, same order as the file store
        return history

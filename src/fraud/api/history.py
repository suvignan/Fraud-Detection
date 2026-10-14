"""Card history for /predict: the API only knows get_history(); the storage behind it can change."""
from datetime import datetime
from pathlib import Path

import pandas as pd

from fraud.logger import get_logger


logger = get_logger(__name__)

HISTORY_COLUMNS = [
    "cc_num",
    "trans_date_trans_time",
    "amt",
]

class HistoryStore:
    """Base interface for transaction history storage."""

    def get_history(self,cc_num: int,before: datetime,) -> pd.DataFrame:
        raise NotImplementedError

class FileHistoryStore(HistoryStore):
    """File-based implementation of the HistoryStore."""
    def __init__(self, paths: list[Path]):
        frames = [pd.read_parquet(path, columns=HISTORY_COLUMNS) for path in paths]
        df = pd.concat(frames, ignore_index=True)

        df = df.sort_values(
            ['cc_num','trans_date_trans_time']
        ).reset_index(drop=True)

        self._history_by_card = dict(tuple(df.groupby('cc_num')))
        logger.info(
            "History loaded: %s cards, %s transactions",
            len(self._history_by_card),
            len(df),
        )

    def get_history(self,cc_num:int,before:datetime)->pd.DataFrame:
        card_history = self._history_by_card.get(cc_num)

        if card_history is None:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        
        history = card_history[
            card_history["trans_date_trans_time"] < before
        ]

        return history

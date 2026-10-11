"""One shared logger: writes timestamped messages to the console and logs/.

Usage in any module:
    from fraud.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded %d rows", len(df))

Banking rule: never log personal data (no raw cc_num, no names). Logs get copied everywhere.
"""
import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"   # project root/logs, wherever you run from
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"   # one file per run

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

"""Small shared helpers: read_yaml, save_json, etc."""
from pathlib import Path
import json
import yaml

from fraud.exception import FraudException
from fraud.logger import get_logger

logger = get_logger(__name__)


def read_yaml(file_path: str | Path) -> dict:
    """Read a YAML file and return its contents as a dictionary."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if content is None:
            raise ValueError(f"YAML file is empty: {file_path}")
        logger.info("Read YAML file: %s", file_path)
        return content
    except Exception as e:
        logger.error("Error reading YAML file %s: %s", file_path, e)
        raise FraudException(e) from e


def save_json(file_path: str | Path, data: dict) -> None:
    """Save a dictionary as a readable JSON file, creating the folder if needed."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved JSON file: %s", file_path)
    except Exception as e:
        logger.exception("Error saving JSON file %s", file_path)
        raise FraudException(e) from e



def load_json(file_path: str | Path) -> dict:
    """Read a JSON file and return its contents as a dictionary."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        logger.info("Loaded JSON file: %s", file_path)
        return content
    except Exception as e:
        logger.exception("Error loading JSON file %s", file_path)
        raise FraudException(e) from e

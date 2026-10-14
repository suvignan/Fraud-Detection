"""card_key: an HMAC-SHA256 of the card number, so raw card numbers never reach the database."""
import hashlib
import hmac
import os

from dotenv import load_dotenv

load_dotenv()  # once, at import: copies .env into the environment (an already-set variable wins)


def card_key(cc_num: int) -> str:
    """Same card -> same key; can't be reversed or brute-forced without CARD_KEY_SECRET."""
    secret = os.environ.get("CARD_KEY_SECRET")
    if not secret:
        raise RuntimeError("CARD_KEY_SECRET is not set in .env")
    return hmac.new(secret.encode(), str(cc_num).encode(), hashlib.sha256).hexdigest()

"""card_key: stable, distinct, irreversible, and refuses to run without a secret."""
import pytest

from fraud.api.card_key import card_key

CARD = 4005676619255478


@pytest.fixture(autouse=True)
def test_secret(monkeypatch):
    """Every test uses a fake secret: tests never depend on the real .env (and work in CI)."""
    monkeypatch.setenv("CARD_KEY_SECRET", "test-secret")


def test_same_card_gives_the_same_key():
    assert card_key(CARD) == card_key(CARD)


def test_different_cards_give_different_keys():
    assert card_key(CARD) != card_key(CARD + 1)


def test_key_is_64_hex_chars_and_hides_the_card_number():
    key = card_key(CARD)
    assert len(key) == 64
    assert str(CARD) not in key


def test_a_different_secret_gives_a_different_key(monkeypatch):
    first = card_key(CARD)
    monkeypatch.setenv("CARD_KEY_SECRET", "another-secret")
    assert card_key(CARD) != first   # why the secret must never change: every key would move


def test_missing_secret_fails_loudly(monkeypatch):
    monkeypatch.delenv("CARD_KEY_SECRET")
    with pytest.raises(RuntimeError, match="CARD_KEY_SECRET"):
        card_key(CARD)

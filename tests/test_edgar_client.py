"""
Mocked so it runs with zero network access -- no reliance on SEC being up,
no risk of hitting the rate limit while testing.
"""
from unittest.mock import patch

from src.ingestion.edgar_client import EdgarClient

FAKE_TICKER_MAP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}


def test_get_cik_for_ticker_pads_to_ten_digits():
    with patch.object(EdgarClient, "_load_ticker_map", return_value=FAKE_TICKER_MAP):
        client = EdgarClient(user_agent="Test test@example.com")
        assert client.get_cik_for_ticker("AAPL") == "0000320193"
        assert client.get_cik_for_ticker("aapl") == "0000320193"  # case-insensitive


def test_get_cik_for_ticker_raises_on_unknown_ticker():
    with patch.object(EdgarClient, "_load_ticker_map", return_value=FAKE_TICKER_MAP):
        client = EdgarClient(user_agent="Test test@example.com")
        try:
            client.get_cik_for_ticker("NOPE")
            assert False, "expected ValueError for unknown ticker"
        except ValueError:
            pass

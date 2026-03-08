"""Tests for CCXTFetcher – all tests run without a real exchange connection.

The ccxt exchange object is replaced by a MagicMock so no network calls
are made.  This keeps tests fast and reproducible in CI environments.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.ccxt_fetcher import CCXTFetcher, _CANDLES_PER_REQUEST


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candles(n: int, start_ms: int = 0, step_ms: int = 86_400_000) -> list[list]:
    """Generate n synthetic OHLCV candles."""
    return [
        [start_ms + i * step_ms, 100.0 + i, 110.0 + i, 90.0 + i, 105.0 + i, 1000.0 + i]
        for i in range(n)
    ]


@pytest.fixture()
def mock_exchange():
    """Return a MagicMock that mimics a ccxt exchange instance."""
    ex = MagicMock()
    ex.has = {"fetchOHLCV": True}
    ex.timeframes = {"1m": "1m", "1h": "1h", "1d": "1d"}
    ex.markets = {
        "BTC/USDT": {"type": "spot"},
        "ETH/USDT": {"type": "spot"},
        "BTC/USDT:USDT": {"type": "swap"},
    }
    return ex


@pytest.fixture()
def fetcher(tmp_path, mock_exchange):
    """Return a CCXTFetcher with a mocked exchange and temp cache dir."""
    with patch("ccxt.binance", return_value=mock_exchange):
        f = CCXTFetcher(exchange_id="binance", cache_dir=tmp_path / "cache")
    f.exchange = mock_exchange
    return f


# ---------------------------------------------------------------------------
# _to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    def test_columns_and_index(self):
        candles = _make_candles(5)
        df = CCXTFetcher._to_dataframe(candles)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_utc_timezone(self):
        df = CCXTFetcher._to_dataframe(_make_candles(3))
        assert str(df.index.tz) == "UTC"

    def test_values_are_float(self):
        df = CCXTFetcher._to_dataframe(_make_candles(3))
        assert all(df.dtypes == float)


# ---------------------------------------------------------------------------
# _to_ms
# ---------------------------------------------------------------------------

class TestToMs:
    def test_known_date(self):
        ms = CCXTFetcher._to_ms("2020-01-01")
        assert ms == 1_577_836_800_000  # 2020-01-01 00:00:00 UTC in ms

    def test_returns_int(self):
        assert isinstance(CCXTFetcher._to_ms("2022-06-15"), int)


# ---------------------------------------------------------------------------
# _fetch  (single page – fewer candles than limit)
# ---------------------------------------------------------------------------

class TestFetch:
    def test_single_page(self, fetcher, mock_exchange):
        candles = _make_candles(10)
        mock_exchange.fetch_ohlcv.return_value = candles

        df = fetcher._fetch("BTC/USDT", "1d", "2022-01-01", "2022-01-10")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        mock_exchange.fetch_ohlcv.assert_called_once()

    def test_pagination(self, fetcher, mock_exchange):
        """When exchange returns exactly LIMIT candles, fetcher paginates."""
        page1 = _make_candles(_CANDLES_PER_REQUEST, start_ms=0)
        page2 = _make_candles(5, start_ms=_CANDLES_PER_REQUEST * 86_400_000)
        # page2 has fewer candles than LIMIT → loop stops without a 3rd call
        mock_exchange.fetch_ohlcv.side_effect = [page1, page2]

        df = fetcher._fetch("BTC/USDT", "1d", None, None)

        assert len(df) == _CANDLES_PER_REQUEST + 5
        assert mock_exchange.fetch_ohlcv.call_count == 2

    def test_empty_response_raises(self, fetcher, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = []
        with pytest.raises(ValueError, match="No OHLCV data"):
            fetcher._fetch("BTC/USDT", "1d", None, None)

    def test_no_fetchohlcv_support_raises(self, fetcher, mock_exchange):
        mock_exchange.has = {"fetchOHLCV": False}
        with pytest.raises(NotImplementedError):
            fetcher._fetch("BTC/USDT", "1d", None, None)

    def test_until_filter(self, fetcher, mock_exchange):
        """Candles beyond 'until' are stripped."""
        # 5 daily candles starting 2020-01-01
        start_ms = CCXTFetcher._to_ms("2020-01-01")
        day_ms = 86_400_000
        candles = _make_candles(5, start_ms=start_ms, step_ms=day_ms)
        mock_exchange.fetch_ohlcv.return_value = candles

        until_ms = start_ms + 2 * day_ms  # keep only first 3 candles (0, 1, 2)
        until_str = "2020-01-03"

        df = fetcher._fetch("BTC/USDT", "1d", "2020-01-01", until_str)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# fetch_ohlcv  (public method with caching)
# ---------------------------------------------------------------------------

class TestFetchOhlcv:
    def test_caches_result(self, fetcher, mock_exchange, tmp_path):
        candles = _make_candles(5)
        mock_exchange.fetch_ohlcv.return_value = candles

        df1 = fetcher.fetch_ohlcv("BTC/USDT", "1d", "2022-01-01", "2022-01-05")
        df2 = fetcher.fetch_ohlcv("BTC/USDT", "1d", "2022-01-01", "2022-01-05")

        # Exchange was only called once; second call served from cache
        assert mock_exchange.fetch_ohlcv.call_count == 1
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_cache(self, fetcher, mock_exchange):
        candles = _make_candles(5)
        mock_exchange.fetch_ohlcv.return_value = candles

        fetcher.fetch_ohlcv("BTC/USDT", "1d", cache=False)
        fetcher.fetch_ohlcv("BTC/USDT", "1d", cache=False)

        assert mock_exchange.fetch_ohlcv.call_count == 2

    def test_returns_dataframe(self, fetcher, mock_exchange):
        mock_exchange.fetch_ohlcv.return_value = _make_candles(3)
        df = fetcher.fetch_ohlcv("BTC/USDT", "1d", cache=False)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    def test_list_timeframes(self, fetcher, mock_exchange):
        tfs = fetcher.list_timeframes()
        assert "1d" in tfs

    def test_list_symbols_spot(self, fetcher, mock_exchange):
        symbols = fetcher.list_symbols(market_type="spot")
        assert "BTC/USDT" in symbols
        assert "BTC/USDT:USDT" not in symbols

    def test_list_symbols_swap(self, fetcher, mock_exchange):
        symbols = fetcher.list_symbols(market_type="swap")
        assert "BTC/USDT:USDT" in symbols


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_invalid_exchange_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown ccxt exchange"):
            CCXTFetcher(exchange_id="not_a_real_exchange_xyz", cache_dir=tmp_path)

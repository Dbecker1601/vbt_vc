"""OHLCV data fetcher backed by the ccxt library.

Supports any ccxt-compatible exchange: Binance, Bybit, OKX, and 100+
others. Works for spot markets, perpetual futures (``BTC/USDT:USDT``),
dated futures, and forex pairs (where the exchange offers them).

Usage example::

    from data import CCXTFetcher

    fetcher = CCXTFetcher(exchange_id="binance")
    df = fetcher.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1d",
        since="2022-01-01",
        until="2024-01-01",
        cache=True,
    )
    print(df.tail())
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .base_fetcher import BaseFetcher

# ccxt is an optional dependency – imported lazily so the rest of the
# codebase stays importable even without it installed.
try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ccxt is required for CCXTFetcher. Install it with: pip install ccxt"
    ) from exc

_CANDLES_PER_REQUEST = 1000  # most exchanges cap at 1 000 candles/call


class CCXTFetcher(BaseFetcher):
    """Fetch OHLCV data from any ccxt-compatible exchange.

    Parameters
    ----------
    exchange_id:
        ccxt exchange identifier, e.g. ``"binance"``, ``"bybit"``,
        ``"okx"``.  See ``ccxt.exchanges`` for the full list.
    cache_dir:
        Directory for Parquet cache files.  Defaults to
        ``data/cache/`` relative to this file.
    **exchange_kwargs:
        Extra keyword arguments forwarded to the ccxt exchange
        constructor, e.g. ``{"enableRateLimit": True}``.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        cache_dir: Path | None = None,
        **exchange_kwargs,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        if not hasattr(ccxt, exchange_id):
            raise ValueError(
                f"Unknown ccxt exchange: '{exchange_id}'. "
                f"Check ccxt.exchanges for valid IDs."
            )
        default_kwargs = {"enableRateLimit": True}
        default_kwargs.update(exchange_kwargs)
        self.exchange: ccxt.Exchange = getattr(ccxt, exchange_id)(default_kwargs)
        self._exchange_id = exchange_id

    # ------------------------------------------------------------------
    # BaseFetcher interface
    # ------------------------------------------------------------------

    def _source_id(self) -> str:
        return self._exchange_id

    def _fetch(
        self,
        symbol: str,
        timeframe: str,
        since: str | None,
        until: str | None,
    ) -> pd.DataFrame:
        if not self.exchange.has.get("fetchOHLCV"):
            raise NotImplementedError(
                f"Exchange '{self._exchange_id}' does not support fetchOHLCV."
            )

        since_ms = self._to_ms(since) if since else None
        until_ms = self._to_ms(until) if until else None

        all_candles: list[list] = []
        cursor_ms = since_ms

        while True:
            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=cursor_ms,
                limit=_CANDLES_PER_REQUEST,
            )
            if not candles:
                break

            # Filter out candles beyond 'until'
            if until_ms is not None:
                candles = [c for c in candles if c[0] <= until_ms]

            all_candles.extend(candles)

            # Stop if we received fewer candles than requested (end of data)
            # or if the last candle is already past 'until'
            if len(candles) < _CANDLES_PER_REQUEST:
                break
            if until_ms is not None and candles[-1][0] >= until_ms:
                break

            # Advance cursor to the timestamp after the last candle
            cursor_ms = candles[-1][0] + 1

        if not all_candles:
            raise ValueError(
                f"No OHLCV data returned for {symbol} on {self._exchange_id} "
                f"(timeframe={timeframe}, since={since}, until={until})."
            )

        return self._to_dataframe(all_candles)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ms(date_str: str) -> int:
        """Convert an ISO-8601 date string to milliseconds since epoch."""
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _to_dataframe(candles: list[list]) -> pd.DataFrame:
        """Convert raw ccxt candle list to a labelled DataFrame."""
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df.index.name = "Date"
        return df.astype(float)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def list_timeframes(self) -> list[str]:
        """Return the timeframes supported by the exchange."""
        return list(self.exchange.timeframes.keys())

    def list_symbols(self, market_type: str = "spot") -> list[str]:
        """Return available symbols for a given market type.

        Parameters
        ----------
        market_type:
            One of ``"spot"``, ``"future"``, ``"swap"``, ``"option"``.
        """
        self.exchange.load_markets()
        return [
            s
            for s, m in self.exchange.markets.items()
            if m.get("type") == market_type
        ]

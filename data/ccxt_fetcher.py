"""
CCXT-based data fetcher for Bitcoin (and any other CCXT-supported asset).

Provides:
  - fetch_ohlcv() : Paginated historical OHLCV download.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd


def fetch_ohlcv(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: Optional[int] = None,
    until: Optional[int] = None,
    limit: int = 1000,
    exchange_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Download paginated historical OHLCV data from any CCXT-compatible exchange.

    Parameters
    ----------
    exchange_id : str
        CCXT exchange ID (e.g. 'binance', 'bybit', 'kraken').
    symbol : str
        Market symbol (e.g. 'BTC/USDT').
    timeframe : str
        Candle duration (e.g. '1m', '5m', '1h', '1d').
    since : int, optional
        Start time as Unix timestamp in **milliseconds**.
        If None, fetches from the earliest available data.
    until : int, optional
        Stop fetching after this timestamp (ms). If None, fetches up to now.
    limit : int
        Candles per API call (max per exchange; typically 500–1000).
    exchange_params : dict, optional
        Additional parameters passed to the exchange constructor
        (e.g. {'apiKey': '...', 'secret': '...'} for private endpoints).

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume  (DatetimeIndex, UTC).
    """
    import ccxt  # imported lazily so the module loads without ccxt installed

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class(exchange_params or {})

    all_ohlcv: list = []
    current_since = since

    while True:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe, since=current_since, limit=limit
        )
        if not candles:
            break

        all_ohlcv.extend(candles)

        last_ts = candles[-1][0]
        if until is not None and last_ts >= until:
            break
        if len(candles) < limit:
            break

        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_ohlcv:
        raise RuntimeError(
            f"No OHLCV data returned from {exchange_id} for {symbol} {timeframe}."
        )

    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if until is not None:
        until_dt = pd.to_datetime(until, unit="ms", utc=True)
        df = df[df.index <= until_dt]

    return df.astype(np.float64)
"""
CCXT-based data fetcher for Bitcoin (and any other CCXT-supported asset).

Provides:
  - fetch_ohlcv()                  : Paginated historical OHLCV download.
  - fetch_orderbook_snapshot()     : Single live order-book snapshot.
  - OrderBookSnapshot              : Dataclass with feature extraction helpers.
  - simulate_l2_features_from_ohlcv(): L2 feature proxies derived from OHLCV
                                       (used for historical back-testing when
                                        real tick-level order book data is
                                        unavailable).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OrderBookSnapshot
# ---------------------------------------------------------------------------

@dataclass
class OrderBookSnapshot:
    """
    A single order-book snapshot as returned by CCXT fetch_order_book().

    Attributes
    ----------
    timestamp : int
        Unix timestamp in milliseconds.
    bids : list of [price, volume]
        Sorted descending by price (best bid first).
    asks : list of [price, volume]
        Sorted ascending by price (best ask first).
    """
    timestamp: int
    bids: List[List[float]]
    asks: List[List[float]]

    # ------------------------------------------------------------------
    # Basic microstructure metrics
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def relative_spread(self) -> float:
        mid = self.mid_price
        return self.spread / mid if mid > 0 else 0.0

    def imbalance(self, depth: int = 10) -> float:
        """Bid volume / (bid + ask volume) at top *depth* levels."""
        bid_vol = sum(b[1] for b in self.bids[:depth])
        ask_vol = sum(a[1] for a in self.asks[:depth])
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def cum_bid_volume(self, depth: int) -> float:
        return sum(b[1] for b in self.bids[:depth])

    def cum_ask_volume(self, depth: int) -> float:
        return sum(a[1] for a in self.asks[:depth])

    def vwap_bid(self, depth: int = 5) -> float:
        """Volume-weighted average bid price at top *depth* levels."""
        total_vol = self.cum_bid_volume(depth)
        if total_vol == 0:
            return self.best_bid
        return sum(b[0] * b[1] for b in self.bids[:depth]) / total_vol

    def vwap_ask(self, depth: int = 5) -> float:
        """Volume-weighted average ask price at top *depth* levels."""
        total_vol = self.cum_ask_volume(depth)
        if total_vol == 0:
            return self.best_ask
        return sum(a[0] * a[1] for a in self.asks[:depth]) / total_vol

    def to_features(self, depth: int = 10) -> np.ndarray:
        """
        Extract 8 normalised L2 features suitable for the observation vector.

        Feature layout (matches simulate_l2_features_from_ohlcv columns):
          0: relative_spread
          1: imbalance(depth)
          2: top-1 bid volume share (top-1 / cumulative depth)
          3: top-1 ask volume share
          4: VWAP bid deviation from mid  (normalised by mid)
          5: VWAP ask deviation from mid
          6: log(1 + cumulative bid volume at depth)
          7: log(1 + cumulative ask volume at depth)
        """
        mid = self.mid_price
        if mid <= 0:
            return np.zeros(8, dtype=np.float32)

        bid_depth_total = self.cum_bid_volume(depth) + 1e-10
        ask_depth_total = self.cum_ask_volume(depth) + 1e-10

        features = np.array([
            self.relative_spread,
            self.imbalance(depth),
            self.cum_bid_volume(1) / bid_depth_total,
            self.cum_ask_volume(1) / ask_depth_total,
            (self.vwap_bid(5) - mid) / mid,
            (self.vwap_ask(5) - mid) / mid,
            np.log1p(bid_depth_total - 1e-10),
            np.log1p(ask_depth_total - 1e-10),
        ], dtype=np.float32)

        return features


# ---------------------------------------------------------------------------
# CCXT helpers
# ---------------------------------------------------------------------------

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


def fetch_orderbook_snapshot(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    depth: int = 20,
    exchange_params: Optional[dict] = None,
) -> OrderBookSnapshot:
    """
    Fetch a single live order-book snapshot.

    Parameters
    ----------
    exchange_id : str
        CCXT exchange ID.
    symbol : str
        Market symbol.
    depth : int
        Number of price levels to retrieve on each side.
    exchange_params : dict, optional
        Constructor parameters (API keys, etc.).

    Returns
    -------
    OrderBookSnapshot
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class(exchange_params or {})
    ob = exchange.fetch_order_book(symbol, depth)

    return OrderBookSnapshot(
        timestamp=ob.get("timestamp") or int(time.time() * 1000),
        bids=ob["bids"][:depth],
        asks=ob["asks"][:depth],
    )


# ---------------------------------------------------------------------------
# L2 feature simulation from OHLCV (for historical back-testing)
# ---------------------------------------------------------------------------

def simulate_l2_features_from_ohlcv(df: pd.DataFrame, depth: int = 10) -> pd.DataFrame:
    """
    Derive Level-2 order-book feature *proxies* from OHLCV bars.

    Real historical order-book data is expensive / unavailable on most
    exchanges.  This function produces reasonable approximations using
    market-microstructure intuition:

    - Spread proxy  : (High – Low) * spread_factor / mid_price
                      High-Low range is correlated with intrabar bid-ask spread.
    - Imbalance     : (Close – Low) / (High – Low)
                      When price closes near the High, buyers were dominant → bid
                      imbalance > 0.5 (and vice-versa for sellers).
    - Depth proxies : Scaled from bar Volume, split by imbalance.

    The 8-column output matches OrderBookSnapshot.to_features() layout so that
    real and simulated L2 data are interchangeable in BitcoinEnv.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Open, High, Low, Close, Volume columns.
    depth : int
        Notional depth used for depth-share calculations (affects scaling only).

    Returns
    -------
    pd.DataFrame
        8 float32 columns, same index as *df*.
    """
    high_low = (df["High"] - df["Low"]).clip(lower=1e-10)
    mid = (df["High"] + df["Low"]) / 2.0

    # 0: relative spread (proxy: 5 % of intrabar range / mid)
    spread = (high_low * 0.05 / mid).clip(0.0, 0.02)

    # 1: order book imbalance
    imbalance = ((df["Close"] - df["Low"]) / high_low).clip(0.0, 1.0)

    # 2 & 3: top-1 depth share (proxy: concentrated near mid when imbalanced)
    bid_top1 = (imbalance * 0.25 + 0.05).clip(0.0, 1.0)
    ask_top1 = ((1.0 - imbalance) * 0.25 + 0.05).clip(0.0, 1.0)

    # 4 & 5: VWAP bid/ask deviations from mid (proxy: ±half-spread)
    vwap_bid_dev = -spread * 0.4
    vwap_ask_dev =  spread * 0.4

    # 6 & 7: log cumulative depth (proxy: log-volume split by imbalance)
    log_vol = np.log1p(df["Volume"])
    log_bid = log_vol * imbalance
    log_ask = log_vol * (1.0 - imbalance)

    result = pd.DataFrame(
        {
            "l2_spread":         spread,
            "l2_imbalance":      imbalance,
            "l2_bid_top1_share": bid_top1,
            "l2_ask_top1_share": ask_top1,
            "l2_vwap_bid_dev":   vwap_bid_dev,
            "l2_vwap_ask_dev":   vwap_ask_dev,
            "l2_log_bid_depth":  log_bid,
            "l2_log_ask_depth":  log_ask,
        },
        index=df.index,
    )
    return result.astype(np.float32)

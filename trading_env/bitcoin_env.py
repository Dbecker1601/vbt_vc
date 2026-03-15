"""
BitcoinEnv – deep-RL trading environment for Bitcoin with Level-2 features.

Observation vector (flat, per step):
    [window_size × 16 features]  +  [position_value]  +  [tick_ratio]

16 features per tick = 8 OHLCV/technical + 8 Level-2 order-book:

  OHLCV / technical indicators (8):
    0  log_return        – log price return vs previous bar
    1  hl_range          – (High – Low) / Close
    2  co_range          – (Close – Open) / Close
    3  vol_ratio         – Volume / 20-bar moving-average volume
    4  rsi               – RSI(14), normalised to [0, 1]
    5  macd              – (EMA12 – EMA26) / Close
    6  bb_pos            – Bollinger-Band position, clipped to [−1, 1]
    7  atr               – ATR(14) / Close

  Level-2 order-book (8)  – real or simulated via simulate_l2_features_from_ohlcv:
    8  l2_spread          – relative bid-ask spread
    9  l2_imbalance       – bid volume / (bid + ask) volume at top N levels
   10  l2_bid_top1_share  – top-1 bid volume share within total bid depth
   11  l2_ask_top1_share  – top-1 ask volume share within total ask depth
   12  l2_vwap_bid_dev    – VWAP bid deviation from mid (normalised)
   13  l2_vwap_ask_dev    – VWAP ask deviation from mid (normalised)
   14  l2_log_bid_depth   – log(1 + total bid volume at depth)
   15  l2_log_ask_depth   – log(1 + total ask volume at depth)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from trading_env.trading_env import TradingEnv, Actions, Positions


# ---------------------------------------------------------------------------
# Technical indicator helper
# ---------------------------------------------------------------------------

def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 8 OHLCV-derived technical features for every row in *df*.

    All features are bounded / dimensionless so that z-score normalisation
    in BitcoinEnv._process_data() works well across different price regimes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Open, High, Low, Close, Volume columns.

    Returns
    -------
    pd.DataFrame
        8 float32 columns, same index as *df*.  NaN-filled with 0.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    result = pd.DataFrame(index=df.index)

    # Log return (first bar is 0 by convention)
    result["log_return"] = np.log(close / close.shift(1)).fillna(0.0)

    # Intrabar range features
    result["hl_range"] = (high - low) / close.clip(lower=1e-10)
    result["co_range"] = (close - df["Open"]) / close.clip(lower=1e-10)

    # Volume ratio vs. 20-bar MA (capped to avoid outlier spikes)
    vol_ma = volume.rolling(20, min_periods=1).mean()
    result["vol_ratio"] = (volume / vol_ma.clip(lower=1e-10)).clip(0.0, 10.0)

    # RSI(14) – normalised to [0, 1]
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs   = gain / (loss + 1e-10)
    result["rsi"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0

    # MACD signal normalised by Close
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["macd"] = ((ema12 - ema26) / close.clip(lower=1e-10)).clip(-0.1, 0.1)

    # Bollinger-Band position: (Close – SMA20) / (2 * std20), clipped to [−1, 1]
    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
    result["bb_pos"] = (
        (close - sma20) / (2.0 * std20 + 1e-10)
    ).clip(-2.0, 2.0) / 2.0

    # ATR(14) normalised by Close
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr"] = (
        tr.rolling(14, min_periods=1).mean() / close.clip(lower=1e-10)
    ).clip(0.0, 0.2)

    return result.fillna(0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# BitcoinEnv
# ---------------------------------------------------------------------------

class BitcoinEnv(TradingEnv):
    """
    Bitcoin trading environment with Level-2 order-book features.

    Extends :class:`trading_env.TradingEnv` with richer feature engineering
    (technical indicators + L2 order-book data) and Binance-style BTC fees.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
        Any DatetimeIndex is accepted; numeric positional indexing is used
        internally.
    l2_df : pd.DataFrame, optional
        Pre-computed L2 feature DataFrame with 8 columns in the same order as
        ``OrderBookSnapshot.to_features()``.  If *None*, features are
        simulated via :func:`data.ccxt_fetcher.simulate_l2_features_from_ohlcv`.
    window_size : int
        Number of past bars visible to the agent (default: 20).
    frame_bound : tuple[int, int], optional
        (start_index, end_index) within *df* for this episode.
        Defaults to (window_size, len(df)).
    render_mode : str or None
        'human' for live matplotlib rendering.
    trade_fee_bid_percent : float
        Taker fee on buys (default: 0.001 = 0.1 % – Binance spot taker fee).
    trade_fee_ask_percent : float
        Taker fee on sells (default: 0.001 = 0.1 %).
    """

    OHLCV_FEATURE_DIM: int = 8
    L2_FEATURE_DIM: int    = 8
    TOTAL_FEATURE_DIM: int = OHLCV_FEATURE_DIM + L2_FEATURE_DIM  # 16

    def __init__(
        self,
        df: pd.DataFrame,
        l2_df: Optional[pd.DataFrame] = None,
        window_size: int = 20,
        frame_bound: Optional[Tuple[int, int]] = None,
        render_mode: Optional[str] = None,
        trade_fee_bid_percent: float = 0.001,
        trade_fee_ask_percent: float = 0.001,
    ):
        assert "Close" in df.columns, "df must contain a 'Close' column"

        self._l2_df = l2_df

        if frame_bound is None:
            frame_bound = (window_size, len(df))

        super().__init__(
            df=df,
            window_size=window_size,
            frame_bound=frame_bound,
            render_mode=render_mode,
            trade_fee_bid_percent=trade_fee_bid_percent,
            trade_fee_ask_percent=trade_fee_ask_percent,
        )

    # ------------------------------------------------------------------
    # _process_data
    # ------------------------------------------------------------------

    def _process_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build the 16-feature signal matrix for the episode frame.

        Technical indicators are computed on the *full* df so that EMA / RSI
        warm-up periods are satisfied before the episode slice begins.  The
        result is then sliced to [start − window_size : end] and z-score
        normalised over the slice (matching DI-engine convention).
        """
        start, end = self.frame_bound

        # 1. Compute indicators on full df (ensures proper EMA / RSI warmup)
        tech = _compute_technical_indicators(self.df)  # (len(df), 8)

        # 2. L2 features (real or simulated)
        if self._l2_df is not None:
            l2 = self._l2_df
        else:
            from data.ccxt_fetcher import simulate_l2_features_from_ohlcv
            l2 = simulate_l2_features_from_ohlcv(self.df)   # (len(df), 8)

        # 3. Slice to episode frame (includes lookback window)
        idx = slice(start - self.window_size, end)
        raw_prices     = self.df["Close"].to_numpy(dtype=np.float32)[idx]
        tech_slice     = tech.values[idx]    # (frame_len, 8)
        l2_slice       = l2.values[idx]      # (frame_len, 8)

        signal_features = np.concatenate([tech_slice, l2_slice], axis=1)  # (frame_len, 16)

        # 4. Z-score normalise per column over the episode slice
        EPS = 1e-10
        mean = signal_features.mean(axis=0)
        std  = signal_features.std(axis=0)
        signal_features = (signal_features - mean) / (std + EPS)

        # Keep raw prices accessible as self.prices (required by base class)
        self.prices = raw_prices

        return raw_prices, signal_features.astype(np.float32), self.TOTAL_FEATURE_DIM

    # ------------------------------------------------------------------
    # _calculate_reward  (same DI-engine log-reward as StocksEnv)
    # ------------------------------------------------------------------

    def _calculate_reward(self, action: int) -> float:
        """
        Logarithmic reward: non-zero only when a trade cycle closes.

        - LONG cycle closed (SELL / DOUBLE_SELL from LONG):
            reward = log(current / prev_trade) + log(cost)
        - SHORT cycle closed (BUY / DOUBLE_BUY from SHORT):
            reward = log(2 − current / prev_trade) + log(cost)
        """
        step_reward = 0.0

        current_price    = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        ratio = current_price / (last_trade_price + 1e-10)
        cost  = np.log(
            (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)
        )

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2.0 - ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2.0 - ratio) + cost

        return float(step_reward)

    # ------------------------------------------------------------------
    # max_possible_profit  (DI-engine greedy algorithm)
    # ------------------------------------------------------------------

    def max_possible_profit(self) -> float:
        """
        Theoretical upper-bound profit computed by the DI-engine greedy
        algorithm (always trade perfectly with fees applied).
        """
        current_tick    = self._start_tick
        last_trade_tick = current_tick - 1
        profit          = 1.0

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                # Downswing: short opportunity
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] < self.prices[current_tick - 1]
                ):
                    current_tick += 1
                ratio  = self.prices[current_tick - 1] / self.prices[last_trade_tick]
                profit *= max(2.0 - ratio, 1e-10)
                profit *= (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)
            else:
                # Upswing: long opportunity
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] >= self.prices[current_tick - 1]
                ):
                    current_tick += 1
                ratio  = self.prices[current_tick - 1] / self.prices[last_trade_tick]
                profit *= ratio
                profit *= (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)

            last_trade_tick = current_tick - 1

        return profit

    def __repr__(self) -> str:
        return (
            f"BitcoinEnv(frame_bound={self.frame_bound}, "
            f"window_size={self.window_size}, "
            f"l2={'real' if self._l2_df is not None else 'simulated'})"
        )

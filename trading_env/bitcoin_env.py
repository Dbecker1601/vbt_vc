"""
BitcoinEnv – deep-RL trading environment for Bitcoin (OHLCV + technical features).

Observation vector (flat, per step):
    [window_size × 13 features]  +  [position_value]  +  [tick_ratio]

13 features per tick:

  OHLCV / technical indicators (8):
    0  log_return   – log price return vs previous bar
    1  hl_range     – (High – Low) / Close
    2  co_range     – (Close – Open) / Close
    3  vol_ratio    – Volume / 20-bar MA volume
    4  rsi          – RSI(14), normalised to [0, 1]
    5  macd         – (EMA12 – EMA26) / Close
    6  bb_pos       – Bollinger-Band position, clipped to [−1, 1]
    7  atr          – ATR(14) / Close

  Extended features (5):
    8  ema200_dev   – (Close – EMA200) / Close  (trend distance)
    9  vol_delta    – buyer/seller pressure proxy from intrabar range
   10  vwap_dev     – (Close – daily VWAP) / Close  (0 if no datetime info)
   11  hour_sin     – sin(2π × hour/24)  (0 if no datetime info)
   12  hour_cos     – cos(2π × hour/24)  (0 if no datetime info)

Improvements over baseline:
  - 5 extra features (EMA200, volume delta, VWAP, time-of-day cyclical)
  - Intermediate step reward (small unrealized P&L signal each bar)
  - Higher default fees (0.2 %) to encourage selective trading
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
    Compute 13 OHLCV-derived technical features for every row in *df*.

    If *df* contains a 'datetime' column (pd.Timestamp or similar), the
    function also computes daily-VWAP deviation and time-of-day features.
    Otherwise those three columns are filled with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Open, High, Low, Close, Volume columns.
        May optionally contain a 'datetime' column.

    Returns
    -------
    pd.DataFrame
        13 float32 columns, same index as *df*.  NaN-filled with 0.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    volume = df["Volume"]

    result = pd.DataFrame(index=df.index)

    # ── Original 8 features ────────────────────────────────────────────

    # 0: Log return (first bar = 0)
    result["log_return"] = np.log(close / close.shift(1)).fillna(0.0)

    # 1-2: Intrabar range features
    result["hl_range"] = (high - low) / close.clip(lower=1e-10)
    result["co_range"] = (close - open_) / close.clip(lower=1e-10)

    # 3: Volume ratio vs 20-bar MA (capped to avoid outlier spikes)
    vol_ma = volume.rolling(20, min_periods=1).mean()
    result["vol_ratio"] = (volume / vol_ma.clip(lower=1e-10)).clip(0.0, 10.0)

    # 4: RSI(14) normalised to [0, 1]
    delta = close.diff()
    gain  = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs    = gain / (loss + 1e-10)
    result["rsi"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0

    # 5: MACD normalised by Close
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["macd"] = ((ema12 - ema26) / close.clip(lower=1e-10)).clip(-0.1, 0.1)

    # 6: Bollinger-Band position (Close – SMA20) / (2 * std20)
    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
    result["bb_pos"] = (
        (close - sma20) / (2.0 * std20 + 1e-10)
    ).clip(-2.0, 2.0) / 2.0

    # 7: ATR(14) / Close
    tr = pd.concat(
        [high - low,
         (high - close.shift(1)).abs(),
         (low  - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    result["atr"] = (
        tr.rolling(14, min_periods=1).mean() / close.clip(lower=1e-10)
    ).clip(0.0, 0.2)

    # ── Extended 5 features ────────────────────────────────────────────

    # 8: EMA200 deviation – trend distance, clipped to ±20 %
    ema200 = close.ewm(span=200, adjust=False).mean()
    result["ema200_dev"] = (
        (close - ema200) / close.clip(lower=1e-10)
    ).clip(-0.2, 0.2)

    # 9: Volume delta – buyer/seller pressure from intrabar price action
    hl = (high - low).clip(lower=1e-10)
    result["vol_delta"] = ((2.0 * close - high - low) / hl).clip(-1.0, 1.0)

    # 10-12: VWAP deviation + time-of-day (only if datetime info available)
    if "datetime" in df.columns:
        dt    = pd.to_datetime(df["datetime"])
        dates = dt.dt.date

        typical  = (high + low + close) / 3.0
        vwap_num = (typical * volume).groupby(dates).cumsum()
        vwap_den = volume.groupby(dates).cumsum().clip(lower=1e-10)
        vwap     = vwap_num / vwap_den

        result["vwap_dev"] = (
            (close - vwap) / close.clip(lower=1e-10)
        ).clip(-0.05, 0.05)

        # Cyclical hour encoding (minute precision)
        hour_frac = dt.dt.hour + dt.dt.minute / 60.0
        result["hour_sin"] = np.sin(2.0 * np.pi * hour_frac / 24.0)
        result["hour_cos"] = np.cos(2.0 * np.pi * hour_frac / 24.0)
    else:
        result["vwap_dev"]  = 0.0
        result["hour_sin"]  = 0.0
        result["hour_cos"]  = 0.0

    return result.fillna(0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# BitcoinEnv
# ---------------------------------------------------------------------------

class BitcoinEnv(TradingEnv):
    """
    Bitcoin trading environment with OHLCV-derived technical features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame. If a 'datetime' column is present, VWAP and
        time-of-day features are activated.
    window_size : int
        Number of past bars visible to the agent (default: 60 for 1 m bars).
    frame_bound : tuple[int, int], optional
        (start_index, end_index) within *df*. Defaults to (window_size, len(df)).
    render_mode : str or None
    trade_fee_bid_percent : float
        Taker fee on buys (default: 0.002 = 0.2 %).
    trade_fee_ask_percent : float
        Taker fee on sells (default: 0.002 = 0.2 %).
    intermediate_reward_scale : float
        Scale of the per-step unrealised P&L reward (default: 0.05).
        Set to 0 to disable.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        frame_bound: Optional[Tuple[int, int]] = None,
        render_mode: Optional[str] = None,
        trade_fee_bid_percent: float = 0.002,
        trade_fee_ask_percent: float = 0.002,
        intermediate_reward_scale: float = 0.001,
    ):
        assert "Close" in df.columns, "df must contain a 'Close' column"

        self._intermediate_reward_scale = intermediate_reward_scale

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
        Build the feature signal matrix for the episode frame.

        Feature count is dynamic (13 when datetime is present, 10 otherwise).
        """
        start, end = self.frame_bound

        # Compute on full df for proper EMA/RSI warmup
        tech = _compute_technical_indicators(self.df)      # (len(df), F)
        feature_dim = tech.shape[1]

        idx             = slice(start - self.window_size, end)
        raw_prices      = self.df["Close"].to_numpy(dtype=np.float32)[idx]
        signal_features = tech.values[idx].copy()          # (frame_len, F)

        # Z-score normalise per column over the episode slice
        mean            = signal_features.mean(axis=0)
        std             = signal_features.std(axis=0)
        signal_features = (signal_features - mean) / (std + 1e-10)

        self.prices = raw_prices

        return raw_prices, signal_features.astype(np.float32), feature_dim

    # ------------------------------------------------------------------
    # _calculate_reward
    # ------------------------------------------------------------------

    def _calculate_reward(self, action: int) -> float:
        """
        Two-component reward:

        1. **Terminal reward** (non-zero only when trade cycle closes):
             LONG closed  → log(current / entry) + log(1 - fees²)
             SHORT closed → log(2 − current / entry) + log(1 - fees²)

        2. **Intermediate reward** (every step while in a position):
             small fraction of the one-bar log return in the position direction.
             This gives the agent a learning signal before the trade closes.
        """
        step_reward   = 0.0
        current_price = self.prices[self._current_tick]
        prev_price    = self.prices[max(self._current_tick - 1, 0)]
        last_price    = self.prices[self._last_trade_tick]
        ratio         = current_price / (last_price + 1e-10)
        cost          = np.log(
            (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)
        )

        # ── 1. Terminal reward on trade close ──────────────────────────
        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        elif action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        elif action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2.0 - ratio) + cost

        elif action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2.0 - ratio) + cost

        # ── 2. Intermediate step reward while in a position ────────────
        if self._intermediate_reward_scale > 0 and prev_price > 0:
            bar_log_ret = np.log(current_price / (prev_price + 1e-10))
            if self._position == Positions.LONG:
                step_reward += self._intermediate_reward_scale * bar_log_ret
            elif self._position == Positions.SHORT:
                step_reward -= self._intermediate_reward_scale * bar_log_ret

        return float(step_reward)

    # ------------------------------------------------------------------
    # max_possible_profit
    # ------------------------------------------------------------------

    def max_possible_profit(self) -> float:
        """
        Theoretical upper-bound profit: only trade swings that are larger
        than the round-trip fee cost (avoids fee-death on 1 m data).
        """
        fee_rt    = (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)
        current_tick    = self._start_tick
        last_trade_tick = current_tick - 1
        profit          = 1.0

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] < self.prices[current_tick - 1]
                ):
                    current_tick += 1
                ratio = self.prices[current_tick - 1] / (self.prices[last_trade_tick] + 1e-10)
                if (2.0 - ratio) * fee_rt > 1.0:   # profitable short after fees
                    profit *= max(2.0 - ratio, 1e-10) * fee_rt
            else:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] >= self.prices[current_tick - 1]
                ):
                    current_tick += 1
                ratio = self.prices[current_tick - 1] / (self.prices[last_trade_tick] + 1e-10)
                if ratio * fee_rt > 1.0:            # profitable long after fees
                    profit *= ratio * fee_rt

            last_trade_tick = current_tick - 1

        return profit

    def __repr__(self) -> str:
        return (
            f"BitcoinEnv(frame_bound={self.frame_bound}, "
            f"window_size={self.window_size})"
        )

"""
StocksEnv – concrete trading environment for stock market data.

Feature engineering, reward function, and max-profit calculation follow the
DI-engine implementation:
  https://github.com/opendilab/DI-engine/blob/main/dizoo/gym_anytrading/envs/stocks_env.py

The environment extends the gymnasium-compatible TradingEnv base class so that
it can be used directly with Stable Baselines 3.
"""

from typing import Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

from trading_env.trading_env import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):
    """
    Stock-market trading environment.

    Uses three features (Close, Diff, Volume) as in DI-engine, applies
    z-score normalisation over the episode frame, and uses the DI-engine
    logarithmic reward function.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame indexed by date with at least the columns
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'].
    window_size : int
        Number of past ticks visible to the agent.
    frame_bound : tuple[int, int]
        (start_index, end_index) – the slice of *df* used for this episode.
    render_mode : str or None
        'human' for live matplotlib rendering, None otherwise.
    trade_fee_bid_percent : float
        Proportional transaction cost on the buy side (default 1 %).
    trade_fee_ask_percent : float
        Proportional transaction cost on the sell side (default 0.5 %).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        frame_bound: Tuple[int, int],
        render_mode: Optional[str] = None,
        trade_fee_bid_percent: float = 0.01,
        trade_fee_ask_percent: float = 0.005,
    ):
        super().__init__(
            df=df,
            window_size=window_size,
            frame_bound=frame_bound,
            render_mode=render_mode,
            trade_fee_bid_percent=trade_fee_bid_percent,
            trade_fee_ask_percent=trade_fee_ask_percent,
        )

    # ------------------------------------------------------------------
    # _process_data – DI-engine feature engineering
    # ------------------------------------------------------------------

    def _process_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build the Close, Diff, Volume feature matrix for the episode frame.

        Raw close prices are kept un-normalised (used for reward calculation),
        while the feature matrix is z-score normalised per column – matching
        the DI-engine preprocessing.

        Returns
        -------
        raw_prices      : np.ndarray, shape (N,)
        signal_features : np.ndarray, shape (N, 3)  – normalised
        feature_dim     : int = 3
        """
        start, end = self.frame_bound
        EPS = 1e-10

        # Raw close prices for reward computation (not normalised)
        raw_prices = self.df["Close"].to_numpy()[start - self.window_size: end]

        # Build feature matrix: Close, Diff, Volume
        close  = self.df["Close"].to_numpy()[start - self.window_size: end]
        volume = self.df["Volume"].to_numpy()[start - self.window_size: end]
        diff   = np.insert(np.diff(close), 0, 0)  # first tick has no prior, set to 0 (DI-engine convention)

        features = np.column_stack([close, diff, volume])

        # Z-score normalisation (DI-engine style) per column
        mean = features.mean(axis=0)
        std  = features.std(axis=0)
        signal_features = (features - mean) / (std + EPS)

        # Expose raw close prices as self.prices (used internally)
        self.prices = raw_prices.astype(np.float32)

        return raw_prices.astype(np.float32), signal_features.astype(np.float32), 3

    # ------------------------------------------------------------------
    # _calculate_reward – DI-engine logarithmic reward
    # ------------------------------------------------------------------

    def _calculate_reward(self, action: int) -> float:
        """
        Logarithmic reward function from DI-engine.

        Reward is only non-zero when a profitable transition completes:
          - Going LONG  (buying):  log(curr / prev) + log(cost)
          - Going SHORT (shorting): log(2 − curr / prev) + log(cost)

        This means maximising Σr is equivalent to maximising the
        compound profit across long and short cycles.
        """
        step_reward = 0.0
        current_price    = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        ratio = current_price / (last_trade_price + 1e-10)
        cost  = np.log(
            (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)
        )

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2.0 - ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2.0 - ratio) + cost

        return float(step_reward)

    # ------------------------------------------------------------------
    # max_possible_profit – DI-engine calculation
    # ------------------------------------------------------------------

    def max_possible_profit(self) -> float:
        """
        Theoretical upper-bound profit for the current episode frame.

        Follows the DI-engine greedy algorithm: always enter long at the
        start of an upswing and short at the start of a downswing, with
        trading fees applied.
        """
        current_tick   = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                # Downswing: short opportunity
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] < self.prices[current_tick - 1]
                ):
                    current_tick += 1
                current_price    = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                tmp = profit * (2.0 - current_price / last_trade_price)
                tmp *= (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)
                profit = max(profit, tmp)
            else:
                # Upswing: long opportunity
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] >= self.prices[current_tick - 1]
                ):
                    current_tick += 1
                current_price    = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                tmp = profit * (current_price / last_trade_price)
                tmp *= (1.0 - self.trade_fee_ask_percent) * (1.0 - self.trade_fee_bid_percent)
                profit = max(profit, tmp)

            last_trade_tick = current_tick - 1

        return profit

    def __repr__(self) -> str:
        return (
            f"StocksEnv(frame_bound={self.frame_bound}, "
            f"window_size={self.window_size})"
        )

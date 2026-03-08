"""vectorbt-backed backtesting for RL trading agents.

The module provides two layers:

1. **Signal generation** – runs the trained RL model over an episode and
   extracts a position/profit history.
2. **Portfolio simulation** – converts the position history into vectorbt
   entry/exit signals and simulates a portfolio, then computes metrics.

If vectorbt is *not* installed, the simulation falls back to the built-in
pure-Python metrics in :mod:`backtest.metrics` so that the module remains
importable in environments without vectorbt.

Usage example::

    from stable_baselines3 import PPO
    from trading_env.stocks_env import StocksEnv
    from backtest import VbtBacktest

    model = PPO.load("models/ppo_BTC_USDT_1d.zip")
    bt = VbtBacktest(
        env_class=StocksEnv,
        df=df,
        window_size=20,
        frame_bound=(20, len(df)),
        trade_fee_bid_percent=0.001,
        trade_fee_ask_percent=0.001,
        periods_per_year=365,   # crypto – daily candles
    )
    result = bt.run(model)
    print(result.metrics.summary())
    result.plot()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

import numpy as np
import pandas as pd

from .metrics import BacktestMetrics, compute_metrics

try:
    import vectorbt as vbt  # type: ignore

    _VBT_AVAILABLE = True
except ImportError:
    _VBT_AVAILABLE = False

# ── Public types ──────────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """All outputs of a single backtest run."""

    metrics: BacktestMetrics

    # Raw series aligned to the episode ticks
    equity_curve: pd.Series
    position_series: pd.Series
    price_series: pd.Series

    # Optional vectorbt portfolio (None when vbt not installed)
    vbt_portfolio: Any | None = field(default=None, repr=False)

    # Comparison benchmark (Buy-and-Hold)
    benchmark_return: float = 0.0

    def plot(self, save_path: str | None = None) -> None:
        """Two-panel plot: price + position markers / equity curve."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed – cannot plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(
            f"Backtest  |  Return: {self.metrics.total_return:+.2%}  "
            f"Sharpe: {self.metrics.sharpe_ratio:.2f}  "
            f"MaxDD: {self.metrics.max_drawdown:.2%}",
            fontsize=12,
        )

        prices = self.price_series.values
        pos = self.position_series.values
        ticks = np.arange(len(prices))

        long_mask = pos == 1
        short_mask = pos == -1
        flat_mask = pos == 0

        ax1.plot(ticks, prices, linewidth=1, label="Close", color="steelblue")
        if long_mask.any():
            ax1.scatter(ticks[long_mask], prices[long_mask], color="green", marker="^", s=20, label="Long", zorder=3)
        if short_mask.any():
            ax1.scatter(ticks[short_mask], prices[short_mask], color="red", marker="v", s=20, label="Short", zorder=3)
        if flat_mask.any():
            ax1.scatter(ticks[flat_mask], prices[flat_mask], color="steelblue", marker="o", s=6, label="Flat", alpha=0.4, zorder=2)
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left", fontsize=8)

        equity = self.equity_curve.values
        ax2.plot(ticks[: len(equity)], equity, color="purple", linewidth=1.5, label="RL Agent")
        ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
        bh_final = 1.0 + self.benchmark_return
        bh_curve = np.linspace(1.0, bh_final, len(equity))
        ax2.plot(ticks[: len(equity)], bh_curve, color="orange", linewidth=1, linestyle="--", label="Buy & Hold")
        ax2.set_ylabel("Equity (normalised)")
        ax2.set_xlabel("Tick")
        ax2.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def print_comparison(self) -> None:
        """Print a side-by-side comparison of RL agent vs Buy-and-Hold."""
        print(self.metrics.summary())
        print(f"  Buy & Hold Return  : {self.benchmark_return:+.2%}")


# ── Main class ────────────────────────────────────────────────────────────────


class VbtBacktest:
    """Run a full backtest of a trained RL model against historical data.

    Parameters
    ----------
    env_class:
        A concrete subclass of :class:`trading_env.TradingEnv`
        (e.g. ``StocksEnv``).
    df:
        OHLCV DataFrame used for the backtest episode.
    window_size:
        Lookback window passed to the environment.
    frame_bound:
        ``(start, end)`` index pair defining the episode slice in *df*.
    trade_fee_bid_percent:
        Proportional bid fee (default 0.1 %).
    trade_fee_ask_percent:
        Proportional ask fee (default 0.1 %).
    periods_per_year:
        252 for daily stocks, 365 for daily crypto, 8760 for hourly crypto.
    initial_capital:
        Notional starting capital for vectorbt simulation (default 10 000).
    """

    def __init__(
        self,
        env_class: Type,
        df: pd.DataFrame,
        window_size: int,
        frame_bound: tuple[int, int],
        trade_fee_bid_percent: float = 0.001,
        trade_fee_ask_percent: float = 0.001,
        periods_per_year: int = 252,
        initial_capital: float = 10_000.0,
    ) -> None:
        self.env_class = env_class
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.trade_fee_bid_percent = trade_fee_bid_percent
        self.trade_fee_ask_percent = trade_fee_ask_percent
        self.periods_per_year = periods_per_year
        self.initial_capital = initial_capital

    # ── Public API ────────────────────────────────────────────────────

    def run(
        self,
        model: Any,
        use_vbt: bool = True,
    ) -> BacktestResult:
        """Run the backtest and return a :class:`BacktestResult`.

        Parameters
        ----------
        model:
            A trained model with a ``predict(obs, deterministic=True)``
            method (e.g. an SB3 ``PPO`` instance), **or** a plain callable
            ``model(obs) -> int`` for testing.
        use_vbt:
            If ``True`` (default) and vectorbt is installed, use vectorbt
            for portfolio simulation.  Falls back to pure-Python metrics
            automatically when vectorbt is not available.

        Returns
        -------
        BacktestResult
        """
        profit_history, position_history, prices = self._run_episode(model)

        metrics = compute_metrics(
            profit_history=profit_history,
            position_history=position_history,
            periods_per_year=self.periods_per_year,
        )

        start, end = self.frame_bound
        price_series = pd.Series(
            prices,
            name="Close",
        )
        equity_series = pd.Series(profit_history, name="equity")
        position_series = pd.Series(position_history, name="position")

        # Buy-and-Hold benchmark
        benchmark_return = float(prices[-1] / prices[0] - 1.0) if len(prices) >= 2 else 0.0

        vbt_portfolio = None
        if use_vbt and _VBT_AVAILABLE:
            vbt_portfolio = self._run_vbt(prices, position_history)

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_series,
            position_series=position_series,
            price_series=price_series,
            vbt_portfolio=vbt_portfolio,
            benchmark_return=benchmark_return,
        )

    def compare(self, model: Any) -> None:
        """Run backtest and print comparison vs Buy-and-Hold."""
        result = self.run(model)
        result.print_comparison()
        return result

    # ── Episode runner ────────────────────────────────────────────────

    def _run_episode(
        self, model: Any
    ) -> tuple[list[float], list[int], np.ndarray]:
        """Step through the environment and collect histories.

        Returns
        -------
        profit_history   list of cumulative profit ratios
        position_history list of integer positions (-1, 0, 1)
        prices           np.ndarray of close prices over the episode
        """
        env = self.env_class(
            df=self.df,
            window_size=self.window_size,
            frame_bound=self.frame_bound,
            trade_fee_bid_percent=self.trade_fee_bid_percent,
            trade_fee_ask_percent=self.trade_fee_ask_percent,
        )
        obs, _ = env.reset()

        profit_history: list[float] = [1.0]
        position_history: list[int] = [0]  # start FLAT

        while True:
            action = self._predict(model, obs)
            obs, _, terminated, truncated, info = env.step(action)
            profit_history.append(info["total_profit"])
            position_history.append(int(info["position"]))
            if terminated or truncated:
                break

        prices = env.prices[env._start_tick: env._end_tick + 1]
        return profit_history, position_history, np.asarray(prices, dtype=float)

    # ── vectorbt simulation ───────────────────────────────────────────

    def _run_vbt(
        self,
        prices: np.ndarray,
        position_history: list[int],
    ):
        """Build a vectorbt Portfolio from the position history.

        Returns the vbt Portfolio object (can be None-checked by callers).
        """
        pos = np.asarray(position_history, dtype=int)
        n = min(len(prices), len(pos))
        pos = pos[:n]
        px = prices[:n]
        price_series = pd.Series(px)

        # Long signals
        long_entries = _rising_edge(pos, target=1)
        long_exits = _falling_edge(pos, from_val=1)

        # Short signals
        short_entries = _rising_edge(pos, target=-1)
        short_exits = _falling_edge(pos, from_val=-1)

        fees = (self.trade_fee_bid_percent + self.trade_fee_ask_percent) / 2.0

        portfolio = vbt.Portfolio.from_signals(
            close=price_series,
            entries=pd.Series(long_entries),
            exits=pd.Series(long_exits),
            short_entries=pd.Series(short_entries),
            short_exits=pd.Series(short_exits),
            fees=fees,
            init_cash=self.initial_capital,
            freq="1D",
        )
        return portfolio

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _predict(model: Any, obs: np.ndarray) -> int:
        """Unified predict call: handles SB3 models and plain callables."""
        if callable(model) and not hasattr(model, "predict"):
            return int(model(obs))
        action, _ = model.predict(obs, deterministic=True)
        return int(action)


# ── Signal helpers ────────────────────────────────────────────────────────────


def _rising_edge(positions: np.ndarray, target: int) -> np.ndarray:
    """Boolean array – True where position transitions TO *target*."""
    result = np.zeros(len(positions), dtype=bool)
    for i in range(1, len(positions)):
        if positions[i] == target and positions[i - 1] != target:
            result[i] = True
    return result


def _falling_edge(positions: np.ndarray, from_val: int) -> np.ndarray:
    """Boolean array – True where position transitions AWAY FROM *from_val*."""
    result = np.zeros(len(positions), dtype=bool)
    for i in range(1, len(positions)):
        if positions[i - 1] == from_val and positions[i] != from_val:
            result[i] = True
    return result

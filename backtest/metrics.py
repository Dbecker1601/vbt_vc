"""Performance metrics for backtesting results.

All metrics are computed from a profit-ratio series (e.g. ``[1.0, 1.02,
0.98, ...]``) where 1.0 means 100 % of initial capital.  No external
dependencies beyond NumPy and pandas are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Container for all performance metrics of a single backtest run."""

    # Returns
    total_return: float       # e.g. 0.35 → +35 %
    annualized_return: float  # CAGR

    # Risk-adjusted
    sharpe_ratio: float       # annualised, risk-free rate = 0
    sortino_ratio: float      # downside deviation variant
    calmar_ratio: float       # annualised return / max drawdown

    # Drawdown
    max_drawdown: float       # e.g. 0.20 → −20 % peak-to-trough
    avg_drawdown: float

    # Trade statistics
    n_trades: int
    win_rate: float           # fraction of winning trades
    profit_factor: float      # gross profit / gross loss

    # Raw series for further analysis
    equity_curve: pd.Series = field(repr=False)
    drawdown_series: pd.Series = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            "── Backtest Metrics ─────────────────────────",
            f"  Total Return       : {self.total_return:+.2%}",
            f"  Annualised Return  : {self.annualized_return:+.2%}",
            f"  Sharpe Ratio       : {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio      : {self.sortino_ratio:.3f}",
            f"  Calmar Ratio       : {self.calmar_ratio:.3f}",
            f"  Max Drawdown       : {self.max_drawdown:.2%}",
            f"  Avg Drawdown       : {self.avg_drawdown:.2%}",
            f"  Trades             : {self.n_trades}",
            f"  Win Rate           : {self.win_rate:.1%}",
            f"  Profit Factor      : {self.profit_factor:.2f}",
            "─────────────────────────────────────────────",
        ]
        return "\n".join(lines)


def compute_metrics(
    profit_history: Sequence[float],
    position_history: Sequence[int],
    prices: Sequence[float] | None = None,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Compute all performance metrics from raw backtest output.

    Parameters
    ----------
    profit_history:
        Sequence of cumulative profit ratios (e.g. ``[1.0, 1.02, ...]``).
        ``1.0`` = start of episode.
    position_history:
        Sequence of integer positions at each step: -1 SHORT, 0 FLAT, 1 LONG.
        Must be the same length as *profit_history*.
    prices:
        Raw close prices (same length as *profit_history*).  Used only for
        trade win/loss classification; optional.
    periods_per_year:
        Number of trading periods per year.  252 for daily, 8760 for hourly
        crypto, 365 for daily crypto, etc.

    Returns
    -------
    BacktestMetrics
    """
    equity = np.asarray(profit_history, dtype=float)
    positions = np.asarray(position_history, dtype=int)

    if len(equity) < 2:
        raise ValueError("profit_history must contain at least 2 values.")
    if len(equity) != len(positions):
        raise ValueError("profit_history and position_history must have the same length.")

    equity_series = pd.Series(equity, name="equity")

    # ── Returns ────────────────────────────────────────────────────────
    step_returns = np.diff(equity) / (equity[:-1] + 1e-12)
    total_return = equity[-1] / equity[0] - 1.0
    n_periods = len(equity) - 1
    annualized_return = (equity[-1] / equity[0]) ** (periods_per_year / max(n_periods, 1)) - 1.0

    # ── Risk-adjusted metrics ──────────────────────────────────────────
    sharpe_ratio = _sharpe(step_returns, periods_per_year)
    sortino_ratio = _sortino(step_returns, periods_per_year)

    # ── Drawdown ───────────────────────────────────────────────────────
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / (running_max + 1e-12)
    max_drawdown = float(abs(drawdown.min()))
    avg_drawdown = float(abs(drawdown[drawdown < 0].mean())) if (drawdown < 0).any() else 0.0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 1e-9 else np.nan

    # ── Trade statistics ───────────────────────────────────────────────
    n_trades, win_rate, profit_factor = _trade_stats(positions, equity)

    return BacktestMetrics(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        calmar_ratio=float(calmar_ratio),
        max_drawdown=float(max_drawdown),
        avg_drawdown=float(avg_drawdown),
        n_trades=n_trades,
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        equity_curve=equity_series,
        drawdown_series=pd.Series(drawdown, name="drawdown"),
    )


# ── Private helpers ────────────────────────────────────────────────────────


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: int) -> float:
    mu = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std < 1e-12:
        return np.nan
    return float(mu / downside_std * np.sqrt(periods_per_year))


def _trade_stats(
    positions: np.ndarray,
    equity: np.ndarray,
) -> tuple[int, float, float]:
    """Extract individual trade PnL from position changes.

    A trade opens when position transitions to ±1 and closes when it
    transitions back to 0 or reverses sign.

    Returns
    -------
    n_trades, win_rate, profit_factor
    """
    trade_returns: list[float] = []
    in_trade_since: int | None = None
    entry_equity: float = 1.0

    for i in range(1, len(positions)):
        prev, curr = positions[i - 1], positions[i]

        # Open trade
        if prev == 0 and curr != 0:
            in_trade_since = i
            entry_equity = equity[i]

        # Close trade
        elif in_trade_since is not None and (curr == 0 or (prev != 0 and curr != prev)):
            pnl = equity[i] / (entry_equity + 1e-12) - 1.0
            trade_returns.append(float(pnl))
            in_trade_since = None
            # Immediately re-open if position reversed
            if curr != 0:
                in_trade_since = i
                entry_equity = equity[i]

    if not trade_returns:
        return 0, 0.0, 0.0

    n_trades = len(trade_returns)
    wins = [r for r in trade_returns if r > 0]
    losses = [abs(r) for r in trade_returns if r < 0]
    win_rate = len(wins) / n_trades
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else np.inf

    return n_trades, win_rate, float(profit_factor)

"""Tests for backtest/ – metrics and VbtBacktest.

All tests run without a real trained model, real market data, or vectorbt.
A minimal mock environment and a simple callable "model" are used throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import BacktestMetrics, compute_metrics, _sharpe, _sortino, _trade_stats
from backtest.vbt_backtest import (
    VbtBacktest,
    BacktestResult,
    _rising_edge,
    _falling_edge,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int = 60) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with a smooth upward trend."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.2, 1.0, n))
    close = np.maximum(close, 1.0)
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="D"),
    )
    return df


def _constant_profit(n: int, value: float = 1.0) -> list[float]:
    return [value] * n


def _linear_profit(n: int, start: float = 1.0, end: float = 1.5) -> list[float]:
    return list(np.linspace(start, end, n))


def _constant_positions(n: int, pos: int = 1) -> list[int]:
    return [pos] * n


class _DummyEnv:
    """Minimal environment stub used by VbtBacktest._run_episode tests."""

    def __init__(self, df, window_size, frame_bound, **kwargs):
        self._prices = df["Close"].to_numpy().astype(float)
        self._start_tick = window_size
        self._end_tick = frame_bound[1] - 1
        self._tick = self._start_tick
        self.prices = self._prices

    def reset(self, seed=None, options=None):
        self._tick = self._start_tick
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        self._tick += 1
        truncated = self._tick >= self._end_tick
        info = {"total_profit": 1.0 + self._tick * 0.001, "position": 1}
        return np.zeros(5, dtype=np.float32), 0.0, False, truncated, info


def _hold_model(obs: np.ndarray) -> int:
    """Always HOLD (action 2)."""
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_BacktestMetrics(self):
        m = compute_metrics(
            profit_history=_linear_profit(50),
            position_history=_constant_positions(50, 1),
        )
        assert isinstance(m, BacktestMetrics)

    def test_total_return_positive_trend(self):
        m = compute_metrics(
            profit_history=_linear_profit(50, 1.0, 1.5),
            position_history=_constant_positions(50, 1),
        )
        assert m.total_return == pytest.approx(0.5, rel=1e-3)

    def test_total_return_flat(self):
        m = compute_metrics(
            profit_history=_constant_profit(50, 1.0),
            position_history=_constant_positions(50, 0),
        )
        assert m.total_return == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_is_zero_for_monotone_growth(self):
        m = compute_metrics(
            profit_history=_linear_profit(50, 1.0, 2.0),
            position_history=_constant_positions(50, 1),
        )
        assert m.max_drawdown == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_detected(self):
        # equity goes up then drops significantly
        equity = [1.0, 1.2, 1.5, 1.0, 0.8, 1.1]
        m = compute_metrics(
            profit_history=equity,
            position_history=[0, 1, 1, 1, 1, 1],
        )
        # peak was 1.5, trough was 0.8 → drawdown ≥ 46%
        assert m.max_drawdown > 0.40

    def test_sharpe_positive_for_steady_growth(self):
        m = compute_metrics(
            profit_history=_linear_profit(252, 1.0, 1.3),
            position_history=_constant_positions(252, 1),
        )
        assert m.sharpe_ratio > 0

    def test_raises_on_too_short_history(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_metrics([1.0], [0])

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            compute_metrics([1.0, 1.1, 1.2], [0, 1])

    def test_equity_curve_in_result(self):
        m = compute_metrics(
            profit_history=_linear_profit(10),
            position_history=_constant_positions(10, 0),
        )
        assert isinstance(m.equity_curve, pd.Series)
        assert len(m.equity_curve) == 10

    def test_drawdown_series_in_result(self):
        m = compute_metrics(
            profit_history=_linear_profit(10),
            position_history=_constant_positions(10, 0),
        )
        assert isinstance(m.drawdown_series, pd.Series)

    def test_summary_is_string(self):
        m = compute_metrics(
            profit_history=_linear_profit(30),
            position_history=_constant_positions(30, 1),
        )
        s = m.summary()
        assert isinstance(s, str)
        assert "Total Return" in s


class TestSharpe:
    def test_zero_volatility(self):
        returns = np.zeros(100)
        assert _sharpe(returns, 252) == 0.0

    def test_positive_returns_positive_sharpe(self):
        # Needs some variance for a finite, positive Sharpe
        rng = np.random.default_rng(7)
        returns = 0.002 + rng.normal(0, 0.005, 200)  # positive mean, small noise
        assert _sharpe(returns, 252) > 0


class TestSortino:
    def test_no_downside_returns_nan(self):
        returns = np.full(50, 0.01)
        result = _sortino(returns, 252)
        assert np.isnan(result)

    def test_mixed_returns(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0005, 0.01, 200)
        result = _sortino(returns, 252)
        assert np.isfinite(result)


class TestTradeStats:
    def test_no_position_changes(self):
        pos = np.zeros(10, dtype=int)
        equity = np.ones(10)
        n, wr, pf = _trade_stats(pos, equity)
        assert n == 0
        assert wr == 0.0
        assert pf == 0.0

    def test_single_winning_trade(self):
        # Enter LONG at tick 2, exit at tick 5
        pos = np.array([0, 0, 1, 1, 1, 0, 0])
        equity = np.array([1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.06])
        n, wr, pf = _trade_stats(pos, equity)
        assert n == 1
        assert wr == 1.0
        assert pf == np.inf  # no losses

    def test_single_losing_trade(self):
        pos = np.array([0, 0, 1, 1, 1, 0, 0])
        equity = np.array([1.0, 1.0, 1.0, 0.98, 0.96, 0.94, 0.94])
        n, wr, pf = _trade_stats(pos, equity)
        assert n == 1
        assert wr == 0.0

    def test_mixed_trades(self):
        pos = np.array([0, 1, 1, 0, 1, 1, 0])
        equity = np.array([1.0, 1.0, 1.1, 1.1, 1.1, 0.9, 0.9])
        n, wr, pf = _trade_stats(pos, equity)
        assert n == 2
        assert 0 < wr < 1


# ─────────────────────────────────────────────────────────────────────────────
# Signal helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRisingEdge:
    def test_detects_entry(self):
        pos = np.array([0, 0, 1, 1, 0, 1])
        result = _rising_edge(pos, target=1)
        assert result[2] is np.bool_(True)
        assert result[5] is np.bool_(True)
        assert not result[0]
        assert not result[3]

    def test_short_entry(self):
        pos = np.array([0, -1, -1, 0])
        result = _rising_edge(pos, target=-1)
        assert result[1]
        assert not result[2]


class TestFallingEdge:
    def test_detects_exit(self):
        pos = np.array([1, 1, 0, 0, 1])
        result = _falling_edge(pos, from_val=1)
        assert result[2]
        assert not result[0]
        assert not result[3]

    def test_short_exit(self):
        pos = np.array([-1, -1, 0, 0])
        result = _falling_edge(pos, from_val=-1)
        assert result[2]
        assert not result[0]


# ─────────────────────────────────────────────────────────────────────────────
# VbtBacktest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVbtBacktest:
    def _make_bt(self):
        df = _make_df(60)
        return VbtBacktest(
            env_class=_DummyEnv,
            df=df,
            window_size=5,
            frame_bound=(5, 60),
            trade_fee_bid_percent=0.001,
            trade_fee_ask_percent=0.001,
            periods_per_year=365,
        )

    def test_run_returns_BacktestResult(self):
        bt = self._make_bt()
        result = bt.run(_hold_model, use_vbt=False)
        assert isinstance(result, BacktestResult)

    def test_run_metrics_populated(self):
        bt = self._make_bt()
        result = bt.run(_hold_model, use_vbt=False)
        m = result.metrics
        assert np.isfinite(m.total_return)
        assert np.isfinite(m.sharpe_ratio)
        assert 0.0 <= m.max_drawdown <= 1.0

    def test_run_series_lengths_match(self):
        bt = self._make_bt()
        result = bt.run(_hold_model, use_vbt=False)
        assert len(result.equity_curve) == len(result.position_series)

    def test_benchmark_return_is_float(self):
        bt = self._make_bt()
        result = bt.run(_hold_model, use_vbt=False)
        assert isinstance(result.benchmark_return, float)

    def test_vbt_portfolio_none_when_disabled(self):
        bt = self._make_bt()
        result = bt.run(_hold_model, use_vbt=False)
        assert result.vbt_portfolio is None

    def test_sb3_style_model(self):
        """Model with .predict() method should also work."""
        class FakeModel:
            def predict(self, obs, deterministic=True):
                return np.array(2), None  # HOLD

        bt = self._make_bt()
        result = bt.run(FakeModel(), use_vbt=False)
        assert isinstance(result, BacktestResult)

    def test_compare_returns_result(self):
        bt = self._make_bt()
        result = bt.compare(_hold_model)
        assert isinstance(result, BacktestResult)

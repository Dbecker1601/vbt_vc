"""
Unit tests for trading_env package.

Tests cover:
- State machine (transform function)
- StocksEnv gymnasium interface (reset/step/observation/reward)
- Stable Baselines 3 check_env compatibility
- render_all saves a non-empty file
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from trading_env import Actions, Positions, StocksEnv, transform
from stable_baselines3.common.env_checker import check_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open":      close * 0.99,
            "High":      close * 1.01,
            "Low":       close * 0.98,
            "Close":     close,
            "Adj Close": close,
            "Volume":    rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


WINDOW = 10
FRAME = (WINDOW, 250)
DF = make_df()


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------

class TestStateMachine:
    """Tests for the DI-engine state machine transition function."""

    def test_flat_buy_goes_long_and_trades(self):
        pos, trade = transform(Positions.FLAT, Actions.BUY)
        assert pos == Positions.LONG
        assert trade is True

    def test_flat_sell_goes_short_and_trades(self):
        pos, trade = transform(Positions.FLAT, Actions.SELL)
        assert pos == Positions.SHORT
        assert trade is True

    def test_long_sell_goes_flat_no_trade(self):
        pos, trade = transform(Positions.LONG, Actions.SELL)
        assert pos == Positions.FLAT
        assert trade is False

    def test_short_buy_goes_flat_no_trade(self):
        pos, trade = transform(Positions.SHORT, Actions.BUY)
        assert pos == Positions.FLAT
        assert trade is False

    def test_long_double_sell_goes_short(self):
        pos, trade = transform(Positions.LONG, Actions.DOUBLE_SELL)
        assert pos == Positions.SHORT
        assert trade is True

    def test_flat_double_sell_goes_short(self):
        pos, trade = transform(Positions.FLAT, Actions.DOUBLE_SELL)
        assert pos == Positions.SHORT
        assert trade is True

    def test_short_double_buy_goes_long(self):
        pos, trade = transform(Positions.SHORT, Actions.DOUBLE_BUY)
        assert pos == Positions.LONG
        assert trade is True

    def test_flat_double_buy_goes_long(self):
        pos, trade = transform(Positions.FLAT, Actions.DOUBLE_BUY)
        assert pos == Positions.LONG
        assert trade is True

    def test_hold_does_nothing(self):
        for pos in Positions:
            new_pos, trade = transform(pos, Actions.HOLD)
            assert new_pos == pos
            assert trade is False

    def test_long_buy_no_change(self):
        pos, trade = transform(Positions.LONG, Actions.BUY)
        assert pos == Positions.LONG
        assert trade is False

    def test_short_sell_no_change(self):
        pos, trade = transform(Positions.SHORT, Actions.SELL)
        assert pos == Positions.SHORT
        assert trade is False


# ---------------------------------------------------------------------------
# Environment interface tests
# ---------------------------------------------------------------------------

class TestStocksEnv:
    """Tests for the StocksEnv gymnasium interface."""

    def test_observation_space_shape(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        expected = WINDOW * 3 + 2  # 3 features + position + tick_ratio
        assert env.observation_space.shape == (expected,)

    def test_observation_space_dtype(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        assert env.observation_space.dtype == np.float32

    def test_action_space_size(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        assert env.action_space.n == len(Actions)  # 5

    def test_reset_returns_obs_and_info(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_position_is_flat(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        assert env._position == Positions.FLAT

    def test_step_returns_correct_tuple(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        obs, rew, term, trunc, info = env.step(Actions.BUY)
        assert obs.shape == env.observation_space.shape
        assert isinstance(rew, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

    def test_full_episode_terminates(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
        assert steps > 0

    def test_total_profit_equals_exp_total_reward(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
        assert np.isclose(env._total_profit, np.exp(env._total_reward), rtol=1e-5)

    def test_profit_history_length(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
        # profit_history starts with [1.0] at reset, then grows by 1 per step
        assert len(env._profit_history) == steps + 1

    def test_max_possible_profit_positive(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
        assert env.max_possible_profit() > 0

    def test_sb3_check_env(self):
        """Stable Baselines 3 compatibility check."""
        env = StocksEnv(DF, WINDOW, FRAME)
        check_env(env, warn=True)

    def test_multiple_resets(self):
        env = StocksEnv(DF, WINDOW, FRAME)
        for _ in range(3):
            obs, _ = env.reset()
            assert obs.shape == env.observation_space.shape

    def test_reward_zero_on_hold(self):
        """HOLD from FLAT should produce zero reward (no position change)."""
        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        _, rew, _, _, _ = env.step(Actions.HOLD)
        assert rew == 0.0


# ---------------------------------------------------------------------------
# Visualisation test
# ---------------------------------------------------------------------------

class TestRenderAll:
    """Tests for the AminHP-style render_all() method."""

    def test_render_all_saves_file(self):
        import matplotlib
        matplotlib.use("Agg")

        env = StocksEnv(DF, WINDOW, FRAME)
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            env.render_all(title="Test", save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000, "Expected a non-trivial PNG file"
        finally:
            os.unlink(path)
            env.close()

"""
Base trading environment that combines:
- DI-engine's state machine logic (3 positions, 5 actions, logarithmic reward)
  Source: https://github.com/opendilab/DI-engine/tree/main/dizoo/gym_anytrading/envs
- AminHP's result visualization (render_all with position-colored markers, profit history)
  Source: https://github.com/AminHP/gym-anytrading
- Stable Baselines 3 compatible gymnasium interface
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


# ---------------------------------------------------------------------------
# Enums – DI-engine style (3 positions, 5 actions)
# ---------------------------------------------------------------------------

class Actions(int, Enum):
    """Five trading actions as defined in the DI-engine state machine."""
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


class Positions(int, Enum):
    """Three trading positions as defined in the DI-engine state machine."""
    SHORT = -1
    FLAT = 0
    LONG = 1


# ---------------------------------------------------------------------------
# State machine transition – DI-engine logic
# ---------------------------------------------------------------------------

def transform(position: Positions, action: int) -> Tuple[Positions, bool]:
    """
    Transition function for the DI-engine state machine.

    Args:
        position: Current trading position (SHORT, FLAT, or LONG).
        action:   Action chosen by the agent.

    Returns:
        (next_position, trade_executed)  – trade_executed is True when a
        position change results in a profit/loss event being recorded.
    """
    if action == Actions.SELL:
        if position == Positions.LONG:
            return Positions.FLAT, False
        if position == Positions.FLAT:
            return Positions.SHORT, True

    if action == Actions.BUY:
        if position == Positions.SHORT:
            return Positions.FLAT, False
        if position == Positions.FLAT:
            return Positions.LONG, True

    if action == Actions.DOUBLE_SELL and position in (Positions.LONG, Positions.FLAT):
        return Positions.SHORT, True

    if action == Actions.DOUBLE_BUY and position in (Positions.SHORT, Positions.FLAT):
        return Positions.LONG, True

    # HOLD or no valid transition
    return position, False


# ---------------------------------------------------------------------------
# Base TradingEnv
# ---------------------------------------------------------------------------

class TradingEnv(gym.Env):
    """
    Abstract base class for trading environments.

    Implements the DI-engine state machine (SHORT/FLAT/LONG, 5 actions) and
    logarithmic reward function while providing an AminHP-compatible
    ``render_all()`` method for result visualization.

    The environment exposes a standard gymnasium interface so it can be used
    directly with Stable Baselines 3.

    Observation vector (flat):
        [window_size × feature_dim]  +  [position_value]  +  [tick_ratio]

    Subclasses must implement:
        - ``_process_data()``    → (prices, signal_features, feature_dim)
        - ``_calculate_reward()``
        - ``max_possible_profit()``
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(
        self,
        df,
        window_size: int,
        frame_bound: Tuple[int, int],
        render_mode: Optional[str] = None,
        trade_fee_bid_percent: float = 0.01,
        trade_fee_ask_percent: float = 0.005,
    ):
        """
        Args:
            df:                    pandas DataFrame with OHLCV data.
            window_size:           Number of past ticks visible to the agent.
            frame_bound:           (start_index, end_index) slice of ``df``.
            render_mode:           'human' for live plotting, None otherwise.
            trade_fee_bid_percent: Transaction cost on the bid side.
            trade_fee_ask_percent: Transaction cost on the ask side.
        """
        assert df.ndim == 2, "df must be a 2-D DataFrame"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert len(frame_bound) == 2, "frame_bound must be (start, end)"

        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.render_mode = render_mode
        self.trade_fee_bid_percent = trade_fee_bid_percent
        self.trade_fee_ask_percent = trade_fee_ask_percent

        self.prices, self.signal_features, self._feature_dim = self._process_data()

        # Observation: flattened window + position value + tick ratio
        obs_shape = (window_size * self._feature_dim + 2,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(Actions))

        # Episode state (initialised in reset)
        self._start_tick: int = self.window_size
        self._end_tick: int = len(self.prices) - 1
        self._current_tick: Optional[int] = None
        self._last_trade_tick: Optional[int] = None
        self._position: Optional[Positions] = None
        self._position_history: list = []
        self._profit_history: list = []
        self._total_reward: float = 0.0
        self._total_profit: float = 1.0
        self._truncated: bool = False
        self.history: dict = {}

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.FLAT
        self._position_history = [self._position] * self.window_size + [self._position]
        self._profit_history = [1.0]
        self._total_reward = 0.0
        self._total_profit = 1.0
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: int):
        self._truncated = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._position, trade = transform(self._position, action)
        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        self._total_profit = float(np.exp(self._total_reward))
        self._profit_history.append(self._total_profit)

        observation = self._get_observation()
        info = self._get_info()

        if self._truncated:
            info["max_possible_profit"] = self.max_possible_profit()

        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def render(self, mode="human"):
        if self.render_mode == "rgb_array":
            return self._render_to_array()
        self._render_frame()

    def close(self):
        plt.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Flat observation: window features + position value + tick ratio."""
        window = self.signal_features[
            (self._current_tick - self.window_size + 1): self._current_tick + 1
        ].flatten()
        eps_length = self._end_tick - self._start_tick
        tick_ratio = (self._current_tick - self._last_trade_tick) / max(eps_length, 1)
        return np.concatenate(
            [window, [float(self._position.value)], [tick_ratio]]
        ).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "position": self._position.value,
        }

    def _update_history(self, info: dict):
        if not self.history:
            self.history = {key: [] for key in info}
        for key, value in info.items():
            self.history.setdefault(key, []).append(value)

    def _render_frame(self):
        """Live incremental render (called each step when render_mode='human')."""
        plt.clf()
        plt.suptitle(
            f"Total Reward: {self._total_reward:.4f}  |  "
            f"Total Profit: {self._total_profit:.4f}"
        )
        self._plot_positions()
        plt.pause(1.0 / self.metadata["render_fps"])

    def _render_to_array(self) -> np.ndarray:
        fig, _ = plt.subplots()
        self._plot_positions()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    def _plot_positions(self):
        """Shared plotting logic used by render_frame and render_all."""
        eps_prices = self.prices[: self._current_tick + 1]
        window_ticks = np.arange(len(self._position_history))

        short_ticks, flat_ticks, long_ticks = [], [], []
        for i, pos in enumerate(self._position_history):
            if pos == Positions.SHORT:
                short_ticks.append(i)
            elif pos == Positions.LONG:
                long_ticks.append(i)
            else:
                flat_ticks.append(i)

        plt.plot(eps_prices, label="Close Price", linewidth=1)
        plt.plot(long_ticks,  eps_prices[long_ticks],  "g^", markersize=4, label="Long")
        plt.plot(flat_ticks,  eps_prices[flat_ticks],  "bo", markersize=2, label="Flat")
        plt.plot(short_ticks, eps_prices[short_ticks], "rv", markersize=4, label="Short")
        plt.legend(loc="upper left")
        plt.xlabel("Tick")
        plt.ylabel("Price")

    # ------------------------------------------------------------------
    # AminHP-style end-of-episode rendering
    # ------------------------------------------------------------------

    def render_all(self, title: Optional[str] = None, save_path: Optional[str] = None):
        """
        Plot the full episode result in AminHP style:
          - Top panel:    Close price with colour-coded position markers.
          - Bottom panel: Accumulated profit curve.

        Args:
            title:     Optional figure title.
            save_path: If given, the figure is saved to this file path.
        """
        fig, (ax_price, ax_profit) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=False
        )
        fig.suptitle(
            f"Total Reward: {self._total_reward:.6f}  ~  "
            f"Total Profit: {self._total_profit:.6f}",
            fontsize=12,
        )
        if title:
            ax_price.set_title(title)

        # ---- price + position markers ----
        eps_prices = self.prices[self._start_tick: self._end_tick + 1]
        window_ticks = np.arange(len(self._position_history))

        short_ticks, flat_ticks, long_ticks = [], [], []
        for i, pos in enumerate(self._position_history):
            if pos == Positions.SHORT:
                short_ticks.append(i)
            elif pos == Positions.LONG:
                long_ticks.append(i)
            else:
                flat_ticks.append(i)

        ax_price.plot(eps_prices, linewidth=1, label="Close Price")
        if long_ticks:
            ax_price.plot(
                long_ticks, eps_prices[np.clip(long_ticks, 0, len(eps_prices) - 1)],
                "g^", markersize=4, label="Long",
            )
        if flat_ticks:
            ax_price.plot(
                flat_ticks, eps_prices[np.clip(flat_ticks, 0, len(eps_prices) - 1)],
                "bo", markersize=2, label="Flat",
            )
        if short_ticks:
            ax_price.plot(
                short_ticks, eps_prices[np.clip(short_ticks, 0, len(eps_prices) - 1)],
                "rv", markersize=4, label="Short",
            )
        ax_price.set_ylabel("Close Price")
        ax_price.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

        # ---- profit history ----
        ax_profit.plot(self._profit_history, color="purple", linewidth=1.5)
        ax_profit.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_profit.set_xlabel("Trading Days")
        ax_profit.set_ylabel("Profit Ratio")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def save_rendering(self, filepath: str):
        """Save the current matplotlib figure to *filepath*."""
        plt.savefig(filepath, dpi=150, bbox_inches="tight")

    def pause_rendering(self):
        """Block execution until the user closes the plot window."""
        plt.show()

    # ------------------------------------------------------------------
    # Abstract methods – implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _process_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process raw DataFrame into price array and feature matrix.

        Returns:
            prices          (N,)    – close prices for the episode frame
            signal_features (N, D)  – normalised feature matrix
            feature_dim     int     – D (number of feature columns)
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self, action: int) -> float:
        """Compute the step reward for the given action."""
        raise NotImplementedError

    @abstractmethod
    def max_possible_profit(self) -> float:
        """Return the theoretical maximum profit for the current episode."""
        raise NotImplementedError

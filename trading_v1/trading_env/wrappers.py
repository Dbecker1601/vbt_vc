"""
Gymnasium wrappers for reward shaping and observation enhancement.
"""

import numpy as np
import gymnasium as gym
from collections import deque


class SharpeRewardWrapper(gym.Wrapper):
    """
    Wraps a trading environment to add a Sharpe-ratio component to the reward.

    Instead of pure log-return rewards, this wrapper tracks recent returns and
    adds a risk-adjusted component:

        adjusted_reward = base_reward + sharpe_scale * (return - mean) / (std + eps)

    This encourages the agent to produce consistent returns rather than
    high-variance gambles.

    Parameters
    ----------
    env : gym.Env
        The base trading environment.
    sharpe_window : int
        Rolling window for Sharpe computation (default: 50 steps).
    sharpe_scale : float
        Weight of the Sharpe component (default: 0.1).
    """

    def __init__(self, env, sharpe_window: int = 50, sharpe_scale: float = 0.1):
        super().__init__(env)
        self.sharpe_window = sharpe_window
        self.sharpe_scale = sharpe_scale
        self._returns = deque(maxlen=sharpe_window)

    def reset(self, **kwargs):
        self._returns.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._returns.append(reward)
        info["base_reward"] = reward

        if len(self._returns) >= 2:
            returns_arr = np.array(self._returns)
            mean_r = returns_arr.mean()
            std_r = returns_arr.std()
            sharpe_component = self.sharpe_scale * (reward - mean_r) / (std_r + 1e-8)
            reward = reward + sharpe_component
            info["sharpe_component"] = sharpe_component

        return obs, reward, terminated, truncated, info


class DrawdownPenaltyWrapper(gym.Wrapper):
    """
    Penalizes the agent when cumulative profit drops below its peak (drawdown).

    penalty = -drawdown_scale * (peak_profit - current_profit) / peak_profit

    This teaches the agent to protect gains.
    """

    def __init__(self, env, drawdown_scale: float = 0.05):
        super().__init__(env)
        self.drawdown_scale = drawdown_scale
        self._peak_profit = 1.0

    def reset(self, **kwargs):
        self._peak_profit = 1.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_profit = info.get("total_profit", 1.0)
        self._peak_profit = max(self._peak_profit, current_profit)

        if current_profit < self._peak_profit:
            drawdown = (self._peak_profit - current_profit) / self._peak_profit
            penalty = -self.drawdown_scale * drawdown
            reward += penalty
            info["drawdown_penalty"] = penalty
            info["drawdown_pct"] = drawdown * 100

        return obs, reward, terminated, truncated, info

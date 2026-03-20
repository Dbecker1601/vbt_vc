"""
train.py – Train a PPO agent on StocksEnv using Stable Baselines 3,
then evaluate and visualise results using the AminHP-style render_all().

Usage
-----
    python train.py [--ticker TICKER] [--timesteps N] [--window W]
                    [--train-end YYYY-MM-DD] [--save-fig results.png]

Requirements
------------
    pip install -r requirements.txt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI / headless runs
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required: pip install yfinance")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    sys.exit("stable-baselines3 is required: pip install stable-baselines3")

from trading_env import StocksEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str = "2010-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    print(f"Downloading {ticker} from {start} to {end} …")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        sys.exit(f"No data returned for {ticker}. Check the ticker symbol.")
    # Flatten multi-level columns if present (yfinance ≥ 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"  Downloaded {len(df)} rows.")
    return df


def make_env(df: pd.DataFrame, window_size: int, frame_bound, render_mode=None):
    """Factory function that returns a (monitored) StocksEnv."""
    env = StocksEnv(
        df=df,
        window_size=window_size,
        frame_bound=frame_bound,
        render_mode=render_mode,
    )
    return Monitor(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO on StocksEnv (SB3)")
    parser.add_argument("--ticker",     default="GOOGL",        help="Yahoo Finance ticker")
    parser.add_argument("--start",      default="2010-01-01",   help="Data start date")
    parser.add_argument("--end",        default="2023-12-31",   help="Data end date")
    parser.add_argument("--train-end",  default="2021-12-31",   help="Training/test split date")
    parser.add_argument("--timesteps",  type=int, default=50000, help="SB3 training timesteps")
    parser.add_argument("--window",     type=int, default=10,    help="Observation window size")
    parser.add_argument("--save-fig",   default="results.png",  help="Path to save result figure")
    parser.add_argument("--model-path", default="ppo_stocks",   help="Path to save/load the SB3 model")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Download & split data
    # ------------------------------------------------------------------
    df = download_data(args.ticker, start=args.start, end=args.end)

    train_mask = df.index <= args.train_end
    test_mask  = df.index >  args.train_end

    df_train = df[train_mask].reset_index(drop=True)
    df_test  = df[test_mask].reset_index(drop=True)

    if len(df_train) < args.window + 2:
        sys.exit("Not enough training data. Adjust --start / --train-end.")
    if len(df_test) < args.window + 2:
        sys.exit("Not enough test data. Adjust --train-end / --end.")

    train_frame = (args.window, len(df_train))
    test_frame  = (args.window, len(df_test))

    print(f"Train rows: {len(df_train)}, frame: {train_frame}")
    print(f"Test  rows: {len(df_test)},  frame: {test_frame}")

    # ------------------------------------------------------------------
    # 2. Validate environment (SB3 checker)
    # ------------------------------------------------------------------
    print("\nRunning SB3 environment check …")
    check_env(
        StocksEnv(df=df_train, window_size=args.window, frame_bound=train_frame),
        warn=True,
    )
    print("  Environment check passed.")

    # ------------------------------------------------------------------
    # 3. Train with PPO
    # ------------------------------------------------------------------
    print(f"\nTraining PPO for {args.timesteps:,} timesteps …")
    train_env = DummyVecEnv([lambda: make_env(df_train, args.window, train_frame)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        tensorboard_log="./tb_logs/",
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    print(f"  Model saved to {args.model_path}.zip")

    # ------------------------------------------------------------------
    # 4. Evaluate on test data
    # ------------------------------------------------------------------
    print("\nEvaluating on test data …")
    test_env = StocksEnv(df=df_test, window_size=args.window, frame_bound=test_frame)

    obs, info = test_env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    print(f"  Total reward : {total_reward:.6f}")
    print(f"  Total profit : {test_env._total_profit:.6f}")
    print(f"  Max possible : {test_env.max_possible_profit():.6f}")

    # ------------------------------------------------------------------
    # 5. Visualise results (AminHP-style render_all)
    # ------------------------------------------------------------------
    print(f"\nSaving result figure to {args.save_fig} …")
    test_env.render_all(
        title=f"{args.ticker} – PPO Test Episode",
        save_path=args.save_fig,
    )
    print("Done.")


if __name__ == "__main__":
    main()

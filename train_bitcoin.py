"""
train_bitcoin.py – End-to-end training script for the Bitcoin deep-RL agent.

Pipeline
--------
1. Download historical OHLCV from any CCXT exchange (default: Binance BTC/USDT).
2. Split data into train / test sets.
3. Validate environment with SB3 check_env.
4. Train a PPO agent with TensorBoard logging.
5. Evaluate the trained agent on the test set.
6. Render and save the result chart.

Usage examples
--------------
# Basic – fetch 2 years of hourly BTC/USDT from Binance and train:
    python train_bitcoin.py

# Custom exchange, symbol, timeframe:
    python train_bitcoin.py --exchange bybit --symbol BTC/USDT --timeframe 4h

# Resume from a saved model:
    python train_bitcoin.py --model-path ppo_bitcoin --timesteps 100000

# Use local CSV instead of CCXT:
    python train_bitcoin.py --csv-path data/btc_1h.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from trading_env.bitcoin_env import BitcoinEnv


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data_from_ccxt(
    exchange: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download OHLCV bars from a CCXT exchange and return a cleaned DataFrame."""
    from data.ccxt_fetcher import fetch_ohlcv
    import ccxt

    print(f"[data] Fetching {symbol} {timeframe} from {exchange} "
          f"({start} -> {end}) ...")

    exchange_obj = getattr(ccxt, exchange)()
    since = int(
        pd.Timestamp(start, tz="UTC").timestamp() * 1000
    )
    until = int(
        pd.Timestamp(end, tz="UTC").timestamp() * 1000
    )

    df = fetch_ohlcv(
        exchange_id=exchange,
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        until=until,
    )
    df = df.dropna()
    print(f"[data] Downloaded {len(df)} bars  ({df.index[0]} – {df.index[-1]})")
    return df


def load_data_from_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]

    # Accept common column name variations
    rename = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
        "vol": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    assert "Close" in df.columns, (
        "CSV must contain an Open/High/Low/Close/Volume column set."
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"[data] Loaded {len(df)} bars from {path}")
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split df into train / test portions by row index."""
    split = int(len(df) * train_ratio)
    return df.iloc[:split], df.iloc[split:]


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(df: pd.DataFrame, window_size: int, frame_bound):
    """Create a monitored BitcoinEnv."""
    env = BitcoinEnv(
        df=df,
        window_size=window_size,
        frame_bound=frame_bound,
    )
    return Monitor(env)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    df_train: pd.DataFrame,
    window_size: int,
    timesteps: int,
    model_path: str,
    learning_rate: float,
    tb_log_dir: str = "./tb_logs/",
) -> PPO:
    frame_bound = (window_size, len(df_train))
    env = make_env(df_train, window_size, frame_bound)

    print("[env] Validating training environment ...")
    check_env(env, warn=True)

    print(f"[train] Observation space : {env.observation_space.shape}")
    print(f"[train] Action space      : {env.action_space.n} actions")
    print(f"[train] Training for {timesteps:,} timesteps ...")

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        learning_rate=learning_rate,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tb_log_dir,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(model_path)
    print(f"[train] Model saved -> {model_path}.zip")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: PPO,
    df_test: pd.DataFrame,
    window_size: int,
    save_fig: str,
) -> dict:
    frame_bound = (window_size, len(df_test))
    env = BitcoinEnv(df=df_test, window_size=window_size, frame_bound=frame_bound)

    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    total_profit = np.exp(total_reward)
    max_profit   = env.max_possible_profit()

    print("\n[eval] -- Test-set results -------------------------")
    print(f"       Total reward        : {total_reward:.6f}")
    print(f"       Total profit        : {total_profit:.4f}x  ({(total_profit-1)*100:.2f} %)")
    print(f"       Max possible profit : {max_profit:.4f}x")
    print(f"       Agent / max ratio   : {total_profit / max_profit:.2%}")
    print("---------------------------------------------------\n")

    env.render_all(
        title=f"BTC – PPO  |  Profit {total_profit:.4f}×  |  Max {max_profit:.4f}×",
        save_path=save_fig,
    )
    print(f"[eval] Chart saved -> {save_fig}")

    return {
        "total_reward":  total_reward,
        "total_profit":  total_profit,
        "max_profit":    max_profit,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a PPO agent on Bitcoin OHLCV data via CCXT."
    )

    # Data source
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv-path", default=None,
                     help="Path to a local CSV file (skips CCXT download).")
    src.add_argument("--exchange", default="binance",
                     help="CCXT exchange ID (default: binance).")

    p.add_argument("--symbol",    default="BTC/USDT",
                   help="Trading pair (default: BTC/USDT).")
    p.add_argument("--timeframe", default="1h",
                   help="Candle timeframe (default: 1h).")
    p.add_argument("--start",     default="2022-01-01",
                   help="Historical start date (default: 2022-01-01).")
    p.add_argument("--end",       default="2024-12-31",
                   help="Historical end date  (default: 2024-12-31).")

    # Episode / training
    p.add_argument("--window",      type=int,   default=20,
                   help="Observation window size in bars (default: 20).")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Fraction of data used for training (default: 0.8).")
    p.add_argument("--timesteps",   type=int,   default=200_000,
                   help="Total PPO training timesteps (default: 200 000).")
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="PPO learning rate (default: 3e-4).")

    # Paths
    p.add_argument("--model-path", default="ppo_bitcoin",
                   help="Filename (no extension) for saving the model.")
    p.add_argument("--save-fig",   default="bitcoin_result.png",
                   help="Output path for the result chart.")
    p.add_argument("--tb-log",     default="./tb_logs/",
                   help="TensorBoard log directory.")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ── 1. Load data ──────────────────────────────────────────────────
    if args.csv_path:
        df = load_data_from_csv(args.csv_path)
    else:
        df = load_data_from_ccxt(
            exchange=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
        )

    df = df.reset_index(drop=True)  # use integer index internally

    # ── 2. Train / test split ─────────────────────────────────────────
    df_train, df_test = split_data(df, train_ratio=args.train_ratio)
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    print(f"[split] Train: {len(df_train)} bars  |  Test: {len(df_test)} bars")

    # ── 3. Train ──────────────────────────────────────────────────────
    model = train(
        df_train=df_train,
        window_size=args.window,
        timesteps=args.timesteps,
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        tb_log_dir=args.tb_log,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────
    evaluate(
        model=model,
        df_test=df_test,
        window_size=args.window,
        save_fig=args.save_fig,
    )


if __name__ == "__main__":
    main()

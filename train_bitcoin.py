"""
train_bitcoin.py – End-to-end training script for the Bitcoin deep-RL agent.

Improvements over baseline:
  1. More timesteps (2 M default)
  2. Larger observation window (60 bars = 1 h of 1 m context)
  3. RecurrentPPO (PPO + LSTM) via sb3-contrib for sequential memory
  4. MlpLstmPolicy – LSTM inside the policy network
  5. 13 features: +EMA200, +Volume-Delta, +VWAP-dev, +hour_sin/cos
  6. Intermediate step reward (unrealised P&L signal while position is open)
  7. Higher fees (0.2 %) to force selective trading

Pipeline
--------
1. Download 3 months of 1 m BTC/USDT OHLCV from Binance (default).
2. Preserve datetime for VWAP / time-of-day features, then reset integer index.
3. Chronological train (80 %) / test (20 %) split.
4. Validate env with SB3 check_env.
5. Train RecurrentPPO with TensorBoard logging.
6. Evaluate on held-out test set (LSTM states handled correctly).
7. Render and save result chart.

Usage examples
--------------
    # Default: last 3 months, 1 m bars, 2 M steps
    python train_bitcoin.py

    # Custom timeframe
    python train_bitcoin.py --timeframe 5m --timesteps 1000000

    # Local CSV
    python train_bitcoin.py --csv-path data/btc_1m.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import RecurrentPPO
    _RECURRENT = True
except ImportError:
    from stable_baselines3 import PPO as RecurrentPPO
    _RECURRENT = False
    print("[warn] sb3-contrib not found – falling back to standard PPO (no LSTM).")
    print("       Install with: pip install sb3-contrib")

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
    """Download OHLCV from CCXT and return a cleaned DataFrame with DatetimeIndex."""
    from data.ccxt_fetcher import fetch_ohlcv
    import ccxt

    print(f"[data] Fetching {symbol} {timeframe} from {exchange} ({start} -> {end}) ...")

    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    until = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)

    df = fetch_ohlcv(
        exchange_id=exchange,
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        until=until,
    )
    df = df.dropna()
    print(f"[data] Downloaded {len(df):,} bars  ({df.index[0]} – {df.index[-1]})")
    return df


def load_data_from_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from a local CSV file."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume", "vol": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    assert "Close" in df.columns, "CSV must contain Open/High/Low/Close/Volume columns."
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"[data] Loaded {len(df):,} bars from {path}")
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve datetime info as a column (needed for VWAP / time-of-day features)
    then reset to an integer index for positional slicing inside the env.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index.name = "datetime"
        df = df.reset_index()          # 'datetime' becomes a regular column
    else:
        df = df.reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Chronological train / test split (no lookahead)."""
    split = int(len(df) * train_ratio)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(df: pd.DataFrame, window_size: int, frame_bound, fee: float):
    """Create a monitored BitcoinEnv."""
    env = BitcoinEnv(
        df=df,
        window_size=window_size,
        frame_bound=frame_bound,
        trade_fee_bid_percent=fee,
        trade_fee_ask_percent=fee,
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
    fee: float,
    tb_log_dir: str = "./tb_logs/",
):
    frame_bound = (window_size, len(df_train))
    env = make_env(df_train, window_size, frame_bound, fee)

    print("[env] Validating training environment ...")
    check_env(env, warn=True)

    algo_name = "RecurrentPPO (LSTM)" if _RECURRENT else "PPO"
    print(f"[train] Algorithm         : {algo_name}")
    print(f"[train] Observation space : {env.observation_space.shape}")
    print(f"[train] Action space      : {env.action_space.n} actions")
    print(f"[train] Training for {timesteps:,} timesteps ...")

    model = RecurrentPPO(
        "MlpLstmPolicy" if _RECURRENT else "MlpPolicy",
        env,
        n_steps=512,            # shorter rollout for LSTM stability
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
    model,
    df_test: pd.DataFrame,
    window_size: int,
    fee: float,
    save_fig: str,
) -> dict:
    frame_bound = (window_size, len(df_test))
    env = BitcoinEnv(
        df=df_test,
        window_size=window_size,
        frame_bound=frame_bound,
        trade_fee_bid_percent=fee,
        trade_fee_ask_percent=fee,
    )

    obs, _     = env.reset()
    total_reward = 0.0
    done         = False

    # Handle LSTM hidden states (RecurrentPPO) vs. standard PPO
    if _RECURRENT:
        lstm_states    = None
        episode_starts = np.ones((1,), dtype=bool)
        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            episode_starts = np.array([done])
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
    else:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
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
        title=f"BTC – RecurrentPPO  |  Profit {total_profit:.4f}×  |  Max {max_profit:.4f}×",
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
        description="Train a RecurrentPPO (LSTM) agent on Bitcoin OHLCV data."
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv-path",  default=None,
                     help="Path to a local CSV file (skips CCXT download).")
    src.add_argument("--exchange",  default="binance",
                     help="CCXT exchange ID (default: binance).")

    p.add_argument("--symbol",    default="BTC/USDT")
    p.add_argument("--timeframe", default="1m",
                   help="Candle timeframe (default: 1m).")
    p.add_argument("--start",     default="2025-12-15",
                   help="Start date – 3 months of 1 m data (default: 2025-12-15).")
    p.add_argument("--end",       default="2026-03-15",
                   help="End date (default: 2026-03-15).")

    p.add_argument("--window",        type=int,   default=60,
                   help="Observation window in bars (default: 60 = 1 h for 1 m bars).")
    p.add_argument("--train-ratio",   type=float, default=0.8,
                   help="Train fraction (default: 0.8).")
    p.add_argument("--timesteps",     type=int,   default=2_000_000,
                   help="Total training timesteps (default: 2 000 000).")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--fee",           type=float, default=0.002,
                   help="Taker fee per side (default: 0.002 = 0.2 %%).")

    p.add_argument("--model-path", default="ppo_bitcoin")
    p.add_argument("--save-fig",   default="bitcoin_result.png")
    p.add_argument("--tb-log",     default="./tb_logs/")

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

    # ── 2. Preserve datetime, reset integer index ─────────────────────
    df = prepare_df(df)

    # ── 3. Train / test split (chronological) ─────────────────────────
    df_train, df_test = split_data(df, train_ratio=args.train_ratio)
    print(f"[split] Train : {len(df_train):,} bars  ({args.train_ratio*100:.0f} %)")
    print(f"[split] Test  : {len(df_test):,} bars  ({(1-args.train_ratio)*100:.0f} %)")

    # ── 4. Train ──────────────────────────────────────────────────────
    model = train(
        df_train=df_train,
        window_size=args.window,
        timesteps=args.timesteps,
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        fee=args.fee,
        tb_log_dir=args.tb_log,
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────
    evaluate(
        model=model,
        df_test=df_test,
        window_size=args.window,
        fee=args.fee,
        save_fig=args.save_fig,
    )


if __name__ == "__main__":
    main()

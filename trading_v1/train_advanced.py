"""
train_advanced.py – Unified advanced training for Stocks & Bitcoin environments.

Improvements over baseline:
  1. Callbacks: EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
  2. Learning rate scheduling: linear / cosine decay
  3. Custom networks: LargeMLP with residual connections, Attention-based extractor
  4. VecNormalize: automatic observation normalization
  5. Better hyperparameters: tuned clip range, GAE lambda, n_epochs
  6. Reward shaping: Sharpe-ratio wrapper + drawdown penalty
  7. Multi-algorithm: PPO, A2C, DQN

Usage
-----
    # Stocks with PPO (default)
    python train_advanced.py --env stocks --ticker GOOGL --timesteps 200000

    # Bitcoin with A2C + attention network
    python train_advanced.py --env bitcoin --algo a2c --network attention --timesteps 500000

    # Bitcoin with DQN
    python train_advanced.py --env bitcoin --algo dqn --timesteps 300000

    # Full featured
    python train_advanced.py --env bitcoin --algo ppo --network attention \\
        --timesteps 1000000 --sharpe --drawdown --lr-schedule cosine
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        StopTrainingOnNoModelImprovement,
        CallbackList,
    )
    from stable_baselines3.common.evaluation import evaluate_policy
except ImportError:
    sys.exit("stable-baselines3 is required: pip install stable-baselines3")

from trading_env import StocksEnv
from trading_env.bitcoin_env import BitcoinEnv
from trading_env.wrappers import SharpeRewardWrapper, DrawdownPenaltyWrapper
from trading_env.custom_networks import LargeMLPExtractor, AttentionExtractor


# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------

def linear_schedule(initial_lr: float) -> Callable[[float], float]:
    """Linear decay from initial_lr to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return func


def cosine_schedule(initial_lr: float) -> Callable[[float], float]:
    """Cosine annealing from initial_lr to ~0."""
    def func(progress_remaining: float) -> float:
        return initial_lr * 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
    return func


def warmup_cosine_schedule(initial_lr: float, warmup_frac: float = 0.05) -> Callable[[float], float]:
    """Linear warmup then cosine decay."""
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress < warmup_frac:
            return initial_lr * (progress / warmup_frac)
        adjusted = (progress - warmup_frac) / (1.0 - warmup_frac)
        return initial_lr * 0.5 * (1.0 + np.cos(np.pi * adjusted))
    return func


LR_SCHEDULES = {
    "constant": lambda lr: lr,
    "linear": linear_schedule,
    "cosine": cosine_schedule,
    "warmup_cosine": warmup_cosine_schedule,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stocks_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance required: pip install yfinance")

    print(f"[data] Downloading {ticker} ({start} -> {end}) ...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        sys.exit(f"No data for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"[data] {len(df)} rows downloaded.")
    return df


def load_bitcoin_data(csv_path=None, exchange="binance", symbol="BTC/USDT",
                      timeframe="1m", start="2025-12-15", end="2026-03-15") -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.columns = [c.strip() for c in df.columns]
        rename = {"open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "volume": "Volume", "vol": "Volume"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        print(f"[data] Loaded {len(df)} bars from {csv_path}")
        return df
    else:
        from data.ccxt_fetcher import fetch_ohlcv
        print(f"[data] Fetching {symbol} {timeframe} from {exchange} ...")
        since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        until = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        df = fetch_ohlcv(exchange_id=exchange, symbol=symbol,
                         timeframe=timeframe, since=since, until=until)
        df = df.dropna()
        print(f"[data] {len(df):,} bars downloaded.")
        return df


def prepare_bitcoin_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index.name = "datetime"
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(env_type: str, df: pd.DataFrame, window_size: int,
             frame_bound: tuple, fee_bid: float, fee_ask: float,
             use_sharpe: bool = False, use_drawdown: bool = False,
             monitor: bool = True):
    """Create a trading environment with optional wrappers."""

    if env_type == "stocks":
        env = StocksEnv(
            df=df, window_size=window_size, frame_bound=frame_bound,
            trade_fee_bid_percent=fee_bid, trade_fee_ask_percent=fee_ask,
        )
    else:
        env = BitcoinEnv(
            df=df, window_size=window_size, frame_bound=frame_bound,
            trade_fee_bid_percent=fee_bid, trade_fee_ask_percent=fee_ask,
        )

    if use_sharpe:
        env = SharpeRewardWrapper(env, sharpe_window=50, sharpe_scale=0.1)

    if use_drawdown:
        env = DrawdownPenaltyWrapper(env, drawdown_scale=0.05)

    if monitor:
        env = Monitor(env)

    return env


# ---------------------------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------------------------

ALGO_MAP = {"ppo": PPO, "a2c": A2C, "dqn": DQN}


def build_model(algo_name: str, env, lr_schedule, network: str,
                window_size: int, tb_log: str):
    """Build an SB3 model with tuned hyperparameters and custom network."""

    AlgoClass = ALGO_MAP[algo_name]

    # Policy kwargs for custom feature extractor
    policy_kwargs = {}
    if network == "large_mlp":
        policy_kwargs["features_extractor_class"] = LargeMLPExtractor
        policy_kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        policy_kwargs["net_arch"] = dict(pi=[128, 64], vf=[128, 64])
    elif network == "attention":
        policy_kwargs["features_extractor_class"] = AttentionExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": 128,
            "window_size": window_size,
            "n_heads": 4,
        }
        policy_kwargs["net_arch"] = dict(pi=[128, 64], vf=[128, 64])
    else:
        # Default MLP but with better architecture
        policy_kwargs["net_arch"] = dict(pi=[256, 128], vf=[256, 128])

    if algo_name == "ppo":
        model = AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log,
        )
    elif algo_name == "a2c":
        model = AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_log,
        )
    elif algo_name == "dqn":
        # DQN doesn't support custom net_arch the same way
        dqn_kwargs = {}
        if network in ("large_mlp", "attention"):
            dqn_kwargs["policy_kwargs"] = {
                "features_extractor_class": policy_kwargs.get("features_extractor_class"),
                "features_extractor_kwargs": policy_kwargs.get("features_extractor_kwargs", {}),
                "net_arch": [256, 128],
            }
        else:
            dqn_kwargs["policy_kwargs"] = {"net_arch": [256, 128]}

        model = AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=tb_log,
            **dqn_kwargs,
        )

    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(eval_env, checkpoint_dir: str, eval_freq: int = 5000,
                    patience: int = 10):
    """Build a callback list with eval, checkpoint, and early stopping."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=patience,
        min_evals=5,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="checkpoint",
        verbose=0,
    )

    return CallbackList([eval_callback, checkpoint_callback])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(model, env_type: str, df_test: pd.DataFrame, window_size: int,
                   fee_bid: float, fee_ask: float, save_fig: str,
                   algo_name: str, is_recurrent: bool = False):
    """Evaluate trained model on test set and render results."""

    frame_bound = (window_size, len(df_test))

    if env_type == "stocks":
        env = StocksEnv(df=df_test, window_size=window_size, frame_bound=frame_bound,
                        trade_fee_bid_percent=fee_bid, trade_fee_ask_percent=fee_ask)
    else:
        env = BitcoinEnv(df=df_test, window_size=window_size, frame_bound=frame_bound,
                         trade_fee_bid_percent=fee_bid, trade_fee_ask_percent=fee_ask)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += reward
        actions_taken[action_int] = actions_taken.get(action_int, 0) + 1
        done = terminated or truncated

    total_profit = env._total_profit
    max_profit = env.max_possible_profit()

    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS ({algo_name.upper()})")
    print("=" * 60)
    print(f"  Total reward        : {total_reward:.6f}")
    print(f"  Total profit        : {total_profit:.4f}x  ({(total_profit - 1) * 100:.2f}%)")
    print(f"  Max possible profit : {max_profit:.4f}x")
    print(f"  Agent / max ratio   : {total_profit / max_profit:.2%}")
    print(f"  Action distribution :")
    action_names = ["DOUBLE_SELL", "SELL", "HOLD", "BUY", "DOUBLE_BUY"]
    total_actions = sum(actions_taken.values())
    for i, name in enumerate(action_names):
        count = actions_taken.get(i, 0)
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"    {name:>12s}: {count:5d} ({pct:5.1f}%)")
    print("=" * 60)

    asset = "BTC" if env_type == "bitcoin" else env_type.upper()
    env.render_all(
        title=f"{asset} – {algo_name.upper()}  |  Profit {total_profit:.4f}x  |  Max {max_profit:.4f}x",
        save_path=save_fig,
    )
    print(f"[eval] Chart saved -> {save_fig}")

    return {
        "total_reward": total_reward,
        "total_profit": total_profit,
        "max_profit": max_profit,
        "actions": actions_taken,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Advanced RL trading trainer (PPO / A2C / DQN)"
    )

    # Environment
    p.add_argument("--env", choices=["stocks", "bitcoin"], default="stocks",
                   help="Environment type")
    p.add_argument("--ticker", default="GOOGL", help="Yahoo Finance ticker (stocks)")
    p.add_argument("--csv-path", default=None, help="CSV path (bitcoin)")
    p.add_argument("--exchange", default="binance", help="CCXT exchange (bitcoin)")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2023-12-31")
    p.add_argument("--train-split", default="0.8",
                   help="Train ratio (float) or split date (YYYY-MM-DD)")

    # Training
    p.add_argument("--algo", choices=["ppo", "a2c", "dqn"], default="ppo")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=list(LR_SCHEDULES.keys()),
                   default="cosine", help="Learning rate schedule")
    p.add_argument("--fee-bid", type=float, default=None,
                   help="Bid fee (default: 0.01 stocks, 0.002 bitcoin)")
    p.add_argument("--fee-ask", type=float, default=None,
                   help="Ask fee (default: 0.005 stocks, 0.002 bitcoin)")

    # Network
    p.add_argument("--network", choices=["default", "large_mlp", "attention"],
                   default="large_mlp", help="Feature extractor architecture")

    # Reward shaping
    p.add_argument("--sharpe", action="store_true",
                   help="Enable Sharpe-ratio reward shaping")
    p.add_argument("--drawdown", action="store_true",
                   help="Enable drawdown penalty")

    # Callbacks
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience (eval rounds)")
    p.add_argument("--eval-freq", type=int, default=5000,
                   help="Evaluation frequency (timesteps)")
    p.add_argument("--no-vecnorm", action="store_true",
                   help="Disable VecNormalize")

    # Output
    p.add_argument("--model-path", default=None)
    p.add_argument("--save-fig", default=None)
    p.add_argument("--checkpoint-dir", default="./checkpoints/")
    p.add_argument("--tb-log", default="./tb_logs/")

    args = p.parse_args(argv)

    # Defaults based on env type
    if args.fee_bid is None:
        args.fee_bid = 0.002 if args.env == "bitcoin" else 0.01
    if args.fee_ask is None:
        args.fee_ask = 0.002 if args.env == "bitcoin" else 0.005
    if args.model_path is None:
        args.model_path = f"{args.algo}_{args.env}"
    if args.save_fig is None:
        args.save_fig = f"{args.algo}_{args.env}_result.png"
    if args.env == "bitcoin" and args.window == 10:
        args.window = 60  # better default for bitcoin
    if args.env == "bitcoin" and args.start == "2010-01-01":
        args.start = "2025-12-15"
        args.end = "2026-03-15"

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 60)
    print("  ADVANCED RL TRADING TRAINER")
    print("=" * 60)
    print(f"  Environment  : {args.env}")
    print(f"  Algorithm    : {args.algo.upper()}")
    print(f"  Network      : {args.network}")
    print(f"  LR Schedule  : {args.lr_schedule} (initial: {args.lr})")
    print(f"  Timesteps    : {args.timesteps:,}")
    print(f"  Window       : {args.window}")
    print(f"  Sharpe       : {'ON' if args.sharpe else 'OFF'}")
    print(f"  Drawdown     : {'ON' if args.drawdown else 'OFF'}")
    print(f"  VecNormalize : {'OFF' if args.no_vecnorm else 'ON'}")
    print("=" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────
    if args.env == "stocks":
        df = load_stocks_data(args.ticker, args.start, args.end)
    else:
        df = load_bitcoin_data(csv_path=args.csv_path, exchange=args.exchange,
                               symbol=args.symbol, timeframe=args.timeframe,
                               start=args.start, end=args.end)
        df = prepare_bitcoin_df(df)

    # ── 2. Train / test split ─────────────────────────────────────────
    try:
        train_ratio = float(args.train_split)
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_test = df.iloc[split_idx:].reset_index(drop=True)
    except ValueError:
        # Date-based split for stocks
        train_mask = df.index <= args.train_split
        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[~train_mask].reset_index(drop=True)

    if len(df_train) < args.window + 2 or len(df_test) < args.window + 2:
        sys.exit("Not enough data. Adjust split or date range.")

    train_frame = (args.window, len(df_train))
    test_frame = (args.window, len(df_test))

    print(f"[split] Train: {len(df_train):,} rows, frame: {train_frame}")
    print(f"[split] Test : {len(df_test):,} rows, frame: {test_frame}")

    # ── 3. Validate environment ───────────────────────────────────────
    print("\n[env] Running SB3 environment check ...")
    check_env_instance = make_env(
        args.env, df_train, args.window, train_frame,
        args.fee_bid, args.fee_ask, monitor=False,
    )
    check_env(check_env_instance, warn=True)
    print("[env] Check passed.")

    # ── 4. Create training env with VecNormalize ──────────────────────
    def _make_train_env():
        return make_env(
            args.env, df_train, args.window, train_frame,
            args.fee_bid, args.fee_ask,
            use_sharpe=args.sharpe, use_drawdown=args.drawdown,
        )

    train_vec_env = DummyVecEnv([_make_train_env])

    if not args.no_vecnorm:
        train_vec_env = VecNormalize(
            train_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
        print("[env] VecNormalize enabled (obs + reward)")

    # ── 5. Create eval env ────────────────────────────────────────────
    def _make_eval_env():
        return make_env(
            args.env, df_test, args.window, test_frame,
            args.fee_bid, args.fee_ask, monitor=True,
        )

    eval_vec_env = DummyVecEnv([_make_eval_env])
    if not args.no_vecnorm:
        eval_vec_env = VecNormalize(
            eval_vec_env, norm_obs=True, norm_reward=False,
            clip_obs=10.0, gamma=0.99, training=False,
        )

    # ── 6. Build model ────────────────────────────────────────────────
    lr_schedule = LR_SCHEDULES[args.lr_schedule](args.lr)
    model = build_model(
        args.algo, train_vec_env, lr_schedule,
        args.network, args.window, args.tb_log,
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"[model] Parameters: {param_count:,}")

    # ── 7. Build callbacks ────────────────────────────────────────────
    callbacks = build_callbacks(
        eval_vec_env,
        checkpoint_dir=args.checkpoint_dir,
        eval_freq=args.eval_freq,
        patience=args.patience,
    )

    # ── 8. Train ──────────────────────────────────────────────────────
    print(f"\n[train] Starting {args.algo.upper()} training for {args.timesteps:,} timesteps ...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    model.save(args.model_path)
    print(f"[train] Model saved -> {args.model_path}.zip")

    if not args.no_vecnorm:
        train_vec_env.save(f"{args.model_path}_vecnorm.pkl")
        print(f"[train] VecNormalize stats saved -> {args.model_path}_vecnorm.pkl")

    # ── 9. Load best model from eval ──────────────────────────────────
    best_path = Path(args.checkpoint_dir) / "best_model.zip"
    if best_path.exists():
        print(f"[eval] Loading best model from {best_path}")
        AlgoClass = ALGO_MAP[args.algo]
        model = AlgoClass.load(best_path)
    else:
        print("[eval] No best model found, using final model.")

    # ── 10. Final evaluation ──────────────────────────────────────────
    results = run_evaluation(
        model, args.env, df_test, args.window,
        args.fee_bid, args.fee_ask, args.save_fig, args.algo,
    )

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()

# vbt_vc – Stock Trading RL Environment

A reinforcement learning trading environment that combines:

- **DI-engine logic** (state machine, 3 positions, 5 actions, logarithmic reward)  
  Source: [opendilab/DI-engine – dizoo/gym_anytrading](https://github.com/opendilab/DI-engine/tree/main/dizoo/gym_anytrading/envs)
- **AminHP-style visualisation** (`render_all` with colour-coded position markers, profit history)  
  Source: [AminHP/gym-anytrading](https://github.com/AminHP/gym-anytrading)
- **Stable Baselines 3 compatibility** (standard `gymnasium` interface, `check_env` validated)

---

## Architecture

```
trading_env/
  __init__.py        # package exports
  trading_env.py     # abstract base – DI-engine state machine + AminHP render_all
  stocks_env.py      # concrete stocks env (3 features, z-score normalisation, log reward)
train.py             # end-to-end PPO training + evaluation + visualisation script
tests/
  test_trading_env.py
requirements.txt
```

---

## State Machine (DI-engine)

Three positions: **SHORT** (-1), **FLAT** (0), **LONG** (+1)  
Five actions: **DOUBLE_SELL** (0), **SELL** (1), **HOLD** (2), **BUY** (3), **DOUBLE_BUY** (4)

| Current → Action | DOUBLE_SELL | SELL | HOLD | BUY | DOUBLE_BUY |
|---|---|---|---|---|---|
| SHORT | – | – | SHORT | FLAT | LONG |
| FLAT | SHORT | SHORT | FLAT | LONG | LONG |
| LONG | SHORT | FLAT | LONG | – | – |

### Reward Function

Reward is non-zero only when a trade cycle closes:

- **Long cycle** (FLAT → LONG → FLAT): `log(close_curr / close_prev) + log(cost)`
- **Short cycle** (FLAT → SHORT → FLAT): `log(2 − close_curr / close_prev) + log(cost)`

where `cost = (1 − bid_fee) × (1 − ask_fee)`.

---

## Observation Space

```
[window_size × 3 features (Close, Diff, Volume – z-score normalised)]
+ [position value: −1 / 0 / 1]
+ [tick ratio: ticks since last trade / episode length]
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```python
import pandas as pd
from trading_env import StocksEnv

df = pd.read_csv("STOCKS_GOOGL.csv", index_col="Date", parse_dates=True)
env = StocksEnv(df=df, window_size=10, frame_bound=(10, len(df)))

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()   # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render_all(title="Random Agent", save_path="result.png")
```

---

## Training with Stable Baselines 3

```bash
python train.py --ticker GOOGL --timesteps 100000 --save-fig results.png
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--ticker` | `GOOGL` | Yahoo Finance ticker symbol |
| `--start` | `2010-01-01` | Data download start |
| `--end` | `2023-12-31` | Data download end |
| `--train-end` | `2021-12-31` | Train/test split date |
| `--timesteps` | `50000` | PPO training timesteps |
| `--window` | `10` | Observation window size |
| `--save-fig` | `results.png` | Where to save the result plot |

---

## Visualisation (AminHP-style `render_all`)

After an episode, call `env.render_all()` to produce a two-panel figure:

- **Top panel:** Close price with position markers  
  - 🟢 green ▲ = Long  
  - 🔵 blue ● = Flat  
  - 🔴 red ▼ = Short
- **Bottom panel:** Accumulated profit ratio curve

```python
env.render_all(title="PPO Agent – Test Set", save_path="result.png")
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
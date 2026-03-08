# Architektur – vbt_vc Trading Platform

---

## Überblick

vbt_vc ist eine modulare Trading-Plattform mit drei unabhängigen, austauschbaren Schichten:

```
┌─────────────────────────────────────────────────────────────┐
│                      Datenquellen                           │
│         ccxt (Krypto/Futures/Forex) · yfinance (Aktien)     │
└────────────────────────┬────────────────────────────────────┘
                         │ OHLCV DataFrame (pandas)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    data/  – Datenschicht                    │
│   BaseFetcher (abstrakt) → CCXTFetcher / YFinanceFetcher    │
│   Caching: Parquet-Dateien lokal                            │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐        ┌─────────────────────────────────┐
│  trading_env/    │        │       backtest/                 │
│  RL-Environment  │        │  vectorbt-Portfolio-Simulation  │
│  (gymnasium)     │        │  Metriken: Sharpe, Drawdown,    │
│  PPO-Training    │        │  Win-Rate, Total Return         │
└──────────┬───────┘        └────────────────┬────────────────┘
           │                                 │
           │ trainiertes Modell              │ validierte Strategie
           └──────────────┬──────────────────┘
                          ▼
              ┌───────────────────────┐
              │        live/          │
              │  BrokerBase (abstrakt)│
              │  CCXTBroker (Live)    │
              │  PaperBroker (Test)   │
              │  RiskManager          │
              └───────────────────────┘
```

---

## Verzeichnisstruktur (Zielzustand)

```
vbt_vc/
│
├── data/                        # Datenschicht
│   ├── __init__.py
│   ├── base_fetcher.py          # Abstrakte Basisklasse
│   ├── ccxt_fetcher.py          # ccxt: Krypto, Futures, Forex
│   └── yfinance_fetcher.py      # Fallback: Aktien, ETFs
│
├── trading_env/                 # RL-Environment
│   ├── __init__.py
│   ├── trading_env.py           # Abstrakte Basisklasse (DI-engine)
│   ├── stocks_env.py            # Legacy: Aktien via yfinance
│   └── universal_env.py         # Neu: alle Asset-Klassen
│
├── backtest/                    # Backtesting
│   ├── __init__.py
│   ├── vbt_backtest.py          # vectorbt-Integration
│   └── metrics.py               # Performance-Metriken
│
├── agent/                       # RL-Agenten (Wrapper)
│   ├── __init__.py
│   └── ppo_agent.py             # PPO via Stable Baselines 3
│
├── live/                        # Live-Trading
│   ├── __init__.py
│   ├── broker_base.py           # Abstrakte Broker-Schnittstelle
│   ├── ccxt_broker.py           # ccxt Live-Trading
│   ├── paper_broker.py          # Paper-Trading (kein echtes Geld)
│   ├── portfolio.py             # Positions- und Kapital-Tracking
│   └── risk_manager.py          # Stop-Loss, Max-Kapital-Limits
│
├── config/                      # Konfiguration
│   ├── default.yaml             # Standardwerte
│   └── .env.example             # API-Key-Vorlage (kein echtes Secret)
│
├── tests/                       # Test-Suite
│   ├── __init__.py
│   ├── test_trading_env.py      # ✅ vorhanden
│   ├── test_ccxt_fetcher.py     # ✅ vorhanden
│   ├── test_backtest.py         # ⬜ offen
│   └── test_live.py             # ⬜ offen
│
├── examples/                    # Jupyter-Notebooks
│   ├── backtest_demo.ipynb      # ⬜ offen
│   └── training_demo.ipynb      # ⬜ offen
│
├── train.py                     # ✅ CLI-Trainingsskript
├── backtest_run.py              # ⬜ CLI-Backtesting-Skript
├── live_run.py                  # ⬜ CLI-Live-Trading-Skript
├── requirements.txt
├── README.md
├── ARCHITECTURE.md              # dieses Dokument
├── REQUIREMENTS.md
├── TASKS.md
└── AGENT_ROLES.md
```

---

## Modulbeschreibungen

### `data/` – Datenschicht

**Verantwortung:** Historische und Echtzeit-OHLCV-Daten von externen Quellen abrufen, normalisieren und cachen.

```
BaseFetcher (abstrakt)
    │
    ├── CCXTFetcher          Exchange-Daten via ccxt
    │       ├── Binance      BTC/USDT, ETH/USDT, Futures
    │       ├── Bybit        Futures, Perpetuals
    │       ├── OKX          Futures, Spot, Forex
    │       └── + 100 weitere Exchanges
    │
    └── YFinanceFetcher      Aktien, ETFs (Yahoo Finance)
```

**Schnittstelle:**
```python
fetcher = CCXTFetcher(exchange_id='binance')
df: pd.DataFrame = fetcher.fetch_ohlcv(
    symbol='BTC/USDT',
    timeframe='1d',
    since='2022-01-01',
    until='2024-12-31',
    cache=True
)
# Rückgabe: DataFrame mit Spalten Open, High, Low, Close, Volume
```

**Caching:** Daten werden als `.parquet`-Dateien unter `data/cache/` gespeichert. Schlüssel: `{exchange}_{symbol}_{timeframe}_{since}_{until}.parquet`

---

### `trading_env/` – RL-Environment

**Verantwortung:** Gymnasium-kompatibles Environment für RL-Agenten-Training.

```
TradingEnv (abstrakt, gymnasium.Env)
    │
    ├── StocksEnv          Legacy – Aktien, 3 Features (Close, Diff, Vol)
    └── UniversalEnv       Neu – alle Assets, konfigurierbare Features
```

**State Machine (DI-engine):**

| Von \ Aktion | DOUBLE_SELL | SELL | HOLD | BUY | DOUBLE_BUY |
|---|---|---|---|---|---|
| SHORT | − | − | SHORT | FLAT | LONG |
| FLAT | SHORT | SHORT | FLAT | LONG | LONG |
| LONG | SHORT | FLAT | LONG | − | − |

**Reward:**
- Long: `log(close_curr / close_prev) + log(cost)`
- Short: `log(2 − close_curr / close_prev) + log(cost)`

---

### `backtest/` – Backtesting

**Verantwortung:** Historische Simulation von RL-Agenten-Signalen via vectorbt.

**Datenfluss:**
```
1. Daten: CCXTFetcher → OHLCV DataFrame
2. Agent: trainiertes Modell → Aktionssignale (0–4) pro Tick
3. Konvertierung: Aktionssignale → long/short Entry/Exit Boolean-Arrays
4. vectorbt: vbt.Portfolio.from_signals(entries, exits, ...)
5. Metriken: Sharpe, Drawdown, Return, Win-Rate
```

---

### `live/` – Live-Trading

**Verantwortung:** Ausführung von Orders auf echten oder simulierten Exchanges.

**Sicherheitsprinzip:** Paper-Trading ist Standard. Live-Trading muss explizit via `--live` aktiviert werden.

```python
# Paper-Trading (Standard)
broker = PaperBroker(initial_capital=10_000)

# Live-Trading (explizit)
broker = CCXTBroker(exchange_id='binance', api_key=..., secret=...)
```

**RiskManager:**
- `max_position_size`: max. Kapital pro Trade (z. B. 5 % des Portfolios)
- `daily_stop_loss`: täglicher Verlustlimit (z. B. −3 %)
- `leverage_limit`: maximaler Hebel (z. B. 3×)

---

## Datenfluss – vollständig

```
1. FETCH
   CCXTFetcher.fetch_ohlcv('BTC/USDT', '1d', '2020-01-01', '2024-01-01')
       └─▶ OHLCV DataFrame (gecacht)

2. TRAIN
   UniversalEnv(df, window_size=10, leverage=1.0)
       └─▶ PPO.learn(total_timesteps=200_000)
       └─▶ Modell gespeichert als models/ppo_BTC_USDT_1d.zip

3. BACKTEST
   VbtBacktest(df, model)
       └─▶ Signale generieren
       └─▶ vbt.Portfolio simulieren
       └─▶ Metriken ausgeben + Plot speichern

4. LIVE (nach Validierung)
   LiveRunner(model, broker=PaperBroker())
       └─▶ Echtzeit-Daten via CCXTFetcher (letzte Kerze)
       └─▶ Aktion berechnen
       └─▶ Order via Broker ausführen
       └─▶ Portfolio aktualisieren
```

---

## Abhängigkeitsgraph

```
data/           ← ccxt, pandas, python-dotenv
trading_env/    ← data/, gymnasium, numpy
backtest/       ← data/, trading_env/, vectorbt
agent/          ← trading_env/, stable-baselines3, torch
live/           ← data/, agent/, ccxt
```

---

## Sicherheitsarchitektur

```
.env                         ← API-Keys (NIEMALS in Git!)
    └─▶ python-dotenv
            └─▶ CCXTBroker  ← liest EXCHANGE_API_KEY, EXCHANGE_SECRET

.gitignore enthält:
    .env
    data/cache/
    models/
    logs/
```

---

## Konfiguration (YAML)

```yaml
# config/default.yaml
exchange: binance
symbol: BTC/USDT
timeframe: 1d
window_size: 20
leverage: 1.0
fees:
  bid: 0.001
  ask: 0.001
risk:
  max_position_pct: 0.05    # 5% des Portfolios pro Trade
  daily_stop_loss_pct: 0.03 # 3% tägliches Verlustlimit
  leverage_limit: 3.0
training:
  timesteps: 200000
  algorithm: ppo
  checkpoint_freq: 10000
```

# Anforderungen – vbt_vc Trading Platform

Dieses Dokument beschreibt alle funktionalen und nicht-funktionalen Anforderungen des Projekts.

> **Stand:** 2026-03-08 – überarbeitet nach Projektzieldefinition
> **Projektziel:** Echtes, produktives Trading-System für Futures, Kryptowährungen und Forex
> **Betrieb:** Erst historisches Backtesting (vectorbt) → Validierung → Live-Trading

---

## 1. Projektziel

Entwicklung einer persönlichen Trading-Plattform, die:

1. **Reinforcement-Learning-Agenten** (RL) trainiert und evaluiert,
2. **Backtesting** über vectorbt mit historischen OHLCV-Daten durchführt,
3. **Live-Trading** über Broker-APIs (ccxt-kompatible Exchanges) ausführt,
4. **Futures, Kryptowährungen und Forex** als Asset-Klassen unterstützt.

---

## 2. Systemarchitektur (Übersicht)

```
Datenquellen (ccxt, vbt-Daten)
        │
        ▼
  data/ – Datenabruf & Normalisierung
        │
        ├──▶ trading_env/ – Gymnasium-RL-Environment (Training)
        │           │
        │           ▼
        │      agent/ – RL-Agenten (PPO, SAC, ...)
        │
        ├──▶ backtest/ – vectorbt-Backtesting
        │
        └──▶ live/ – Live-Trading (Broker-API)
```

---

## 3. Funktionale Anforderungen

### 3.1 Datenschicht (`data/`)

| ID | Anforderung |
|----|-------------|
| F-01 | Es muss eine abstrakte Basisklasse `BaseFetcher` existieren, die die Schnittstelle für alle Datenquellen definiert. |
| F-02 | Ein `CCXTFetcher` muss OHLCV-Daten von ccxt-kompatiblen Exchanges abrufen können (Binance, Bybit, OKX, usw.). |
| F-03 | Der Fetcher unterstützt konfigurierbare Symbole (z. B. `BTC/USDT`), Timeframes (`1m`, `1h`, `1d`) und Zeiträume. |
| F-04 | Rückgabe ist immer ein standardisierter `pandas.DataFrame` mit Spalten: `Open, High, Low, Close, Volume`. |
| F-05 | Daten können optional lokal gecacht werden (Parquet/CSV), um unnötige API-Calls zu vermeiden. |
| F-06 | Der Fetcher muss Futures-Symbole (z. B. `BTC/USDT:USDT`) und Spot-Symbole unterstützen. |
| F-07 | Forex-Daten (z. B. via ccxt-Forex-Exchanges oder OANDA) müssen über dieselbe Schnittstelle abrufbar sein. |

### 3.2 RL-Environment (`trading_env/`)

| ID | Anforderung |
|----|-------------|
| F-08 | Das Environment muss 3 Positionen unterstützen: **SHORT** (−1), **FLAT** (0), **LONG** (+1). |
| F-09 | Es müssen 5 Aktionen definiert sein: **DOUBLE_SELL** (0), **SELL** (1), **HOLD** (2), **BUY** (3), **DOUBLE_BUY** (4). |
| F-10 | Die Zustandsübergänge müssen der DI-engine-Übergangstabelle entsprechen. |
| F-11 | Der Observation Space ist ein flacher Vektor: `window_size × n_features` (normalisiert) + Position + Tick-Ratio. |
| F-12 | Die logarithmische Reward-Funktion (DI-engine) muss erhalten bleiben. |
| F-13 | Das Environment muss die `gymnasium`-API implementieren und SB3 `check_env()` bestehen. |
| F-14 | Das Environment muss mit beliebigen OHLCV-DataFrames arbeiten (nicht nur Aktien). |
| F-15 | Leverage (Hebelwirkung) muss als konfigurierbarer Parameter unterstützt werden (relevant für Futures). |
| F-16 | Rollover-Kosten für Futures müssen in der Reward-Funktion berücksichtigt werden können. |

### 3.3 Backtesting (`backtest/`)

| ID | Anforderung |
|----|-------------|
| F-17 | vectorbt muss als primäres Backtesting-Framework integriert werden. |
| F-18 | RL-Agenten-Signale (Kauf/Verkauf) müssen in vectorbt-kompatible Portfolios überführt werden. |
| F-19 | Das Backtesting muss konfigurierbare Transaktionskosten (Bid/Ask-Fees, Slippage) unterstützen. |
| F-20 | Ergebnisse müssen gängige Metriken liefern: Sharpe Ratio, Max Drawdown, Total Return, Win Rate. |
| F-21 | Backtesting-Läufe müssen reproduzierbar sein (festgelegter Seed, versionierte Daten). |
| F-22 | Vergleich mehrerer Strategien (Buy-and-Hold vs. RL-Agent) muss möglich sein. |

### 3.4 RL-Training (`agent/`, `train.py`)

| ID | Anforderung |
|----|-------------|
| F-23 | PPO (Stable Baselines 3) ist der primäre Algorithmus. |
| F-24 | Weitere Algorithmen (SAC, A2C, TD3) müssen einfach austauschbar sein. |
| F-25 | Trainingsparameter sind über CLI-Argumente konfigurierbar. |
| F-26 | Modell-Checkpoints werden regelmäßig während des Trainings gespeichert. |
| F-27 | TensorBoard-Logging ist aktiv. |
| F-28 | Hyperparameter-Optimierung via Optuna muss vorbereitet sein (optionale Abhängigkeit). |

### 3.5 Live-Trading (`live/`)

| ID | Anforderung |
|----|-------------|
| F-29 | Eine abstrakte `BrokerBase`-Klasse definiert die Schnittstelle für alle Broker. |
| F-30 | Eine ccxt-basierte Implementierung ermöglicht Handel auf ccxt-kompatiblen Exchanges. |
| F-31 | Paper-Trading-Modus (Simulation mit Echtzeit-Daten, ohne echtes Geld) muss unterstützt werden. |
| F-32 | Order-Management: Market Orders, Limit Orders, Stop-Loss müssen unterstützt werden. |
| F-33 | Positions- und Portfolio-Tracking in Echtzeit. |
| F-34 | Risikomanagement: konfigurierbares maximales Kapital pro Trade, täglicher Verlustlimit. |

### 3.6 Visualisierung

| ID | Anforderung |
|----|-------------|
| F-35 | `render_all()` erzeugt ein zweiteiliges Diagramm (Kurs + Profit). |
| F-36 | Backtesting-Ergebnisse werden als vectorbt-Reports exportiert. |
| F-37 | Performance-Metriken werden tabellarisch ausgegeben. |

---

## 4. Nicht-funktionale Anforderungen

| ID | Anforderung |
|----|-------------|
| NF-01 | Kompatibilität mit **Python ≥ 3.10**. |
| NF-02 | Alle Abhängigkeiten sind in `requirements.txt` versioniert. |
| NF-03 | Jede Komponente hat eine eigene Test-Suite (pytest). |
| NF-04 | Bei jeder Code-Generierung durch ein LLM ist der Developer/Kritiker-Workflow (`AGENT_ROLES.md`) einzuhalten. |
| NF-05 | Konfiguration (Exchange, Symbol, Timeframe, etc.) erfolgt über YAML-Dateien oder CLI-Parameter – **kein Hardcoding**. |
| NF-06 | Secrets (API-Keys) werden **niemals** im Code oder in Git gespeichert – ausschließlich via `.env`-Dateien oder Umgebungsvariablen. |
| NF-07 | Der Code ist modular: Datenschicht, Environment, Backtest und Live-Trading sind vollständig entkoppelt. |
| NF-08 | Lokales Caching von Marktdaten verhindert unnötige API-Calls. |

---

## 5. Abhängigkeiten

| Paket | Mindestversion | Zweck |
|-------|---------------|-------|
| `gymnasium` | ≥ 0.26.0 | Standard-RL-Environment-API |
| `stable-baselines3` | ≥ 2.0.0 | PPO und weitere RL-Algorithmen |
| `numpy` | ≥ 1.24.0 | Numerische Berechnungen |
| `pandas` | ≥ 2.0.0 | Datenverarbeitung |
| `ccxt` | ≥ 4.0.0 | Marktdaten + Live-Trading für Krypto, Futures, Forex |
| `vectorbt` | ≥ 0.26.0 | Backtesting-Framework |
| `matplotlib` | ≥ 3.7.0 | Visualisierung |
| `torch` | ≥ 2.0.0 | Neural-Network-Backend für SB3 |
| `python-dotenv` | ≥ 1.0.0 | API-Keys sicher aus `.env` laden |
| `optuna` | ≥ 3.0.0 | Hyperparameter-Optimierung (optional) |
| `yfinance` | ≥ 0.2.30 | Historische Aktien-/ETF-Daten (Fallback) |

---

## 6. Sicherheitsanforderungen

| ID | Anforderung |
|----|-------------|
| S-01 | API-Keys und Secrets werden nur über Umgebungsvariablen übergeben (`.env` + `python-dotenv`). |
| S-02 | `.env`-Dateien sind in `.gitignore` eingetragen. |
| S-03 | Beim Live-Trading muss ein Risiko-Limit (max. Kapital pro Trade, tägliches Stop-Loss) konfigurierbar sein. |
| S-04 | Paper-Trading-Modus ist die Standardeinstellung – Live-Trading muss explizit aktiviert werden. |

---

## 7. Quellenreferenzen

- **DI-engine State Machine & Reward**: [opendilab/DI-engine – dizoo/gym_anytrading](https://github.com/opendilab/DI-engine/tree/main/dizoo/gym_anytrading/envs)
- **Visualisierung**: [AminHP/gym-anytrading](https://github.com/AminHP/gym-anytrading)
- **RL-Training**: [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- **Backtesting**: [vectorbt](https://vectorbt.dev/)
- **Exchange-API**: [ccxt](https://github.com/ccxt/ccxt)
